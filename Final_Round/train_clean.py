import os
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from schedulefree import RAdamScheduleFree
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from torchinfo import summary

from CMI_data_set import MyDataset
from multi_branch_classifier import MultiBranchClassifier
from seed_everything import seed_everything
from config import Config
from Preprocessing import Preprocessor



TARGET_GESTURES = [
    "Above ear - pull hair",
    "Cheek - pinch skin",
    "Eyebrow - pull hair",
    "Eyelash - pull hair",
    "Forehead - pull hairline",
    "Forehead - scratch",
    "Neck - pinch skin",
    "Neck - scratch",
]

NON_TARGET_GESTURES = [
    "Write name on leg",
    "Wave hello",
    "Glasses on/off",
    "Text on phone",
    "Write name in air",
    "Feel around in tray and pull out an object",
    "Scratch knee/leg skin",
    "Pull air toward your face",
    "Drink from bottle/cup",
    "Pinch knee/leg skin",
]

ALL_GESTURES = TARGET_GESTURES + NON_TARGET_GESTURES
GESTURE_TO_IDX = {gesture: idx for idx, gesture in enumerate(ALL_GESTURES)}
N_TARGET = len(TARGET_GESTURES)
N_GESTURES = len(ALL_GESTURES)

# Model hyperparameters
LAMBDA_BIN = 1.0
UNET_WEIGHT = 0.0  # Set to 0.1 for explicit U-Net supervision



def load_and_preprocess_data(config: Config):
    """Load and preprocess training data."""
    print("Loading data...")
    df = pl.read_csv("train.csv")
    df_demo = pl.read_csv("train_demographics.csv")
    df = df.join(df_demo, on="subject", how="left")
    
    # Mark subjects that need rotation
    subjects_rotated = ["SUBJ_019262", "SUBJ_045235"]
    df = df.with_columns(
        pl.col("subject").is_in(subjects_rotated).cast(pl.Int16).alias("rotate")
    )
    
    # Preprocess
    pp = Preprocessor()
    df = pp.preprocess(df)
    
    return df, pp


def prepare_features(df: pl.DataFrame, pp: Preprocessor, config: Config):
    """Extract and organize feature arrays."""
    imu_feature_blocks = [
        ["acc_x", "acc_y", "acc_z"],
        ["rot_w", "rot_x", "rot_y", "rot_z"],
        ["linear_acc_x", "linear_acc_y", "linear_acc_z", "global_acc_x", "global_acc_y", "global_acc_z"],
        ["rotvec_diff_x", "rotvec_diff_y", "rotvec_diff_z"],
    ]
    
    thm_names = [f"thm_{i}" for i in range(1, 6)]
    
    # Build complete feature list
    list_features = (
        sum(imu_feature_blocks, [])
        + thm_names
        + [f"tof_{i}_v{j}" for i in range(1, 6) for j in range(64)]
    )
    
    # Extract features
    X_all = pp.get_feature_array(df, list_features, seq_len=config.seq_len, fit=True)
    
    # Create validity mask
    mask_valid = (X_all != 0).max(axis=1, keepdims=True).astype(np.float32)
    
    # Extract gesture segment ground truth
    gesture_segment_true = extract_gesture_segments(df, config.seq_len, mask_valid)
    
    # Split features by type
    thm_indices = [list_features.index(name) for name in thm_names]
    X_thm_raw = X_all[:, thm_indices, :]
    X_thm = np.transpose(X_thm_raw, (0, 2, 1)).astype(np.float32)  # (N, T, 5)
    
    X_no_thm = np.delete(X_all, thm_indices, axis=1)
    X_imu = X_no_thm[:, :-320].astype(np.float32)  # (N, C_imu, T)
    X_tof = X_no_thm[:, -320:]
    X_tof = np.transpose(X_tof, (0, 2, 1)).astype(np.float32)  # (N, T, 320)
    
    print(f"X (IMU) shape: {X_imu.shape}")
    print(f"X_tof shape: {X_tof.shape}")
    print(f"X_thm shape: {X_thm.shape}")
    print(f"gesture_segment_true shape: {gesture_segment_true.shape}")
    print(f"mask_valid shape: {mask_valid.shape}")
    
    return {
        'X_imu': X_imu,
        'X_tof': X_tof,
        'X_thm': X_thm,
        'gesture_segment_true': gesture_segment_true,
        'mask_valid': mask_valid,
        'imu_feature_blocks': imu_feature_blocks,
    }


def extract_gesture_segments(df: pl.DataFrame, seq_len: int, mask_valid: np.ndarray):
    """Extract gesture segment labels."""
    gesture_segment = (
        df.sort(["sequence_id", "sequence_counter"], descending=[False, True])
        .group_by("sequence_id", maintain_order=True)
        .head(seq_len)
        .with_columns(pl.col("sequence_counter").cum_count().over("sequence_id"))
        .with_columns((pl.col("phase") == "Gesture").cast(pl.Int8).alias("gesture_segment"))
        .pivot(on="sequence_counter", index="sequence_id", values="gesture_segment")
        .fill_null(0)
        .drop("sequence_id")
        .to_numpy()
    )
    
    gesture_segment = gesture_segment[:, np.newaxis, :seq_len].astype(np.float32)
    return gesture_segment * mask_valid


def prepare_labels(df: pl.DataFrame):
    """Extract gesture and orientation labels."""
    orientations = sorted(df["orientation"].unique())
    orientation_to_idx = {v: i for i, v in enumerate(orientations)}
    
    Y_gesture = (
        df.select(["sequence_id", "gesture"])
        .unique(maintain_order=True)
        .select(pl.col("gesture").replace(GESTURE_TO_IDX).cast(pl.Int8))
        .to_series()
        .to_numpy()
    )
    
    Y_orientation = (
        df.select(["sequence_id", "orientation"])
        .unique(maintain_order=True)
        .select(pl.col("orientation").replace(orientation_to_idx).cast(pl.Int8))
        .to_series()
        .to_numpy()
    )
    
    groups = (
        df.select(["sequence_id", "subject"])
        .unique(maintain_order=True)
        .select("subject")
        .to_series()
        .to_numpy()
    )
    
    return Y_gesture, Y_orientation, groups, len(orientations)



def create_model(imu_feature_blocks, n_orientations, config: Config):
    """Create and initialize the model."""
    in_channels = [len(block) for block in imu_feature_blocks]
    out_dim = N_GESTURES + n_orientations
    
    model = MultiBranchClassifier(
        number_imu_blocks=len(imu_feature_blocks),
        in_channels=in_channels,
        out_channels=out_dim,
        initial_channels_per_feature=16,
        cnn1d_channels=(128, 256),
        cnn1d_kernel_size=3,
        ToF_out_channels=32,
        ToF_kernel_size=3,
        THM_out_channels=16,
        THM_kernel_size=3,
        mlp_dropout=0.5,
        lstm_hidden=128,
        unet_in_channels=sum(in_channels),
    )
    
    return model


def train_epoch(model, loader, optimizer, device, epoch, lambda_bin=LAMBDA_BIN, unet_weight=UNET_WEIGHT):
    """Train for one epoch."""
    model.train()
    optimizer.train()
    
    total_loss = 0.0
    n_batches = 0
    
    for batch in loader:
        # Unpack batch
        if len(batch) == 5:
            x, x_tof, x_thm, y_g, y_o = batch
            seg_true = None
            mask_valid = None
        else:
            x, x_tof, x_thm, y_g, y_o, seg_true, mask_valid = batch
        
        # Move to device
        x = x.to(device)
        x_tof = x_tof.to(device)
        x_thm = x_thm.to(device)
        y_g = y_g.to(device)
        y_o = y_o.to(device)
        
        if seg_true is not None:
            seg_true = seg_true.to(device)
        if mask_valid is not None:
            mask_valid = mask_valid.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        preds_stack, gesture_mask_logits = model(x, x_tof, x_thm)
        logits = preds_stack.mean(dim=1)
        
        logits_g = logits[:, :N_GESTURES]
        logits_o = logits[:, N_GESTURES:]
        
        # Compute losses
        loss_g = compute_gesture_loss(logits_g, y_g, lambda_bin)
        loss_o = nn.CrossEntropyLoss()(logits_o, y_o)
        
        loss = loss_g + 0.1 * loss_o  # config.aux_loss_weight
        
        # Add U-Net loss if enabled
        if unet_weight > 0.0 and seg_true is not None:
            unet_loss = model.compute_unet_loss(gesture_mask_logits, seg_true, mask_valid)
            loss = loss + unet_weight * unet_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / n_batches
    print(f"[Epoch {epoch}] Train loss: {avg_loss:.4f}")
    
    return avg_loss


def compute_gesture_loss(logits_g, y_g, lambda_bin, label_smoothing=0.0):
    """Compute combined gesture classification loss."""
    bce_bin = nn.BCEWithLogitsLoss()
    
    # Collapse non-target gestures into one bucket for macro part
    y_collapsed = torch.clamp(y_g, max=N_TARGET)
    
    logits_target = logits_g[:, :N_TARGET]
    logits_nontarget = torch.logsumexp(logits_g[:, N_TARGET:], dim=1, keepdim=True)
    logits_collapsed = torch.cat([logits_target, logits_nontarget], dim=1)
    
    ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    loss_macro = ce_loss(logits_collapsed, y_collapsed)
    
    # Binary classification: target vs non-target
    logit_t = torch.logsumexp(logits_target, dim=1)
    logit_n = torch.logsumexp(logits_g[:, N_TARGET:], dim=1)
    bin_logit = logit_t - logit_n
    y_bin = (y_g < N_TARGET).float()
    loss_bin = bce_bin(bin_logit, y_bin)
    
    return loss_macro + lambda_bin * loss_bin


@torch.inference_mode()
def validate(model, loader, device):
    """Validate model and return logits."""
    model.eval()
    
    all_logits = []
    all_bin_logits = []
    
    for batch in loader:
        x, x_tof, x_thm = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        
        preds_stack, _ = model(x, x_tof, x_thm)
        logits = preds_stack.mean(dim=1)
        logits_g = logits[:, :N_GESTURES]
        
        # Compute binary logit
        logit_t = torch.logsumexp(logits_g[:, :N_TARGET], dim=1)
        logit_n = torch.logsumexp(logits_g[:, N_TARGET:], dim=1)
        bin_logit = logit_t - logit_n
        
        all_logits.append(logits_g.cpu().numpy())
        all_bin_logits.append(bin_logit.cpu().numpy())
    
    return np.concatenate(all_logits), np.concatenate(all_bin_logits)



def predict_with_threshold(logits_g, bin_logit, threshold):
    """Predict gesture indices using threshold."""
    p_target = 1.0 / (1.0 + np.exp(-bin_logit))
    pred = np.zeros(len(p_target), dtype=np.int64)
    
    # Target gestures
    is_target = p_target >= threshold
    pred[is_target] = np.argmax(logits_g[is_target, :N_TARGET], axis=1)
    
    # Non-target gestures
    not_target = ~is_target
    pred[not_target] = N_TARGET + np.argmax(logits_g[not_target, N_TARGET:], axis=1)
    
    return pred


def compute_metric(y_true, y_pred):
    """Compute competition metric (average of binary and macro F1)."""
    f1_binary = f1_score(
        y_true < N_TARGET,
        y_pred < N_TARGET,
        average="binary"
    )
    
    f1_macro = f1_score(
        y_true.clip(max=N_TARGET),
        y_pred.clip(max=N_TARGET),
        average="macro"
    )
    
    return (f1_binary + f1_macro) / 2


def find_best_threshold(y_true, logits, bin_logit, thresholds=None):
    """Find optimal threshold for predictions."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    
    best_score = 0.0
    best_threshold = 0.5
    
    for thr in thresholds:
        y_pred = predict_with_threshold(logits, bin_logit, thr)
        score = compute_metric(y_true, y_pred)
        
        if score > best_score:
            best_score = score
            best_threshold = float(thr)
    
    return best_threshold, best_score


def main():
    """Main training function."""
    # Initialize
    seed_everything(0)
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Paths
    model_dir = Path("./")
    model_dir.mkdir(exist_ok=True)
    
    # Load and prepare data
    df, preprocessor = load_and_preprocess_data(config)
    features = prepare_features(df, preprocessor, config)
    Y_gesture, Y_orientation, groups, n_orientations = prepare_labels(df)
    
    # Save preprocessor
    with open(model_dir / "preprocessor.pickle", "wb") as f:
        pickle.dump(preprocessor, f)
    
    # Initialize cross-validation
    sgkf = StratifiedGroupKFold(n_splits=config.n_splits, shuffle=True, random_state=0)
    
    # OOF prediction storage
    logits_oof = np.zeros((config.n_epochs, len(Y_gesture), N_GESTURES), dtype=np.float32)
    binlogit_oof = np.zeros((config.n_epochs, len(Y_gesture)), dtype=np.float32)
    
    # Cross-validation loop
    for fold, (idx_train, idx_valid) in enumerate(sgkf.split(features['X_imu'], Y_gesture, groups)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold}")
        print(f"{'='*50}")
        
        # Create datasets
        ds_train = MyDataset(
            features['X_imu'][idx_train],
            features['X_tof'][idx_train],
            features['X_thm'][idx_train],
            y_gesture=Y_gesture[idx_train],
            y_orientation=Y_orientation[idx_train],
            gesture_segment_true=features['gesture_segment_true'][idx_train],
            mask_valid=features['mask_valid'][idx_train],
        )
        
        ds_valid = MyDataset(
            features['X_imu'][idx_valid],
            features['X_tof'][idx_valid],
            features['X_thm'][idx_valid],
            y_gesture=Y_gesture[idx_valid],
            y_orientation=Y_orientation[idx_valid],
            gesture_segment_true=features['gesture_segment_true'][idx_valid],
            mask_valid=features['mask_valid'][idx_valid],
        )
        
        # Create data loaders
        dl_train = DataLoader(
            ds_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        
        dl_valid = DataLoader(
            ds_valid,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        
        # Train multiple seeds
        for seed in range(config.n_seeds):
            seed_everything(seed)
            
            # Create model
            model = create_model(features['imu_feature_blocks'], n_orientations, config)
            
            # Print model summary for first fold/seed
            if fold == 0 and seed == 0:
                print(summary(
                    model=model,
                    input_size=[
                        (config.batch_size,) + features['X_imu'].shape[1:],
                        (config.batch_size,) + features['X_tof'].shape[1:],
                        (config.batch_size,) + features['X_thm'].shape[1:],
                    ],
                    col_names=["input_size", "output_size", "num_params"],
                    col_width=20,
                ))
            
            model.to(device)
            
            # Create optimizer
            optimizer = RAdamScheduleFree(
                model.parameters(),
                lr=config.lr,
                betas=config.betas,
                weight_decay=1e-5
            )
            
            print(f"\nFold-Seed: {fold}-{seed}")
            
            # Training loop
            for epoch in range(config.n_epochs):
                # Train
                train_epoch(model, dl_train, optimizer, device, epoch)
                
                # Validate
                logits_valid, binlogit_valid = validate(model, dl_valid, device)
                y_valid = Y_gesture[idx_valid]
                
                # Find best threshold
                best_thr, best_score = find_best_threshold(y_valid, logits_valid, binlogit_valid)
                
                # Baseline score (argmax)
                y_pred_base = np.argmax(logits_valid, axis=1)
                base_score = compute_metric(y_valid, y_pred_base)
                
                print(f"[Epoch {epoch}] Valid - Base: {base_score:.4f} | Best (thr={best_thr:.2f}): {best_score:.4f}")
                
                # Store OOF predictions
                logits_oof[epoch, idx_valid] = logits_valid
                binlogit_oof[epoch, idx_valid] = binlogit_valid
                
                # Save model checkpoints (last 2 epochs)
                if epoch >= config.n_epochs - 2:
                    checkpoint_path = model_dir / f"model_{fold}_{seed}_{epoch}.pth"
                    torch.save(model.state_dict(), checkpoint_path)
    
    # Find global best (epoch, threshold) on OOF
    print("\n" + "="*50)
    print("Finding best global OOF configuration...")
    print("="*50)
    
    best_oof = (0.0, None, None)
    for epoch in range(config.n_epochs):
        for thr in np.linspace(0.05, 0.95, 91):
            y_pred = predict_with_threshold(logits_oof[epoch], binlogit_oof[epoch], thr)
            score = compute_metric(Y_gesture, y_pred)
            
            if score > best_oof[0]:
                best_oof = (score, epoch, thr)
    
    print(f"\nBest OOF Score: {best_oof[0]:.4f}")
    print(f"Best Epoch: {best_oof[1]}")
    print(f"Best Threshold: {best_oof[2]:.3f}")


if __name__ == "__main__":
    main()