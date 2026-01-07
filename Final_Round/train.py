import os
import pickle
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from schedulefree import RAdamScheduleFree
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary

from CMI_data_set import MyDataset
from multi_branch_classifier import MultiBranchClassifier
from seed_everything import seed_everything
from config import Config
from Preprocessing import Preprocessor

target_gestures = [
    "Above ear - pull hair",
    "Cheek - pinch skin",
    "Eyebrow - pull hair",
    "Eyelash - pull hair",
    "Forehead - pull hairline",
    "Forehead - scratch",
    "Neck - pinch skin",
    "Neck - scratch",
]
non_target_gestures = [
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
all_gestures = target_gestures + non_target_gestures
dict_gestures = {v: i for i, v in enumerate(all_gestures)}
N_TARGET = len(target_gestures)  # 8
N_GESTURES = len(all_gestures)   # 18
bce_bin = nn.BCEWithLogitsLoss()
config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
    #Seed everything
    seed_everything(0)
    
    INPUT_DIR = "code/src/models/scripts/data"
    MODEL_DIR = "./"
    OUTPUT_DIR = "./"
    
    df = pl.read_csv(f"train.csv")
    df_demo = pl.read_csv(f"train_demographics.csv")
    df = df.join(df_demo, on="subject", how="left")
    subjects_rotated = ["SUBJ_019262", "SUBJ_045235"]
    df = df.with_columns(pl.col("subject").is_in(subjects_rotated).cast(pl.Int16).alias("rotate"))
    
    pp = Preprocessor()
    df = pp.preprocess(df)
    
    n_imu_blocks = 4


    imu_feature_blocks = [
        ["acc_x", "acc_y", "acc_z"],
        ["rot_w", "rot_x", "rot_y", "rot_z"],
        ["linear_acc_x", "linear_acc_y", "linear_acc_z", "global_acc_x", "global_acc_y", "global_acc_z"],
        ["rotvec_diff_x", "rotvec_diff_y", "rotvec_diff_z"],
    ]

    thm_names = [f"thm_{i}" for i in range(1, 6)]  # thm_1..thm_5


    list_features = (
        sum(imu_feature_blocks, [])
        + thm_names
        + [f"tof_{i}_v{j}" for i in range(1, 6) for j in range(64)]
    )
    X_all = pp.get_feature_array(df, list_features, seq_len=config.seq_len, fit=True)  # (N, C_total, T)
    
    mask_valid = (X_all != 0).max(axis=1, keepdims=True).astype(np.float32)
    
    gesture_segment_true = (
    df.sort(["sequence_id", "sequence_counter"], descending=[False, True])
    .group_by("sequence_id", maintain_order=True)
    .head(config.seq_len)
    .with_columns(pl.col("sequence_counter").cum_count().over("sequence_id"))
    .with_columns((pl.col("phase") == "Gesture").cast(pl.Int8).alias("gesture_segment"))
    .pivot(on="sequence_counter", index="sequence_id", values="gesture_segment")
    .fill_null(0)
    .drop("sequence_id")
    .to_numpy()
    )
    
    gesture_segment_true = gesture_segment_true[:, np.newaxis, : config.seq_len].astype(np.float32)
    gesture_segment_true = gesture_segment_true * mask_valid

    thm_indices = [list_features.index(name) for name in thm_names]
    X_thm_raw = X_all[:, thm_indices, :]                    # (N, 5, T)
    X_thm = np.transpose(X_thm_raw, (0, 2, 1)).astype(np.float32)  # (N, T, 5)
    X_no_thm = np.delete(X_all, thm_indices, axis=1)     

    X, X_tof = X_no_thm[:, :-320], X_no_thm[:, -320:]       # X: (N, C_imu, T), X_tof: (N, 320, T)
    X_tof = np.transpose(X_tof, (0, 2, 1)).astype(np.float32)  # (N, T, 320)
    
    X = X.astype(np.float32)

    print("X (IMU) shape:", X.shape)                        # (N, 16, T)
    print("X_tof shape:", X_tof.shape)                      # (N, T, 320)
    print("X_thm shape:", X_thm.shape)                      # (N, T, 5)
    print("gesture_segment_true shape:", gesture_segment_true.shape)  # (N, 1, T)
    print("mask_valid shape:", mask_valid.shape)            # (N, 1, T)

    with open(f"{MODEL_DIR}/preprocessor.pickle", "wb") as f:
        pickle.dump(pp, f)

    
    orientations = sorted(df["orientation"].unique())
    dict_orientations = {v: i for i, v in enumerate(orientations)}
    Y_main = (
    df.select(["sequence_id", "gesture"])
    .unique(maintain_order=True)
    .select(pl.col("gesture").replace(dict_gestures).cast(pl.Int8))
    .to_series()
    .to_numpy()
    )
    Y_aux = (
        df.select(["sequence_id", "orientation"])
        .unique(maintain_order=True)
        .select(pl.col("orientation").replace(dict_orientations).cast(pl.Int8))
        .to_series()
        .to_numpy()
    )
    Y = np.stack([Y_main, Y_aux], axis=1)
    groups = df.select(["sequence_id", "subject"]).unique(maintain_order=True).select("subject").to_series().to_numpy()

    print("Using:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # IMU channels are defined by imu_feature_blocks (THM is separate)
    in_channels = [len(b) for b in imu_feature_blocks]
    assert X.shape[1] == sum(in_channels), (X.shape, in_channels, sum(in_channels))

    out_dim = len(all_gestures) + len(orientations)

    sgkf = StratifiedGroupKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=0
    )

    # Store OOF logits for gestures (18) + bin_logit (for threshold tuning)
    logits_oof = np.zeros((config.n_epochs, len(Y_main), len(all_gestures)), dtype=np.float32)
    binlogit_oof = np.zeros((config.n_epochs, len(Y_main)), dtype=np.float32)

    for fold, (idx_train, idx_valid) in enumerate(sgkf.split(X, Y_main, groups)):
        print(f"\n================ FOLD {fold} ================")

        # ---------- TRAIN ----------
        ds_train = MyDataset(
            X[idx_train], X_tof[idx_train], X_thm[idx_train],
            y_gesture=Y_main[idx_train],
            y_orientation=Y_aux[idx_train],
            gesture_segment_true=gesture_segment_true[idx_train],
            mask_valid=mask_valid[idx_train],
        )
        dl_train = DataLoader(
            ds_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

        # ---------- VALID ----------
        ds_valid = MyDataset(
            X[idx_valid], X_tof[idx_valid], X_thm[idx_valid],
            y_gesture=Y_main[idx_valid],
            y_orientation=Y_aux[idx_valid],
            gesture_segment_true=gesture_segment_true[idx_valid],
            mask_valid=mask_valid[idx_valid],
        )
        dl_valid = DataLoader(
            ds_valid,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        for seed in range(config.n_seeds):
            seed_everything(seed)

            model = MultiBranchClassifier(
                number_imu_blocks=n_imu_blocks,
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
                unet_in_channels=X.shape[1],
            )

            if fold == 0 and seed == 0:
                print(
                    summary(
                        model=model,
                        input_size=[
                            (config.batch_size,) + X.shape[1:],      # x: (B, C, T)
                            (config.batch_size,) + X_tof.shape[1:],  # x_tof: (B, T, 320)
                            (config.batch_size,) + X_thm.shape[1:],  # x_thm: (B, T, 5)
                        ],
                        col_names=["input_size", "output_size", "num_params"],
                        col_width=20,
                    )
                )

            model.to(device)

            optimizer = RAdamScheduleFree(
                model.parameters(),
                lr=config.lr,
                betas=config.betas,
                weight_decay=1e-5
            )

            # You can tune these:
            LAMBDA_BIN = 1.0
            UNET_W = 0.0  # set to 0.1 if you want explicit U-Net supervision

            print(f"fold-seed: {fold}-{seed}")
            for epoch in range(config.n_epochs):
                # ---- TRAIN ----
                train_loss = train(model, dl_train, optimizer, device, epoch=epoch, lambda_bin=LAMBDA_BIN, unet_weight=UNET_W)

                # ---- VALID ----
                logits_valid, binlogit_valid = predict_logits(model, dl_valid, device)  # (N_valid,18), (N_valid,)
                y_valid = Y_main[idx_valid]

                base = metric_basic(y_valid, logits_valid)
                # quick threshold sweep (cheap): choose thr best on this fold+epoch
                best_thr = 0.5
                best_score = base
                for thr in np.linspace(0.05, 0.95, 19):
                    s = metric_with_threshold(y_valid, logits_valid, binlogit_valid, thr)
                    if s > best_score:
                        best_score = s
                        best_thr = float(thr)

                print(f"[Epoch {epoch}] valid metric base={base:.4f} | best_thr={best_thr:.2f} => {best_score:.4f}")

                # Save OOF (per-epoch, per-fold) into global buffers
                # NOTE: if you use multiple folds, later folds will overwrite indices (that's OK for OOF)
                logits_oof[epoch, idx_valid] = logits_valid
                binlogit_oof[epoch, idx_valid] = binlogit_valid

                # Save last 2 epochs
                if epoch >= config.n_epochs - 2:
                    torch.save(model.state_dict(), f"{MODEL_DIR}/model_{fold}_{seed}_{epoch}.pth")

    # ============================================================
    # Global OOF: pick best (epoch, thr)
    # ============================================================
    best = (-1.0, None, None)
    for epoch in range(config.n_epochs):
        for thr in np.linspace(0.05, 0.95, 91):
            sc = metric_with_threshold(Y_main, logits_oof[epoch], binlogit_oof[epoch], float(thr))
            if sc > best[0]:
                best = (sc, epoch, float(thr))

    print("\n✅ Best OOF:", best)  # (score, epoch, thr)



def train(model, loader, optimizer, device, epoch=None, lambda_bin=1.0, unet_weight=0.0):
    model.train()
    optimizer.train()  

    last_loss = None

    for batch in loader:
        
        if len(batch) == 5:
            x, x_tof, x_thm, y_g, y_o = batch
            seg_true = None
            mask_valid = None
        else:
            x, x_tof, x_thm, y_g, y_o, seg_true, mask_valid = batch

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

        # MultiBranchClassifier returns (predictions, gesture_mask_logits)
        preds_stack, gesture_mask_logits = model(x, x_tof, x_thm)  
        logits = preds_stack.mean(dim=1)                           

        logits_g = logits[:, :N_GESTURES]                          
        logits_o = logits[:, N_GESTURES:]                          

        # Gesture loss aligned with metric
        loss_g, loss_macro_proxy, loss_bin, _ = gesture_losses_from_logits(
            logits_g, y_g, label_smoothing=config.label_smoothing, lambda_bin=lambda_bin
        )

        # Orientation aux (helps representation)
        loss_o = nn.CrossEntropyLoss()(logits_o, y_o)

        loss = loss_g + config.aux_loss_weight * loss_o

        
        if (unet_weight > 0.0) and (seg_true is not None):
            unet_loss = model.compute_unet_loss(gesture_mask_logits, seg_true, mask_valid=mask_valid)
            loss = loss + unet_weight * unet_loss

        last_loss = loss.item()
        loss.backward()
        optimizer.step()

    if epoch is not None:
        print(f"[Epoch {epoch}] Train loss: {last_loss:.4f}")
    else:
        print(f"Train loss: {last_loss:.4f}")

    return last_loss

    
def gesture_losses_from_logits(logits_g18: torch.Tensor, y_g18: torch.Tensor, label_smoothing: float = 0.0, lambda_bin: float = 1.0):
    """
    logits_g18: (B, 18) over all gestures (targets first, then non-targets)
    y_g18:      (B,)   in [0..17]
    """
    # --- Collapse non-target into ONE bucket (index = 8) for macro part ---
    y9 = torch.clamp(y_g18, max=N_TARGET)  

    logits_target = logits_g18[:, :N_TARGET]                      
    logits_nont   = torch.logsumexp(logits_g18[:, N_TARGET:], dim=1, keepdim=True) 
    logits9 = torch.cat([logits_target, logits_nont], dim=1)       

    ce9 = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    loss_macro_proxy = ce9(logits9, y9)

    # --- Binary head derived from logits: target vs non-target ---
    logit_t = torch.logsumexp(logits_target, dim=1)               
    logit_n = torch.logsumexp(logits_g18[:, N_TARGET:], dim=1)    
    bin_logit = logit_t - logit_n                                 
    y_bin = (y_g18 < N_TARGET).float()                            
    loss_bin = bce_bin(bin_logit, y_bin)

    return loss_macro_proxy + lambda_bin * loss_bin, loss_macro_proxy.detach(), loss_bin.detach(), bin_logit.detach()
@torch.inference_mode()
def predict_logits(model, data_loader, device):
    logits_all = []
    bin_logit_all = []
    for batch in data_loader:
        if len(batch) >= 3:
            X, X_tof, X_thm = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        else:
            X, X_tof, X_thm = batch

        preds_stack, _ = model(X, X_tof, X_thm)
        logits = preds_stack.mean(dim=1)  # (B, out_dim)

        logits_g = logits[:, :N_GESTURES]

        # binary logit derived from gesture logits
        logit_t = torch.logsumexp(logits_g[:, :N_TARGET], dim=1)
        logit_n = torch.logsumexp(logits_g[:, N_TARGET:], dim=1)
        bin_logit = logit_t - logit_n

        logits_all.append(logits_g.cpu().numpy())
        bin_logit_all.append(bin_logit.cpu().numpy())

    return np.concatenate(logits_all, axis=0), np.concatenate(bin_logit_all, axis=0)

def pred_indices_from_logits(logits_g18: np.ndarray, bin_logit: np.ndarray, thr: float):
    """
    Returns indices in [0..17] (valid train gestures).
    Rule:
      - if p(target) < thr => choose best non-target gesture (argmax among non-target logits)
      - else => choose best target gesture (argmax among target logits)
    """
    p_target = 1.0 / (1.0 + np.exp(-bin_logit))  # sigmoid
    pred = np.zeros(len(p_target), dtype=np.int64)

    # target case
    is_target = p_target >= thr
    pred[is_target] = np.argmax(logits_g18[is_target, :N_TARGET], axis=1)

    # non-target case
    not_target = ~is_target
    pred[not_target] = N_TARGET + np.argmax(logits_g18[not_target, N_TARGET:], axis=1)

    return pred

def metric_basic(y, logits_g18):
    # original contest metric behavior using argmax + collapse
    f1_binary = f1_score(y < len(target_gestures), np.argmax(logits_g18, axis=1) < len(target_gestures), average="binary")
    f1_macro = f1_score(
        y.clip(max=len(target_gestures)), np.argmax(logits_g18, axis=1).clip(max=len(target_gestures)), average="macro"
    )
    return (f1_binary + f1_macro) / 2

def metric_with_threshold(y, logits_g18, bin_logit, thr):
    pred_idx = pred_indices_from_logits(logits_g18, bin_logit, thr)
    f1_binary = f1_score(y < len(target_gestures), pred_idx < len(target_gestures), average="binary")
    f1_macro  = f1_score(y.clip(max=len(target_gestures)), pred_idx.clip(max=len(target_gestures)), average="macro")
    return (f1_binary + f1_macro) / 2


@torch.inference_mode()
def predict_one_sequence(model, x_seq, x_tof_seq, x_thm_seq, device, thr: float):
    """Returns a gesture string that is guaranteed to be in train labels."""
    model.eval()
    x = torch.tensor(x_seq, dtype=torch.float32, device=device).unsqueeze(0)         # (1, C, T)
    x_tof = torch.tensor(x_tof_seq, dtype=torch.float32, device=device).unsqueeze(0) # (1, T, 320)
    x_thm = torch.tensor(x_thm_seq, dtype=torch.float32, device=device).unsqueeze(0) # (1, T, 5)

    preds_stack, _ = model(x, x_tof, x_thm)
    logits = preds_stack.mean(dim=1)          # (1, out_dim)
    logits_g = logits[:, :N_GESTURES].cpu().numpy()[0]

    logit_t = np.log(np.exp(logits_g[:N_TARGET]).sum() + 1e-9)
    logit_n = np.log(np.exp(logits_g[N_TARGET:]).sum() + 1e-9)
    bin_logit = logit_t - logit_n
    p_target = 1.0 / (1.0 + np.exp(-bin_logit))

    if p_target < thr:
        idx = N_TARGET + int(np.argmax(logits_g[N_TARGET:]))
    else:
        idx = int(np.argmax(logits_g[:N_TARGET]))

    return all_gestures[idx]

if __name__ == "__main__":
    main()