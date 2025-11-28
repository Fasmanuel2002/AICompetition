import sys
import os 

# Add project root to sys path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from src.datasets.gesture_dataset import GestureDataset
from src.datasets.preprocessor import Preprocessor
from src.trainers.trainer_cross_validation import CrossValidator
from src.models.multi_branch_classifier import MultiBranchClassifier
from src.config import GestureConfig, TrainingConfig
import pandas as pd


def prepare_data(train_path):
    """
    Prepare data for training

    Args:
        train_path (str): Path to training data CSV

    Returns:
        tuple: Prepared data tensors and groups
    """
    # Load data
    df = pd.read_csv(train_path)

    # Drop rows with any NaN values LOOK AT IT LATER
    df_clean = df.dropna()

    # Preprocessing
    preprocessor = Preprocessor()
    df_preprocessed = preprocessor.fit_transform(df_clean)

    # Map ALL gestures (target + non-target) to indices
    all_gestures = GestureConfig.TARGET_GESTURES + GestureConfig.NON_TARGET_GESTURES
    gesture_to_idx = {gesture: idx for idx, gesture in enumerate(all_gestures)}

    # Group by sequence_id to create 3D tensors
    sequences = []
    labels = []
    subjects = []

    for _, seq_data in df_preprocessed.groupby('sequence_id'):
        # Extract IMU features for this sequence: shape [time_steps, 6]
        imu_features = seq_data[['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z']].values

        # Extract THM features for this sequence: shape [time_steps, 5]
        thm_features = seq_data[['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']].values

        # Extract ToF features for this sequence: shape [time_steps, 320]
        tof_features = seq_data[[f'tof_{sensor}_v{pixel}'
                                 for sensor in range(1, 6)
                                 for pixel in range(64)]].values

        # Get label (same for all timesteps in a sequence)
        gesture = seq_data['gesture'].iloc[0]
        label = gesture_to_idx[gesture]

        # Get subject (for grouping in cross-validation)
        subject = seq_data['subject'].iloc[0]

        sequences.append({
            'imu': imu_features,
            'thm': thm_features,
            'tof': tof_features
        })
        labels.append(label)
        subjects.append(subject)

    # Find maximum sequence length for padding
    max_len = max(seq['imu'].shape[0] for seq in sequences)
    print(f"Maximum sequence length: {max_len}")
    print(f"Number of sequences: {len(sequences)}")

    # Pad sequences to max_len and convert to numpy arrays
    X_imu = np.zeros((len(sequences), max_len, 6))
    X_thm = np.zeros((len(sequences), max_len, 5))
    X_tof = np.zeros((len(sequences), max_len, 320))

    for i, seq in enumerate(sequences):
        seq_len = seq['imu'].shape[0]
        X_imu[i, :seq_len, :] = seq['imu']
        X_thm[i, :seq_len, :] = seq['thm']
        X_tof[i, :seq_len, :] = seq['tof']

    y = np.array(labels)
    groups = np.array(subjects)

    return X_imu, X_thm, X_tof, y, groups

def train_model():
    """
    Main training function
    """
    # Prepare data
    X_imu, X_thm, X_tof, y, groups = prepare_data('../src/datasets/train_df_clean.csv')

    # Configuration
    config = TrainingConfig(
        number_splits=3,
        number_seeds=1,
        number_epochs=15,
        learning_rate=1e-3,
        lstm_hidden=128,
        dropout_rate=0.3
    )

    # Calculate total number of gestures (target + non-target)
    num_classes = len(GestureConfig.TARGET_GESTURES) + len(GestureConfig.NON_TARGET_GESTURES)

    # Cross-validation
    cross_validator = CrossValidator(
        config=config,
        X_imu=X_imu,
        X_thm=X_thm,
        X_tof=X_tof,
        y=y,
        groups=groups
    )

    # Perform cross-validation
    oof_predictions, mean_score = cross_validator.cross_validate()

    print(f"Mean CV Score: {mean_score}")

    # Optional: Train final model on full dataset
    final_model = MultiBranchClassifier(
        number_imu_blocks=1,
        in_channels=[6],
        out_channels=num_classes,
        initial_channels_per_feature=16,
        cnn1d_channels=(256,  256),
        cnn1d_kernel_size=3,
        ToF_out_channels=32,
        ToF_kernel_size=3,
        THM_out_channels=32,
        THM_kernel_size=3,
        lstm_hidden=config.lstm_hidden,
        mlp_dropout=config.dropout_rate
    )
    
    # Save final model or perform additional operations
    torch.save(final_model.state_dict(), 'final_model.pth')

if __name__ == '__main__':
    train_model()