import logging
import numpy as np
import torch
import torch.nn as nn
import mlflow
from typing import Optional
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

from .gesture_secuence_dataset import GestureSequenceDataset
from ..models.multi_branch_classifier import MultiBranchClassifier
from ..config import GestureConfig
from ..utils.utils import RandomState
from torch.utils.tensorboard import SummaryWriter # type: ignore
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CrossValidator:
    def __init__(
        self, 
        config,
        X_imu,
        X_thm,
        X_tof,
        y,
        groups,
        random_seed: int = 42
    ):
        self.config = config
        self.X_imu = X_imu
        self.X_thm = X_thm
        self.X_tof = X_tof
        self.y = y
        self.groups = groups
        self.random_seed = random_seed

        self.skf = GroupKFold(n_splits=self.config.number_splits)

        self.logger = logging.getLogger(self.__class__.__name__)


    
    def cross_validate(self):
        writer = SummaryWriter(log_dir=f"runs/exp_{time.time()}")
        with mlflow.start_run():

            mlflow.log_params({
                "n_splits": self.config.number_splits,
                "n_seeds": self.config.number_seeds,
                "n_epochs": self.config.number_epochs,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size
            })

            num_classes = (
                len(GestureConfig.TARGET_GESTURES) +
                len(GestureConfig.NON_TARGET_GESTURES)
            )

            oof_predictions = np.zeros((len(self.y), num_classes), dtype=np.float32)
            fold_scores = []

            # Seeds
            for seed in tqdm(range(self.config.number_seeds), desc="Random Seeds"):
                self._set_seed(seed)

                # Folds
                for fold, (train_idx, val_idx) in tqdm(
                    enumerate(self.skf.split(self.X_imu, self.y, groups=self.groups)),
                    desc="Folds",
                    total=self.config.number_splits
                ):

                    # SUBSETS
                    train_dataset = GestureSequenceDataset(
                        self.X_imu[train_idx],
                        self.X_thm[train_idx],
                        self.X_tof[train_idx],
                        self.y[train_idx]
                    )

                    val_dataset = GestureSequenceDataset(
                        self.X_imu[val_idx],
                        self.X_thm[val_idx],
                        self.X_tof[val_idx],
                        self.y[val_idx]
                    )

                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=self.config.batch_size,
                        shuffle=True,
                        num_workers=2,
                        pin_memory=True
                    )

                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=self.config.batch_size,
                        shuffle=False,
                        num_workers=2,
                        pin_memory=True
                    )

                    # MODEL
                    model = MultiBranchClassifier(
                        number_imu_blocks=2,
                        in_channels=[3, 3],
                        out_channels=num_classes,
                        initial_channels_per_feature=self.config.initial_channels_per_feature,
                        cnn1d_channels=self.config.cnn1d_channels,
                        cnn1d_kernel_size=self.config.cnn1d_kernel_size,
                        ToF_out_channels=self.config.tof_out_channels,
                        ToF_kernel_size=self.config.tof_kernel_size,
                        THM_out_channels=self.config.thm_out_channels,
                        THM_kernel_size=self.config.thm_kernel_size,
                        lstm_hidden=self.config.lstm_hidden,
                        mlp_dropout=self.config.dropout_rate
                    ).to(device)

                    optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

                    # TRAINING
                    for epoch in tqdm(range(self.config.number_epochs), desc="Epochs"):
                        train_loss = self._train_epoch(model, optimizer, train_loader)
                        val_metrics = self._validate_epoch(model, val_loader)

                        #TensorBoard Logging
                        writer.add_scalar(f"Seed{seed}/Fold{fold}/Train_Loss", train_loss, epoch)
                        writer.add_scalar(f"Seed{seed}/Fold{fold}/Val_Loss", val_metrics['loss'], epoch)
                        writer.add_scalar(f"Seed{seed}/Fold{fold}/Val_Acc", val_metrics['accuracy'], epoch)

                        mlflow.log_metrics({
                            f"seed{seed}_fold{fold}_train_loss": train_loss,
                            f"seed{seed}_fold{fold}_val_loss": val_metrics['loss'],
                            f"seed{seed}_fold{fold}_val_acc": val_metrics['accuracy'],
                        }, step=epoch)

                    # SAVE OOF PREDICTIONS
                    preds = self._predict(model, val_loader)
                    oof_predictions[val_idx] = preds

                    fold_scores.append(val_metrics['accuracy'])

            mlflow.log_metrics({
                "cv_mean_acc": np.mean(fold_scores),
                "cv_std": np.std(fold_scores)
            })
            
            writer.close()
            return oof_predictions, np.mean(fold_scores)



    def _train_epoch(self, model, optimizer, loader):
        model.train()
        total_loss = 0

        for imu, thm, tof, labels in loader:
            imu, thm, tof, labels = imu.to(device), thm.to(device), tof.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imu, tof, thm)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)


    def _validate_epoch(self, model, loader):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for imu, thm, tof, labels in loader:
                imu, thm, tof, labels = imu.to(device), thm.to(device), tof.to(device), labels.to(device)

                outputs = model(imu, tof, thm)
                loss = nn.CrossEntropyLoss()(outputs, labels)

                total_loss += loss.item()

                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)

        return {
            "loss": total_loss / len(loader),
            "accuracy": 100 * correct / total
        }


    def _predict(self, model, loader):
        model.eval()
        all_preds = []

        with torch.no_grad():
            for imu, thm, tof, _ in loader:
                imu, thm, tof = imu.to(device), thm.to(device), tof.to(device)
                outputs = model(imu, tof, thm)
                all_preds.append(outputs.cpu().numpy())

        return np.vstack(all_preds)


    @staticmethod
    def _set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)