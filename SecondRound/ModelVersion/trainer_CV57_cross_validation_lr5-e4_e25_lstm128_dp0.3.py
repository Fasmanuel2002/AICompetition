import logging
import numpy as np
import torch
import torch.nn as nn
import mlflow
from typing import Optional
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold

from .gesture_secuence_dataset import GestureSequenceDataset
from ..models.multi_branch_classifier import MultiBranchClassifier
from ..config import GestureConfig
from ..utils.utils import RandomState
from torch.utils.tensorboard import SummaryWriter  
import time
from collections import Counter
import copy
import random

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

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

        # Número de clases
        self.num_classes = (
            len(GestureConfig.TARGET_GESTURES)
            + len(GestureConfig.NON_TARGET_GESTURES)
        )

        # Pesos inversamente proporcionales a la frecuencia de cada clase
        class_counts = np.bincount(self.y.astype(int), minlength=self.num_classes)
        class_weights = class_counts.sum() / (self.num_classes * class_counts)

        # Guardamos el criterio con pesos + label smoothing
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=0.1  # <-- mejora estabilidad
        )

        # StratifiedGroupKFold en vez de GroupKFold
        self.skf = StratifiedGroupKFold(
            n_splits=self.config.number_splits,
            shuffle=True,
            random_state=random_seed,
        )

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

            num_classes = self.num_classes

            oof_predictions = np.zeros((len(self.y), num_classes), dtype=np.float32)
            fold_scores = []

            print("\n===== CLASS DISTRIBUTION PER FOLD =====")
            for fold, (_, val_idx) in enumerate(
                self.skf.split(self.X_imu, self.y, groups=self.groups)
            ):
                print(f"Fold {fold}: {Counter(self.y[val_idx])}")
            print("=======================================\n")

            for seed in tqdm(range(self.config.number_seeds), desc="Random Seeds"):
                self._set_seed(seed)

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

                    # ===== Balanced Sampler (por fold) =====
                    train_labels = self.y[train_idx].astype(int)
                    class_sample_counts = np.bincount(train_labels, minlength=self.num_classes)
                    # evitar divisiones por 0
                    class_sample_counts = np.maximum(class_sample_counts, 1)
                    weights_per_class = 1.0 / class_sample_counts
                    sample_weights = weights_per_class[train_labels]
                    sample_weights_tensor = torch.from_numpy(sample_weights).float()

                    train_sampler = WeightedRandomSampler(
                        weights=sample_weights_tensor,
                        num_samples=len(sample_weights_tensor),
                        replacement=True
                    )

                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=self.config.batch_size,
                        shuffle=True,
                        num_workers=20,
                        pin_memory=True,
                        drop_last=False,
                    )

                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=self.config.batch_size,
                        shuffle=False,
                        num_workers=20,
                        pin_memory=True,
                        drop_last=False
                    )

                    # MODEL
                    model = MultiBranchClassifier(
                        number_imu_blocks=1,
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

                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=self.config.learning_rate,
                        weight_decay=1e-5
                    )

                   
                    warmup_epochs = max(1, int(0.1 * self.config.number_epochs))
                    cosine_epochs = self.config.number_epochs - warmup_epochs

                    warmup_scheduler = LinearLR(
                        optimizer,
                        start_factor=0.1,
                        end_factor=1.0,
                        total_iters=warmup_epochs
                    )
                    cosine_scheduler = CosineAnnealingLR(
                        optimizer,
                        T_max=cosine_epochs if cosine_epochs > 0 else 1
                    )

                    scheduler = SequentialLR(
                        optimizer,
                        schedulers=[warmup_scheduler, cosine_scheduler],
                        milestones=[warmup_epochs]
                    )

                    best_val_acc = 0.0
                    best_state_dict = None

                    # TRAINING
                    for epoch in tqdm(range(self.config.number_epochs), desc="Epochs"):
                        train_loss = self._train_epoch(model, optimizer, train_loader)
                        val_metrics = self._validate_epoch(model, val_loader)

                        # Step del scheduler (por epoch)
                        scheduler.step()

                        # TensorBoard Logging
                        writer.add_scalar(f"Seed{seed}/Fold{fold}/Train_Loss", train_loss, epoch)
                        writer.add_scalar(f"Seed{seed}/Fold{fold}/Val_Loss", val_metrics['loss'], epoch)
                        writer.add_scalar(f"Seed{seed}/Fold{fold}/Val_Acc", val_metrics['accuracy'], epoch)

                        mlflow.log_metrics({
                            f"seed{seed}_fold{fold}_train_loss": train_loss,
                            f"seed{seed}_fold{fold}_val_loss": val_metrics['loss'],
                            f"seed{seed}_fold{fold}_val_acc": val_metrics['accuracy'],
                        }, step=epoch)

                        # Guardar mejor modelo según val_acc
                        if val_metrics['accuracy'] > best_val_acc:
                            best_val_acc = val_metrics['accuracy']
                            best_state_dict = copy.deepcopy(model.state_dict())

                    # Cargar mejor modelo antes de predecir
                    if best_state_dict is not None:
                        model.load_state_dict(best_state_dict)

                    # SAVE OOF PREDICTIONS
                    preds = self._predict(model, val_loader)
                    oof_predictions[val_idx] = preds

                    fold_scores.append(best_val_acc)

            mlflow.log_metrics({
                "cv_mean_acc": np.mean(fold_scores),
                "cv_std": np.std(fold_scores)
            })

            writer.close()
            return oof_predictions, np.mean(fold_scores)

    def _train_epoch(self, model, optimizer, loader):
        model.train()
        total_loss = 0.0

        for imu, thm, tof, labels in loader:
            imu, thm, tof, labels = imu.to(device), thm.to(device), tof.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imu, tof, thm)
            loss = self.criterion(outputs, labels)   
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("===============================================================")
        print(f"For the Training Total loss per epoch: {total_loss / len(loader)}")
        return total_loss / len(loader)

    def _validate_epoch(self, model, loader):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for imu, thm, tof, labels in loader:
                imu, thm, tof, labels = imu.to(device), thm.to(device), tof.to(device), labels.to(device)

                outputs = model(imu, tof, thm)
                loss = self.criterion(outputs, labels)   

                total_loss += loss.item()

                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(loader)
        acc = 100 * correct / total

        print(f"For the Validate Total loss per epoch: {avg_loss}")
        print(f"For the Validate Accuracy per epoch: {acc}")
        print("=====================================================================")

        return {
            "loss": avg_loss,
            "accuracy": acc
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
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
