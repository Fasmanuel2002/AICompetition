from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):
    def __init__(self, X, X_tof, X_thm, y_gesture=None, y_orientation=None, gesture_segment_true=None, mask_valid=None):
        self.X     = torch.tensor(X, dtype=torch.float32)         # (N, C, T)
        self.X_tof = torch.tensor(X_tof, dtype=torch.float32)     # (N, T, 320)
        self.X_thm = torch.tensor(X_thm, dtype=torch.float32)     # (N, T, 5)

        self.has_labels = (y_gesture is not None) and (y_orientation is not None)

        if self.has_labels:
            self.y_gesture = torch.tensor(y_gesture, dtype=torch.long)       # (N,)
            self.y_orient  = torch.tensor(y_orientation, dtype=torch.long)   # (N,)

        # Optional: U-Net supervision
        self.has_segment = gesture_segment_true is not None
        if self.has_segment:
            self.gesture_segment_true = torch.tensor(gesture_segment_true, dtype=torch.float32)  # (N, 1, T)

        self.has_mask = mask_valid is not None
        if self.has_mask:
            self.mask_valid = torch.tensor(mask_valid, dtype=torch.float32)  # (N, 1, T)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if not self.has_labels:
            return self.X[idx], self.X_tof[idx], self.X_thm[idx]

        if self.has_segment and self.has_mask:
            return (self.X[idx], self.X_tof[idx], self.X_thm[idx],
                    self.y_gesture[idx], self.y_orient[idx],
                    self.gesture_segment_true[idx], self.mask_valid[idx])

        return self.X[idx], self.X_tof[idx], self.X_thm[idx], self.y_gesture[idx], self.y_orient[idx]