import torch
from torch.utils.data import Dataset

class GestureSequenceDataset(Dataset):
    """
    Dataset para secuencias completas ya preprocesadas y paddeadas.
    X_imu: (N, max_len, 6)
    X_thm: (N, max_len, 5)
    X_tof: (N, max_len, 320)
    y: (N,)
    """
    def __init__(self, X_imu, X_thm, X_tof, y):
        self.X_imu = X_imu
        self.X_thm = X_thm
        self.X_tof = X_tof
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        imu = torch.from_numpy(self.X_imu[idx]).float()        # (time,6)
        imu = imu.transpose(0,1)                               # → (6, time)

        thm = torch.from_numpy(self.X_thm[idx]).float()        # (time,5)
        tof = torch.from_numpy(self.X_tof[idx]).float()        # (time,320)

        label = torch.tensor(self.y[idx]).long()

        return imu, thm, tof, label
