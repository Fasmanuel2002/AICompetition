from dataclasses import dataclass
from typing import List

@dataclass
class GestureConfig:
    """
    Configuration for gesture classification
    Provides a centralized way to manage gesture categories
    """
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

    def all_gestures_in_dataset(self): 
        """
        Combines target and non-target gestures into a single list
        Returns:
            List of all gesture labels
        """
        return self.TARGET_GESTURES + self.NON_TARGET_GESTURES


    def dict_gestures_dataset(self):
        """
        Creates a mapping from gesture labels to integer indices
        Returns:
            Dictionary mapping gesture labels to indices
        """
        return {v : i for i, v in enumerate(self.all_gestures_in_dataset())}
    
    def num_gestures(self):
        """
        Returns the total number of gesture categories
        Returns:
            Integer count of gesture categories
        """
        return len(self.all_gestures_in_dataset())
    
@dataclass
class TrainingConfig:
    number_splits: int = 1
    number_seeds: int = 3
    number_epochs: int = 3
    learning_rate: float = 1e-3
    lstm_hidden: int = 128
    dropout_rate: float = 0.3
    batch_size: int = 32

    # Model architecture parameters
    initial_channels_per_feature: int = 16
    cnn1d_channels: List[int] = None
    cnn1d_kernel_size: int = 3
    tof_out_channels: int = 32
    tof_kernel_size: int = 3
    thm_out_channels: int = 32
    thm_kernel_size: int = 3

    def __post_init__(self):
        if self.cnn1d_channels is None:
            self.cnn1d_channels = [64, 128]
