from torch.utils.data import DataLoader

class DataMapCallback:
    def __init__(self):
        self.epoch_data = []

    def on_epoch_end(self, epoch, logs=None):
        # Calculate training dynamics here
        # Append results to self.epoch_data
        pass

    @property
    def confidence(self):
        # Calculate and return confidence
        return ...

    @property
    def variability(self):
        # Calculate and return variability
        return ...

    @property
    def correctness(self):
        # Calculate and return correctness
        return ...
