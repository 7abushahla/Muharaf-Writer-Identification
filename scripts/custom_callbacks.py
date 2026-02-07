"""
Custom callbacks for Writer Identification Model
"""
import numpy as np
from tensorflow.keras.callbacks import Callback
from IPython.display import clear_output
from tqdm.keras import TqdmCallback


class ClearOutputEveryNEpochs(Callback):
    """Clears output every N epochs to reduce clutter in notebooks"""
    
    def __init__(self, n=10):
        super().__init__()
        self.n = n

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.n == 0:
            clear_output(wait=True)
            print(f"Cleared output at epoch {epoch + 1}")


class PeriodicModelCheckpoint(Callback):
    """Custom callback to save the model at periodic intervals (e.g., every N epochs)"""
    
    def __init__(self, filepath, save_freq_epochs=30, save_best_only=False, verbose=1):
        """
        Initializes the PeriodicModelCheckpoint callback.
        
        Args:
            filepath (str): Path where the model will be saved. Can include formatting options like {epoch}.
            save_freq_epochs (int): Frequency in epochs to save the model.
            save_best_only (bool): If True, saves the model only if the monitored metric improves.
            verbose (int): Verbosity mode. 0 = silent, 1 = messages.
        """
        super(PeriodicModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.save_freq_epochs = save_freq_epochs
        self.save_best_only = save_best_only
        self.verbose = verbose
        if self.save_best_only:
            # Initialize best metric based on 'val_f1'
            self.best = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch += 1  # Epoch indexing starts at 0

        # Save every 'save_freq_epochs' epochs
        if epoch % self.save_freq_epochs == 0:
            if self.save_best_only:
                current = logs.get('val_f1_score')
                if current is None:
                    if self.verbose > 0:
                        print(f"Validation F1-score ('val_f1_score') is not available. Skipping save.")
                    return
                if current > self.best:
                    self.best = current
                    filepath = self.filepath.format(epoch=epoch, **logs)
                    self.model.save(filepath)
                    if self.verbose > 0:
                        print(f"\nEpoch {epoch}: 'val_f1' improved to {current:.4f}. Saving model to {filepath}")
            else:
                # If not saving based on a metric, save unconditionally
                filepath = self.filepath.format(epoch=epoch, **logs)
                self.model.save(filepath)
                if self.verbose > 0:
                    print(f"\nEpoch {epoch}: Saving model to {filepath}")
