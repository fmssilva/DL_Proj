# EarlyStopping: saves the best model checkpoint and signals when to stop training.

import torch


class EarlyStopping:
    """
    Monitors val_loss. Saves best model to checkpoint_path.
    Call instance each epoch; check .stop to know when to break the loop.
    """

    def __init__(self, patience: int, checkpoint_path: str):
        self.patience         = patience
        self.checkpoint_path  = checkpoint_path
        self.best_loss        = float("inf")
        self._counter         = 0
        self._stop            = False

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        if val_loss < self.best_loss:
            # improvement — save checkpoint and reset counter
            self.best_loss = val_loss
            self._counter  = 0
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self._stop = True

    @property
    def stop(self) -> bool:
        """True when patience has been exceeded — caller should break the training loop."""
        return self._stop
