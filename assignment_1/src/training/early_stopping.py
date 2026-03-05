# EarlyStopping: saves the best model checkpoint and signals when to stop training.
# Always MINIMISES the metric. To maximise (e.g. val_f1), pass the negative: stopper(-val_f1, model).

import torch


class EarlyStopping:
    """
    Generic early stopping that always minimises 'metric'.
    To track val_loss:     stopper(val_loss, model)         -- lower is better, pass as-is
    To track val_macro_f1: stopper(-val_metrics["macro_f1"], model)  -- higher is better, negate it

    patience: how many epochs with no improvement before stopping.
    checkpoint_path: where to save the best state_dict.
    """

    def __init__(self, patience: int, checkpoint_path: str):
        self.patience        = patience
        self.checkpoint_path = checkpoint_path
        self.best_metric     = float("inf")
        self._counter        = 0
        self._stop           = False

    def __call__(self, metric: float, model: torch.nn.Module) -> None:
        if metric < self.best_metric:
            # improvement — save checkpoint and reset counter
            self.best_metric = metric
            self._counter    = 0
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self._stop = True

    @property
    def stop(self) -> bool:
        """True when patience has been exceeded — caller should break the training loop."""
        return self._stop
