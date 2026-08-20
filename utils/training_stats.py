import os
import time
import numpy as np
from . import Logger

class TrainingStats:
    def __init__(self, logger: Logger = None, wandb_run=None):
        self.total_epoch_loss = []
        self.epoch_loss_components = {}
        self.epoch_val_loss_components = {}
        self.additional_info = {}
        self.wandb_run = wandb_run
        self._wandb_metrics_defined = False

        self.log = print
        if logger is not None and hasattr(logger, 'log'):
            self.log = logger.log

    def start_train(self):
        self.train_start_time = time.time()

    def end_train(self):
        self.train_end_time = time.time()

    def get_train_time(self):
        return self.train_end_time - self.train_start_time

    def add_loss(self, loss):
        self.total_epoch_loss.append(loss)

    def add_loss_component(self, key: str, loss):
        if key not in self.epoch_loss_components:
            self.epoch_loss_components[key] = []
        self.epoch_loss_components[key].append(loss)

    def add_val_loss_component(self, key: str, loss):
        if key not in self.epoch_val_loss_components:
            self.epoch_val_loss_components[key] = []
        self.epoch_val_loss_components[key].append(loss)

    def add_additional_info(self, key: str, info):
        self.additional_info[key] = info

    def _define_wandb_metrics(self):
        """Declare epoch as the x-axis for train/* and val/* so W&B plots against epoch
        rather than an internal step counter. Test metrics are logged once, unstepped."""
        if self.wandb_run is None or self._wandb_metrics_defined:
            return
        try:
            self.wandb_run.define_metric("epoch")
            self.wandb_run.define_metric("train/*", step_metric="epoch")
            self.wandb_run.define_metric("val/*", step_metric="epoch")
            self.wandb_run.define_metric("test/*")
            self._wandb_metrics_defined = True
        except Exception as exc:
            self.log(f"Warning: failed to define W&B metrics: {exc}")

    def log_metrics(self, metrics: dict, step: int = None):
        """Log a metric payload to W&B if a run is attached; a no-op otherwise.

        Every W&B call here is best-effort: logging must never take down a training job that
        is otherwise fine, so failures are reported and swallowed.
        """
        if self.wandb_run is None or not metrics:
            return
        payload = dict(metrics)
        if step is not None:
            payload["epoch"] = int(step)
        try:
            self._define_wandb_metrics()
            self.wandb_run.log(payload)
        except Exception as exc:
            self.log(f"Warning: failed to log metrics to W&B: {exc}")

    def print_stats_summary(self):
        if len(self.total_epoch_loss) > 0:
            final_loss = float(self.total_epoch_loss[-1])
            self.log(f'Final training Loss: {final_loss:.4e}')
            self.log_metrics({"train/final_loss": final_loss})
            np_epoch_loss = np.array(self.total_epoch_loss)
            self.log(f'Average training Loss: {np_epoch_loss.mean():.4e}')
            self.log(f'Minimum training Loss: {np_epoch_loss.min():.4e}')
            self.log(f'Maximum training Loss: {np_epoch_loss.max():.4e}')

        if self.train_start_time is not None and self.train_end_time is not None:
            train_time = float(self.get_train_time())
            self.log(f'Total training time: {train_time:.4f} seconds')
            self.log_metrics({"train/total_time_s": train_time})

    def save_stats(self, filepath: str):
        dirname = os.path.dirname(filepath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        stats = {
            'train_epoch_loss': np.array(self.total_epoch_loss),
            'train_time': self.get_train_time(),
        }
        np_loss_components = {k: np.array(v) for k, v in self.epoch_loss_components.items()}
        stats.update(np_loss_components)

        np_val_loss_components = {k: np.array(v) for k, v in self.epoch_val_loss_components.items()}
        stats.update(np_val_loss_components)

        stats.update(self.additional_info)
        np.savez(filepath, **stats)
        self.log(f'Saved training stats to: {filepath}')
