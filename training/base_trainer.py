import random
import torch
from contextlib import nullcontext

from data import FloodEventDataset
from torch.nn import Module
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader, ClusterLoader
from typing import Callable, Optional
from utils import Logger, EarlyStopping
from utils.training_stats import TrainingStats


class SameRunBatchSampler(Sampler[list[int]]):
    """Batch samples only with other samples from the same run_id/topology.

    mSWE-GNN events do not share a mesh, so a batch mixing two events would mix two node
    orderings and two partition maps. This keeps every batch inside one event; use
    --batching_strategy cross_event to allow mixed batches instead.
    """
    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.groups = {}
        if not hasattr(dataset, "data_list"):
            raise ValueError("SameRunBatchSampler requires an in-memory dataset with data_list.")
        for idx, sample in enumerate(dataset.data_list):
            if not hasattr(sample, "run_id"):
                raise ValueError("SameRunBatchSampler requires each sample to define run_id.")
            run_id = sample.run_id
            if torch.is_tensor(run_id):
                run_id = int(run_id.view(-1)[0].item())
            else:
                run_id = int(run_id)
            self.groups.setdefault(run_id, []).append(idx)

    def __iter__(self):
        run_ids = list(self.groups)
        if self.shuffle:
            random.shuffle(run_ids)
        for run_id in run_ids:
            indices = list(self.groups[run_id])
            if self.shuffle:
                random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start:start + self.batch_size]

    def __len__(self):
        return sum((len(indices) + self.batch_size - 1) // self.batch_size for indices in self.groups.values())

class BaseTrainer:
    def __init__(self,
                 model: Module,
                 dataset: FloodEventDataset,
                 optimizer: Optimizer,
                 loss_func: Callable,
                 node_loss_weight: float = 1.0,
                 batch_size: int = 64,
                 num_epochs: int = 100,
                 num_epochs_dyn_loss: int = 10,
                 gradient_clip_value: Optional[float] = None,
                 early_stopping_patience: Optional[int] = None,
                 val_dataset: Optional[FloodEventDataset] = None,
                 logger: Logger = None,
                 device: str = 'cpu',
                 isCluster: Optional[bool] = False,
                 same_run_batching: bool = False,
                 amp_mode: str = 'none',
                 wandb_run=None,
                 epoch_end_callback: Optional[Callable] = None):
        self.dataset = dataset  # Store reference for metadata access
        if same_run_batching:
            if logger:
                logger.log("Using same-run DataLoader batching for variable-topology cluster training.")
            batch_sampler = SameRunBatchSampler(dataset, batch_size=batch_size, shuffle=True)
            self.dataloader = DataLoader(dataset, batch_sampler=batch_sampler)
        elif isCluster:
            if logger:
                logger.log("Using standard DataLoader (cluster sampling in trainer).")
            self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.node_loss_weight = node_loss_weight
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_epochs_dyn_loss = num_epochs_dyn_loss
        self.gradient_clip_value = gradient_clip_value
        self.val_dataset = val_dataset
        self.device = device
        self.epoch_end_callback = epoch_end_callback
        self.validation_event_metrics_history = []
        self.amp_mode = amp_mode
        self.use_amp = self.device == 'cuda' and self.amp_mode != 'none'
        if self.amp_mode == 'bf16':
            self.amp_dtype = torch.bfloat16
        elif self.amp_mode == 'fp16':
            self.amp_dtype = torch.float16
        elif self.amp_mode == 'none':
            self.amp_dtype = None
        else:
            raise ValueError(f"Unsupported amp_mode: {self.amp_mode}")
        self.grad_scaler = torch.amp.GradScaler('cuda', enabled=(self.device == 'cuda' and self.amp_mode == 'fp16'))

        if early_stopping_patience is not None:
            assert self.val_dataset is not None, "Validation dataset must be provided if early stopping is used."
            self.early_stopping = EarlyStopping(patience=early_stopping_patience)

        self.training_stats = TrainingStats(logger=logger, wandb_run=wandb_run)
        self.training_stats.log(f"AMP mode: {self.amp_mode}; autocast enabled: {self.use_amp}; grad scaler enabled: {self.grad_scaler.is_enabled()}")

        assert self.num_epochs_dyn_loss <= self.num_epochs, "Number of epochs for dynamic loss scaling must not exceed total number of epochs."

    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type='cuda', dtype=self.amp_dtype)

    def train(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def validate(self):
        raise NotImplementedError("Subclasses should implement this method if using early stopping.")

    def _notify_epoch_end(self, epoch: int, metrics: dict):
        if self.epoch_end_callback is not None:
            self.epoch_end_callback(self, epoch, metrics)

    def _scale_node_pred_loss(self, epoch: int, pred_loss: Tensor) -> Tensor:
        if epoch < self.num_epochs_dyn_loss:
            return pred_loss
        return pred_loss * self.node_loss_weight

    def _clip_gradients(self):
        if self.gradient_clip_value is not None:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.gradient_clip_value)

    def print_stats_summary(self):
        self.training_stats.print_stats_summary()

    def save_stats(self, filepath: str):
        self.training_stats.save_stats(filepath)

    def save_model(self, model_path: str):
        model_to_save = getattr(self.model, '_orig_mod', self.model)
        torch.save(model_to_save.state_dict(), model_path)
        self.training_stats.log(f'Saved model to: {model_path}')
