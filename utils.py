import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback


def _resize_image_tensor(x: torch.Tensor, img_size: int) -> torch.Tensor:
    """Resize CHW or TCHW image tensors without relying on stable_pretraining.Resize."""
    if not torch.is_tensor(x):
        raise TypeError(f"Expected a torch.Tensor image, got {type(x).__name__}.")

    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeeze = True
    elif x.ndim == 4:
        squeeze = False
    else:
        raise ValueError(
            f"Expected image tensor with shape [C, H, W] or [T, C, H, W], got {tuple(x.shape)}."
        )

    if not torch.is_floating_point(x):
        x = x.float()

    x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return x.squeeze(0) if squeeze else x


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.WrapTorchTransform(
        lambda x: _resize_image_tensor(x, img_size),
        source=source,
        target=target,
    )
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer

class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")
