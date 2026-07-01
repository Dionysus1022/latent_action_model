import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from training.utils import ModelObjectCallBack, get_column_normalizer, get_img_preprocessor


def build_hdf5_dataset(cfg):
    """Load datasets across stable_worldmodel versions.

    Some releases export HDF5Dataset from stable_worldmodel.data, while others
    keep it only under stable_worldmodel.data.dataset.
    """
    dataset_cls = getattr(swm.data, "HDF5Dataset", None)
    if dataset_cls is None:
        try:
            from stable_worldmodel.data.dataset import HDF5Dataset as dataset_cls
        except ImportError as exc:
            raise AttributeError(
                "stable_worldmodel does not expose HDF5Dataset. "
                "Please upgrade stable-worldmodel or use a compatible version."
            ) from exc

    dataset_kwargs = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    if not isinstance(dataset_kwargs, dict):
        raise TypeError(
            f"cfg.data.dataset must resolve to a dict, got {type(dataset_kwargs).__name__}."
        )

    return dataset_cls(**dataset_kwargs, transform=None)


def build_vit_encoder(cfg):
    """Create the ViT encoder without relying on stable_pretraining's optional import gate.

    stable_pretraining 0.1.7 marks transformers as unavailable when
    `TimmWrapperModel` is missing, even though `ViTConfig` and `ViTModel`
    are present and sufficient for this training script.
    """
    try:
        from transformers import ViTConfig, ViTModel
    except ImportError as exc:
        raise ImportError(
            "transformers with ViT support is required for LeWM training. "
            "Install or upgrade transformers in the active environment."
        ) from exc

    size_configs = {
        "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
        "small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
        "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
        "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16},
        "huge": {"hidden_size": 1280, "num_hidden_layers": 32, "num_attention_heads": 16},
    }
    if cfg.encoder_scale not in size_configs:
        raise ValueError(
            f"Invalid encoder_scale '{cfg.encoder_scale}'. "
            f"Choose from {list(size_configs.keys())}."
        )

    config_params = dict(size_configs[cfg.encoder_scale])
    config_params["intermediate_size"] = config_params["hidden_size"] * 4
    config_params["image_size"] = cfg.img_size
    config_params["patch_size"] = cfg.patch_size

    config = ViTConfig(**config_params)
    model = ViTModel(
        config,
        add_pooling_layer=False,
        use_mask_token=False,
    )
    model.config.interpolate_pos_encoding = True
    return model


def build_logger(cfg):
    """Build the experiment logger.

    stable_pretraining's Manager crashes when a WandB run is in offline mode
    without a previous run directory to resume from. In that case we disable
    the WandB logger and let Lightning fall back to its default CSV logger.
    """
    if not cfg.wandb.enabled:
        return None

    logger = WandbLogger(**cfg.wandb.config)
    try:
        experiment = logger.experiment
    except Exception as exc:
        print(
            "WandB logger initialization failed; disabling WandB logging. "
            f"Reason: {exc}"
        )
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass
        return None

    if getattr(experiment, "offline", False):
        print(
            "WandB is in offline mode for this working directory; "
            "disabling WandB logger to avoid stable_pretraining resume bugs."
        )
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass
        return None

    logger.log_hyperparams(OmegaConf.to_container(cfg))
    return logger


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = build_hdf5_dataset(cfg)
    transforms = [
        get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)
    ]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    encoder = build_vit_encoder(cfg)
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = build_logger(cfg)

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    resume_ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    if not resume_ckpt_path.exists():
        resume_ckpt_path = None

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=resume_ckpt_path,
    )

    manager()
    return


if __name__ == "__main__":
    run()
