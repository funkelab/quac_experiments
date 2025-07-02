from munch import Munch
from pathlib import Path
from quac.training.config import (
    ExperimentConfig,
    ModelConfig,
    DataConfig,
)
from quac.training.stargan import build_model
from quac.training.solver import Solver
from quac.training.data_loader import get_train_loader, get_test_loader
import timm
import wandb


def initialize_model(config: ModelConfig):
    model = build_model(config)
    # TODO: Load weights if needed
    return model


def initialize_dataloader(args: DataConfig):
    loaders = Munch(
        src=get_train_loader(
            root=args.train_img_dir,
            which="source",
            img_size=args.img_size,
            batch_size=args.batch_size,
            prob=args.randcrop_prob,
            num_workers=args.num_workers,
        ),
        ref=get_train_loader(
            root=args.train_img_dir,
            which="reference",
            img_size=args.img_size,
            batch_size=args.batch_size,
            prob=args.randcrop_prob,
            num_workers=args.num_workers,
        ),
        val=get_test_loader(
            root=args.val_img_dir,
            img_size=args.img_size,
            batch_size=args.val_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        ),
    )
    return loaders


def main(config: ExperimentConfig):
    run = wandb.init(
        project=config.project,
        notes=config.notes,
        tags=config.tags,
        config=config.dict(),
    )
    loaders = initialize_dataloader(config.data)
    model = initialize_model(config.model)
    model_ema = timm.utils.ModelEmaV2(model)

    solver = Solver(model, model_ema, run, config.training)

    # Run the training
    solver.train(loaders)
