from pydantic import BaseModel
from typing import List


class DataConfig(BaseModel):
    data_location: str
    batch_size: int = 32
    augment: bool = True
    balance: bool = True


class ModelConfig(BaseModel):
    type: str = "vgg"
    num_classes: int = 4
    image_size: int = 128
    input_fmaps: int = 3
    fmaps: int = 32


class SchedulerConfig(BaseModel):
    epoch: int = 0
    unfreeze: bool = True


class EMAModelConfig(BaseModel):
    decay: float = 0.9999
    ema_start: int = 0


class TrainingConfig(BaseModel):
    epochs: int = 10
    learning_rate: float = 1e-4
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda"
    scheduler: SchedulerConfig = SchedulerConfig()
    ema: EMAModelConfig = EMAModelConfig()


class ExperimentConfig(BaseModel):
    project: str
    notes: str
    tags: List[str] = ["quac"]
    log_images: int = 8 # Number of images to log to wandb
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    val_data: DataConfig = None