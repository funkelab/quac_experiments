from pydantic import BaseModel
from typing import List


class DataConfig(BaseModel):
    csv_path: str
    data_location: str
    batch_size: int = 32
    augment: bool = True
    balance: bool = True


class ModelConfig(BaseModel):
    type: str = "vgg"
    num_classes: int = 5
    image_size: int = 128
    input_fmaps: int = 1
    fmaps: int = 32


class TrainingConfig(BaseModel):
    epochs: int = 10
    learning_rate: float = 1e-4
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda"


class ExperimentConfig(BaseModel):
    project: str
    notes: str
    tags: List[str] = ["quac"]
    log_images: int = 8 # Number of images to log to wandb
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    val_data: DataConfig = None