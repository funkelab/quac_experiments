# Training the StarGAN
from quac.config import ExperimentConfig
from quac.training.data_loader import TrainingData, ValidationData
from quac.training.stargan import build_model
from quac.training.solver import Solver
from quac.training.logging import Logger
import torch
import yaml
import logging

torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    experiment = ExperimentConfig(**config)

    # Setting up the logger
    logger = Logger.create(
        experiment.log_type,
        hparams=experiment.model_dump(),
        resume_iter=experiment.run.resume_iter,
        **experiment.log.model_dump(),
    )

    logging.info("Loading training data.")
    # Defining the datasets for training and validation
    dataset = TrainingData(
        **experiment.data.model_dump(), latent_dim=experiment.model.latent_dim
    )
    logging.info("Loading validation data.")
    val_dataset = ValidationData(**experiment.validation_data.model_dump())

    # Defining the models
    logging.info("Building the model.")
    nets, nets_ema = build_model(**experiment.model.model_dump())

    # Defining the Solver
    logging.info("Building the solver...")
    solver = Solver(nets, nets_ema, **experiment.solver.model_dump(), run=logger)

    logging.info("Training.")
    # Training the model
    solver.train(
        dataset,
        **experiment.run.model_dump(),
        **experiment.loss.model_dump(),
        val_loader=val_dataset,
        val_config=experiment.validation_config,
    )
