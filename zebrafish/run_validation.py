import wandb
from pathlib import Path
from pprint import pprint
import subprocess


def run_process(run_id, seed):
    command = [
        "bsub",
        "-n",
        "12",
        "-gpu",
        "num=1",
        "-q",
        "gpu_a100",
        "-o",
        f"worker_logs/classification/run_{run_id}_seed_{seed}.out",
        "python",
        "validate_gp_classifier.py",
        "--run_id",
        f"{run_id}",
        "--seed",
        f"{seed}",
    ]
    subprocess.run(command)


if __name__ == "__main__":
    # All available runs
    api = wandb.Api()
    runs = api.runs("adjavon/zebrafish-classification")

    # Runs I can about
    results_directory = Path("/nrs/funke/adjavond/zebrafish/classifiers/checkpoints")
    directory_names = [directory.name for directory in results_directory.iterdir()]
    run_ids = {run.name: run.id for run in runs if run.name in directory_names}

    # seeding for reproducibility
    seed = 42

    for name, run_id in run_ids.items():
        print("Running", name)
        run_process(run_id, seed)
