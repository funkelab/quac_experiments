# %%
import pandas as pd
import subprocess
import sys


def start_worker(
    zarr_path,
    cell_id,
):
    command = [
        "bsub",
        "-n 12",
        "-J",
        f"attr_{cell_id}",
        "-W",
        "00:15",
        "-gpu",
        "num=1",
        "-q",
        "gpu_a100",
        "-o",
        f"worker_logs/attributions/{cell_id}.out",
        "python",
        "run_attribution_worker.py",
        zarr_path,
        f"{cell_id}",
    ]
    subprocess.run(command)


if __name__ == "__main__":
    df = pd.read_csv("syn_class_fractions.csv")
    zarr_path = "/nrs/funke/adjavond/projects/quac/kenyon_cells/results2.zarr"

    try:
        # Run a subset
        num_samples = int(sys.argv[1])
    except IndexError:
        # Run all
        num_samples = len(df)
    to_run = df["pre"][:num_samples]
    print(f"Running {num_samples} samples")
    for i in to_run:
        start_worker(zarr_path, i)
