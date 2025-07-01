import subprocess
import pandas as pd
import sys


def start_worker(zarr_path, index):
    command = [
        "bsub",
        "-J",
        f"evaluation_{index}",
        "-n",
        "12",
        "-W",
        "00:15",
        "-gpu",
        "num=1",
        "-q",
        "gpu_a100",
        "-o",
        f"worker_logs/evaluation/{index}.out",
        "python",
        "run_evaluation_worker.py",
        zarr_path,
        f"{index}",
    ]
    subprocess.run(command)


if __name__ == "__main__":
    result_path = "/nrs/funke/adjavond/projects/quac/kenyon_cells/results2.zarr"

    syn_fraction_path = "syn_class_fractions.csv"
    syn_class_fractions = pd.read_csv(syn_fraction_path, index_col=0)

    try:
        num_samples = int(sys.argv[1])
    except:
        num_samples = len(syn_class_fractions)
    for index in syn_class_fractions.index[:num_samples]:
        start_worker(result_path, index)
