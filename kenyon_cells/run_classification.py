# %% Setup
import zarr
import pandas as pd
import subprocess


classes = [
    "GABA",
    "Acetylcholine",
    "Glutamate",
    "Serotonin",
    "Octopamine",
    "Dopamine",
]


def start_worker(
    name,
):
    command = [
        "bsub",
        "-n",
        "12",
        "-J",
        "kenyon_cells",
        "-o",
        f"worker_logs/{name}.out",
        "python",
        "run_classification_worker.py",
        str(name),
    ]

    subprocess.run(command)


if __name__ == "__main__":
    syn = pd.read_csv("kenyon_cell_synapses2.csv")
    zarr_path = "/nrs/funke/adjavond/projects/quac/kenyon_cells/results2.zarr"
    output_zarr = zarr.open(zarr_path, "w")

    for name, _ in syn.groupby("pre"):
        start_worker(name)
