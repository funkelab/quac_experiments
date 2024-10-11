import subprocess
from pathlib import Path


def run_process(noise_variance, intensity_shift):
    # TODO other parameters
    command = [
        "bsub",
        "-n",
        "12",
        "-gpu",
        "num=1",
        "-q",
        "gpu_a100",
        "-o",
        f"worker_logs/classification/noise_{noise_variance}_shift_{intensity_shift}.out",
        "python",
        "train_gp_classifier.py",
        "--noise_variance",
        f"{noise_variance}",
        "--intensity_shift",
        f"{intensity_shift}",
    ]
    subprocess.run(command)


if __name__ == "__main__":
    noise_variances = [0, 1e-4]
    intensity_shifts = [0, 1e-3, 1e-2, 1e-1]

    print("Creating log directory")
    log_dir = Path("worker_logs/classification")
    log_dir.mkdir(exist_ok=True, parents=True)

    print("Running processes")
    for noise_variance in noise_variances:
        for intensity_shift in intensity_shifts:
            print(f"Variance: {noise_variance}, Shift: {intensity_shift}")
            run_process(noise_variance, intensity_shift)
