import subprocess


def run_evaluation_worker(method):
    command = [
        "bsub",
        "-n 12",
        "-J",
        f"evaluate_{method}",
        "-o",
        f"worker_logs/evaluate_{method}.log",
        "-gpu",
        "num=1",
        "-q",
        "gpu_a100",
        "python",
        "evaluate_attribution_worker.py",
        method,
    ]
    subprocess.run(command)


if __name__ == "__main__":
    for method in ["VanillaIntegratedGradients", "VanillaDeepLift"]:
        run_evaluation_worker(method)
