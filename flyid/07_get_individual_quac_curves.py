# %% Setup
import yaml
import matplotlib.pyplot as plt
import numpy as np
from quac.report import Report
import pandas as pd
from tqdm import tqdm

# %%
metadata = yaml.safe_load(open("configs/stargan.yml"))
subdir = "Day2/val"
kind = "latent"

metadata["report_directory"] = (
    metadata["solver"]["root_dir"] + f"/reports/{kind}/{subdir}"
)
# %% All the curves
reports = {
    method: Report(name=method)
    for method in [
        "DDeepLift",
        "DIntegratedGradients",
        "VanillaDeepLift",
        "VanillaIntegratedGradients",
    ]
}
for method, report in reports.items():
    report.load(metadata["report_directory"] + "/" + method + "/default_report.json")

# %% Save the curves
for method, report in tqdm(reports.items(), total=len(reports)):
    median, p25, p75 = report.get_curve()
    data = {
        "median": median,
        "p25": p25,
        "p75": p75,
    }
    df = pd.DataFrame(data)
    df.to_csv(f"results/{method}_curve.csv", index=False)
