# %% Read the curves from "curve.csv" and get only the values for D-IG and Vanilla IG as two separate CSV files
import pandas as pd

data = pd.read_csv("curve.csv")

# %% Values for D-IG
dig_columns = [col for col in data.columns if "discriminative_ig" in col]
dig_data = data[dig_columns]
# Remove "discriminative_ig_" from the column names
dig_data.columns = [col.replace("discriminative_ig_", "") for col in dig_data.columns]
dig_data.to_csv("DIntegratedGradients_curve.csv", index=False)

# %% Values for Vanilla IG
vig_columns = [col for col in data.columns if "vanilla_ig" in col]
vig_data = data[vig_columns]
# Remove "vanilla_ig_" from the column names
vig_data.columns = [col.replace("vanilla_ig_", "") for col in vig_data.columns]
vig_data.to_csv("VanillaIntegratedGradients_curve.csv", index=False)

# %%
