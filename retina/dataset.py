import pandas as pd
from pathlib import Path
from PIL import Image


class RetinaDataset:
    def __init__(self, csv_path, data_location, transform=None):
        self.metadata = pd.read_csv(csv_path)
        self.data_location = Path(data_location)
        self.transform = transform
        self.classes = sorted(self.metadata["diagnosis"].unique())

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        filename, label = self.metadata.iloc[idx]
        image = Image.open(self.data_location / filename)
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def targets(self):
        return self.metadata["diagnosis"].values
