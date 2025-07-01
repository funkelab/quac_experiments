from quac.data import Sample, PairedSample, SampleWithAttribution


class SourceDataset:
    def __init__(self, zarr_file):
        self.zarr_file = zarr_file
        self.images = self.zarr_file["images"]
        self.predictions = self.zarr_file["predictions"]

    def __getitem__(self, index):
        image = self.images[index]
        source_class_index = int(self.predictions[index].argmax())
        return Sample(image, source_class_index)

    def __len__(self):
        return len(self.images)


class PairedDataset:
    def __init__(self, zarr_file):
        self.zarr_file = zarr_file
        self.images = self.zarr_file["images"]
        self.predictions = self.zarr_file["predictions"]
        self.counterfactuals = self.zarr_file["counterfactuals"]
        # The classes are the groups inside the counterfactuals/counterfactual_predictions
        self.classes = sorted(list(self.counterfactuals.group_keys()))
        self.counterfactual_predictions = self.zarr_file["counterfactual_predictions"]
        # Check that the counterfactuals and the counterfactual_predictions have the same classes
        assert (
            self.counterfactual_predictions_keys
            == self.counterfactual_predictions.group_keys()
        )
        # Pull out only the valid samples
        is_valid = self.zarr_file["is_valid"]
        # Figure out how many samples there are in total, and create a list of the items for getitem
        self.items = [
            (class_name, index)
            for class_name in self.classes
            for index in range(len(self.counterfactuals[class_name]))
            if is_valid[class_name, index]
        ]

    def __getitem__(self, index):
        target_class_name, index = self.items[index]

        # Get the original image
        image = self.images[index]
        source_class_index = int(self.predictions[index].argmax())

        # Get the corresponding counterfactual
        counterfactual = self.counterfactuals[target_class_name][index]
        target_class_index = self.classes.index(target_class_name)
        return PairedSample(
            image, counterfactual, source_class_index, target_class_index
        )

    def __len__(self):
        return len(self.items)


class DatasetWithAttribution:
    def __init__(
        self,
        zarr_file,
        attribution_method,
    ):
        self.zarr_file = zarr_file
        self.images = self.zarr_file["images"]
        self.predictions = self.zarr_file["predictions"]
        self.counterfactuals = self.zarr_file["counterfactuals"]
        # The classes are the groups inside the counterfactuals/counterfactual_predictions
        self.classes = sorted(list(self.counterfactuals.array_keys()))
        self.counterfactual_predictions = self.zarr_file["counterfactual_predictions"]
        # Pull out only the valid samples
        is_valid = self.zarr_file["is_valid"]
        # Figure out how many samples there are in total, and create a list of the items for getitem
        self.items = [
            (class_name, index)
            for class_name in self.classes
            for index in range(len(self.counterfactuals[class_name]))
            if is_valid[class_name][index]
        ]
        # Attribution data
        self.attribution_method = attribution_method
        self.attributions = self.zarr_file["attributions"]

    def __getitem__(self, index):
        target_class_name, index = self.items[index]

        # Get the original image
        image = self.images[index]
        source_class_index = int(self.predictions[index].argmax())

        # Get the corresponding counterfactual
        counterfactual = self.counterfactuals[target_class_name][index]
        target_class_index = self.classes.index(target_class_name)

        attribution = self.attributions[target_class_name][self.attribution_method][
            index
        ]
        return SampleWithAttribution(
            attribution, image, counterfactual, source_class_index, target_class_index
        )

    def __len__(self):
        return len(self.items)
