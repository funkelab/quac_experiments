from quac.data import Sample, PairedSample, SampleWithAttribution


class SourceDataset:
    def __init__(self, zarr_file, neuron_index):
        self.zarr_file = zarr_file
        self.neuron_index = neuron_index
        self.neuron = self.zarr_file[self.neuron_index]
        self.images = self.neuron["images"]
        self.predictions = self.neuron["predictions"]

    def __getitem__(self, index):
        image = self.images[index]
        source_class_index = int(self.predictions[index].argmax())
        return Sample(image, source_class_index)

    def __len__(self):
        return len(self.images)


class PairedDataset:
    def __init__(self, zarr_file, neuron_index, target_class_override=None):
        self.zarr_file = zarr_file
        self.neuron_index = neuron_index
        self.neuron = self.zarr_file[self.neuron_index]
        self.images = self.neuron["images"]
        self.predictions = self.neuron["predictions"]
        self.counterfactuals = self.neuron["counterfactuals"]
        self.counterfactual_predictions = self.neuron["counterfactual_predictions"]
        self.target_class_override = target_class_override

    def __getitem__(self, index):
        image = self.images[index]
        source_class_index = int(self.predictions[index].argmax())
        counterfactual = self.counterfactuals[index]
        if self.target_class_override is not None:
            target_class_index = self.target_class_override
        else:
            target_class_index = self.counterfactual_predictions[index].argmax()
        return PairedSample(
            image, counterfactual, source_class_index, target_class_index
        )

    def __len__(self):
        return len(self.images)


class DatasetWithAttribution:
    def __init__(
        self, zarr_file, neuron_index, attribution_method, target_class_override=None
    ):
        self.zarr_file = zarr_file
        self.neuron_index = neuron_index
        self.neuron = self.zarr_file[self.neuron_index]
        self.images = self.neuron["images"]
        self.predictions = self.neuron["predictions"]
        self.counterfactuals = self.neuron["counterfactuals"]
        self.counterfactual_predictions = self.neuron["counterfactual_predictions"]
        self.attributions = self.neuron["attributions"][attribution_method]
        self.target_class_override = target_class_override

    def __getitem__(self, index):
        image = self.images[index]
        source_class_index = int(self.predictions[index].argmax())
        counterfactual = self.counterfactuals[index]
        if self.target_class_override is not None:
            target_class_index = self.target_class_override
        else:
            target_class_index = self.counterfactual_predictions[index].argmax()
        attribution = self.attributions[index]
        return SampleWithAttribution(
            attribution, image, counterfactual, source_class_index, target_class_index
        )

    def __len__(self):
        return len(self.images)
