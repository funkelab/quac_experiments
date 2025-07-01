import zarr
from funlib.learn.torch.models.vgg2d import Vgg2D
import torch
import gunpowder as gp
from tqdm import tqdm
from scipy.special import softmax
import pandas as pd
import typer


def run_worker(
    name: int,
    metadata_path: str = "kenyon_cell_synapses2.csv",
    output_zarr_path: str = "/nrs/funke/adjavond/projects/quac/kenyon_cells/results2.zarr",
):
    """
    name: int
        Identifier of the neuron
        Will be used to filter the synapses in the metadata file, and as a group name in the zarr file
    metadata_path: str
        Path to the metadata file, which contains the synapse locations
    output_zarr_path: str
        Path to the zarr file where the results will be stored
    """
    df = pd.read_csv(metadata_path)
    syn = df[df["pre"] == name]

    # Metadata
    N = len(syn)
    batch_size = min(512, N)
    chunk_size = min(32, N)

    # Locate the synapses in the volume
    # This is in nm, so we need to convert to voxels
    coordinates = syn[["z", "y", "x"]].values

    # Loading the pre-trained classifier
    model = Vgg2D(input_size=(128, 128), fmaps=12)
    model.load_state_dict(
        torch.load(
            "/nrs/funke/adjavond/checkpoints/synapses/classifier/vgg_checkpoint",
            map_location="cpu",
        )["model_state_dict"]
    )
    model.eval()

    # Setting up the gunpowder pipeline
    voxel_size = gp.Coordinate((40, 4, 4))
    size = gp.Coordinate((1, 128, 128))
    size_nm = size * voxel_size
    raw = gp.ArrayKey("RAW")
    prediction = gp.ArrayKey("PREDICTION")

    pipeline = gp.ZarrSource(
        "/groups/futusa/futusa/projects/fafb/v14_align_tps_20170818_dmg.n5",
        {raw: "volumes/raw/s0"},
        {raw: gp.ArraySpec(interpolatable=True)},
    )

    pipeline += gp.SpecifiedLocation(coordinates.tolist(), choose_randomly=False)
    pipeline += gp.Normalize(raw)
    pipeline += gp.IntensityScaleShift(raw, scale=2, shift=-1)
    # pipeline += gp.PreCache(cache_size=512, num_workers=12)
    pipeline += gp.Stack(batch_size)
    pipeline += gp.torch.Predict(
        model, inputs={"raw": raw}, outputs={0: prediction}, device="cpu"
    )

    # Some more metadata
    output_zarr = zarr.open(output_zarr_path, "a")
    group = output_zarr.require_group(name)

    images = group.require_dataset(
        "images",
        shape=(N, 1, 128, 128),
        chunks=(chunk_size, 1, 128, 128),
        dtype="float32",
    )
    predictions = group.require_dataset(
        "predictions",
        shape=(N, 6),
        chunks=(chunk_size, 6),
        dtype="float32",
    )

    # Processing the data
    request = gp.BatchRequest()
    request[raw] = gp.ArraySpec(roi=gp.Roi((0, 0, 0), size_nm))
    request[prediction] = gp.ArraySpec(nonspatial=True)

    with gp.build(pipeline):
        for i in tqdm(range(0, N, batch_size)):
            start = i
            end = min(i + batch_size, N)
            number = end - start
            batch = pipeline.request_batch(request)
            images[start:end] = batch[raw].data[:number]
            predictions[start:end] = softmax(batch[prediction].data[:number], axis=1)


if __name__ == "__main__":
    typer.run(run_worker)
