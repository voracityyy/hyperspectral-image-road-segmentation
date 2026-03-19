from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from hyperspectral_image_road_segmentation.config import MODELS_DIR, PROCESSED_DATA_DIR

from hyperspectral_image_road_segmentation.models.unet import build_unet
from hyperspectral_image_road_segmentation.dataset import LoadHyperspectralImages 

app = typer.Typer()


@app.command()
def main(
    # # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    # model_path: Path = MODELS_DIR / "model.pkl",
    # # -----------------------------------------
):

    tensor = LoadHyperspectralImages()

    # Defaults:
    # DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
    #        batch_sampler=None, num_workers=0, collate_fn=None,
    #        pin_memory=False, drop_last=False, timeout=0,
    #        worker_init_fn=None, *, prefetch_factor=2,
    #        persistent_workers=False)
    dataloader = DataLoader(tensor, batch_size=10, shuffle=True) # still need to test which param values are best

    # Dataset split: training=80% / validation=10% / testing=10%
    total_size = len(tensor)
    train_size = 0.8 * total_size
    validation_size, test_size = 0.1 * total_size
    train_size, validation_size, test_size = random_split(tensor, [train_size, validation_size, test_size])

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = build_unet().to(device)
    # Updates weights after backpropagation / each batch
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # still need to test which param values are best
    # Calculates model prediction errors
    loss_fn = smp.losses.SoftCrossEntropyLoss(mode="binary") # still need to test which loss function is best
    # Reduced learning rate (lr in optimizer) after x epochs (patience value)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3 # still need to test which param values are best
    )

    # Lists to track training history for plots 
    # Track loss history
    train_loss_hist = [] 
    val_loss_hist = []
    # Track lowest validation loss
    best_val_loss = float("inf")
    # Track the weights from best performing epoch
    best_state = None

    # # ---- REPLACE THIS WITH YOUR OWN CODE ----
    # logger.info("Training some model...")
    # for i in tqdm(range(10), total=10):
    #     if i == 5:
    #         logger.info("Something happened for iteration 5.")
    # logger.success("Modeling training complete.")
    # # -----------------------------------------

if __name__ == "__main__":
    app()
