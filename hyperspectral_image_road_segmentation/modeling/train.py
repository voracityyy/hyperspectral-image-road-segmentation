from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp

from hyperspectral_image_road_segmentation.config import MODELS_DIR, FIGURES_DIR

from models.unet import build_unet
from hyperspectral_image_road_segmentation.dataset import LoadHyperspectralImages 

app = typer.Typer()


@app.command()
def main(
    # # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "unet.pth",
    # # -----------------------------------------
):

    tensor = LoadHyperspectralImages()

    # Dataset split: training=80% / validation=10% / testing=10%
    total_size = len(tensor)
    train_size = int(0.8 * total_size)
    validation_size = int(0.1 * total_size)
    test_size = int(total_size - train_size - validation_size) # aka the remainder
    train_ds, validation_ds, test_ds = random_split(tensor, [train_size, validation_size, test_size])

    # Save for predict.py
    np.save(MODELS_DIR / "test_idx.npy", test_ds.indices)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(validation_ds, batch_size=16, shuffle=False)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = build_unet().to(device)
    # Updates weights after backpropagation / each batch
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # still need to test which param values are best
    # Calculates model prediction errors
    loss_function = smp.losses.DiceLoss(mode="binary") # still need to test which loss function is best

    # loss_function = smp.losses.CrossEntropyLoss()

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
    
    # ---- TRAINING LOOP ----
    for epoch in range(0, 100):

        model.train()
        train_loss_total = 0
        train_sample_total = 0

        # If there is no validation loss improvement after 10 epochs,
        # stop training to prevent overfitting
        no_improvement_count = 0
        patience = 10

        for hyperspectrals, masks in train_loader:
            hyperspectrals = hyperspectrals.to(device)
            masks = masks.to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            predictions = model(hyperspectrals)
            current_loss = loss_function(predictions, masks)
            current_loss.backward()
            optimizer.step()

            train_loss_total += current_loss.item()
            train_sample_total += hyperspectrals.size(0)

        train_loss_avg = train_loss_total / train_sample_total
        train_loss_hist.append(train_loss_avg)

        # ---- VALIDATION LOOP ----
        model.eval()
        val_loss_total = 0
        val_sample_total = 0

        with torch.no_grad():
            for hyperspectrals, masks in val_loader:
                hyperspectrals = hyperspectrals.to(device)
                masks = masks.to(device)
                
                predictions = model(hyperspectrals)
                current_loss = loss_function(predictions, masks)
                
                val_loss_total += current_loss.item()
                val_sample_total += hyperspectrals.size(0)

        val_loss_avg = val_loss_total / val_sample_total
        val_loss_hist.append(val_loss_avg)

        scheduler.step(val_loss_avg)

        logger.info(f"epoch {epoch+1:02d} train_loss_avg={train_loss_avg:.4f} val_loss_avg={val_loss_avg:.4f}")

        # Save model
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_state = model.state_dict()
        else: 
            no_improvement_count += 1

        if no_improvement_count >= patience:
            break

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_path)
    logger.success(f"Model saved to {model_path}")

    # Save history for plotting
    np.save(FIGURES_DIR / "train_loss.npy", train_loss_hist)
    np.save(FIGURES_DIR / "val_loss.npy",   val_loss_hist)
    logger.info(f"Loss history saved at {FIGURES_DIR}")

if __name__ == "__main__":
    app()
