from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from models.unet import build_unet

from hyperspectral_image_road_segmentation.dataset import LoadHyperspectralImages 

from hyperspectral_image_road_segmentation.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    # model_path: Path = MODELS_DIR / "model.pkl",
    # predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # # -----------------------------------------
    model_path: Path = MODELS_DIR / "unet.pth"
):
    # # ---- REPLACE THIS WITH YOUR OWN CODE ----
    # logger.info("Performing inference for model...")
    # for i in tqdm(range(10), total=10):
    #     if i == 5:
    #         logger.info("Something happened for iteration 5.")
    # logger.success("Inference complete.")
    # # -----------------------------------------

    tensor = LoadHyperspectralImages()

    test_idx = np.load(MODELS_DIR / "test_idx.npy")
    test_ds = Subset(tensor, test_idx)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_unet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
        
    all_true = []
    all_pred = []
    all_prob = []

    with torch.no_grad():
        for hyperspectrals, masks in test_loader:
            hyperspectrals = hyperspectrals.to(device)
            masks = masks.to(device)
            output = model(hyperspectrals)
            
            # Normalize outputs into range 0-1
            prob = torch.sigmoid(output)
            # If output > 0.5 then convert to 1 (True)
            pred = (prob > 0.5).long()
            
            all_true.extend(masks.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
            all_prob.extend(prob.cpu().numpy())

    # Possible evaluation metrics to consider:
        # Intersection over Union (IoU)
        # Dice Coefficient
        # Precision
        # Recall
        # F1 Score
        # Mean Absolute Error (MAE)
        # Hausdorff Distance
        # Pixel Accuracy

if __name__ == "__main__":
    app()
