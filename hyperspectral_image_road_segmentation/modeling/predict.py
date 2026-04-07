from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from models.unet import build_unet

from hyperspectral_image_road_segmentation.dataset import LoadHyperspectralImages 

from hyperspectral_image_road_segmentation.config import MODELS_DIR, FIGURES_DIR

from torch.utils.data import Subset

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

    #test_idx = np.load(MODELS_DIR / "test_idx.npy")
    #test_ds = Subset(tensor, test_idx)
    
    test_ds = tensor
    
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    print("Starting inference loop...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_unet().to(device)

     # Load model on CPU (safe)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # ---- Metric counters ----

    #all_true = []
    #all_pred = []
    #all_prob = []

    total_pixels = 0
    correct_pixels = 0

    intersection = 0
    union = 0

    tp = 0
    fp = 0
    fn = 0

    # ---- Inference loop ----
    with torch.no_grad():
        for hyperspectrals, masks in test_loader:
            hyperspectrals = hyperspectrals.to(device)
            masks = masks.to(device)
            output = model(hyperspectrals)
            
            # Normalize outputs into range 0-1
            prob = torch.sigmoid(output)
            
            # If output > 0.5 then convert to 1 (True)
            pred = (prob > 0.5).long()
            
            #all_true.extend(masks.cpu().numpy())
            #all_pred.extend(pred.cpu().numpy())
            #all_prob.extend(prob.cpu().numpy())

            y_true = masks.cpu().numpy().flatten()
            y_pred = pred.cpu().numpy().flatten()

            # Pixel Accuracy
            correct_pixels += (y_true == y_pred).sum()
            total_pixels += len(y_true)

            # IoU
            intersection += np.logical_and(y_true, y_pred).sum()
            union += np.logical_or(y_true, y_pred).sum()

            # F1 components
            tp += np.logical_and(y_true == 1, y_pred == 1).sum()
            fp += np.logical_and(y_true == 0, y_pred == 1).sum()
            fn += np.logical_and(y_true == 1, y_pred == 0).sum()

    # ---- Final metrics ----
    pixel_acc = correct_pixels / total_pixels
    iou = intersection / (union + 1e-8)
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)

    # ---- Save results ----
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        FIGURES_DIR / "metrics.npz",
        pixel_acc=pixel_acc,
        iou=iou,
        f1=f1
    )

    print("Metrics saved to reports/figures/metrics.npz")

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
