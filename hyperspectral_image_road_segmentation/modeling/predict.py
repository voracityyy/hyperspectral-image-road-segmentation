from pathlib import Path

import typer
import numpy as np
import torch

from models.unet import build_unet
from hyperspectral_image_road_segmentation.dataset import LoadHyperspectralImages 
from hyperspectral_image_road_segmentation.config import MODELS_DIR, FIGURES_DIR
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, roc_curve


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
   
    tensor = LoadHyperspectralImages()
    
    test_idx = np.load(MODELS_DIR / "test_idx.npy")
    test_ds = Subset(tensor, test_idx)
    
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    print("Starting inference loop...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_unet().to(device)

    # Load model on CPU (safe)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # ---- Metric counters ----

    total_pixels = 0
    correct_pixels = 0

    intersection = 0
    union = 0

    tp = 0
    fp = 0
    fn = 0

    # For ROC curve
    all_probs = []
    all_labels = []

    max_samples = 100000  # total pixels to store (safe limit)
    current_samples = 0

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
            
            y_true = masks.cpu().numpy().flatten()
            y_pred = pred.cpu().numpy().flatten()
            y_prob = prob.cpu().numpy().flatten()

            # sampling
            remaining = max_samples - current_samples

            if remaining > 0:
                sample_size = min(len(y_prob), remaining)
                indices = np.random.choice(len(y_prob), sample_size, replace=False)

                all_probs.append(y_prob[indices])
                all_labels.append(y_true[indices])

                current_samples += sample_size

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

    if len(all_probs) == 0:
        print("Warning: No samples collected for ROC")
        roc_auc = 0.0
        fpr, tpr = [], []
    else:
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        roc_auc = roc_auc_score(all_labels, all_probs)
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

    # ---- Save results ----
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        FIGURES_DIR / "metrics.npz",
        pixel_acc=pixel_acc,
        iou=iou,
        f1=f1,
        roc_auc=roc_auc,
        fpr=fpr,
        tpr=tpr
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
