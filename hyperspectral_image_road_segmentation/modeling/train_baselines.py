from pathlib import Path

from loguru import logger
import typer

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from hyperspectral_image_road_segmentation.config import MODELS_DIR
import pickle

from hyperspectral_image_road_segmentation.dataset import LoadHyperspectralImages 

app = typer.Typer()


@app.command()
def main(
    # # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    rf_model_path: Path = MODELS_DIR / "random_forest.pkl",
    sgd_model_path: Path = MODELS_DIR / "sgd_classifier.pkl",
    # # -----------------------------------------
):

    tensor = LoadHyperspectralImages()

    # Init 1 tree + warm_start to increment trees and prevent memory issues
    random_forest = RandomForestClassifier(warm_start=True, n_estimators=1)
    
    # 1000 max iterations *should* be enough
    # SGD with log_loss effectively acts like Logistic Regression
    sgd_classifier = SGDClassifier(loss="log_loss", max_iter=1000)
    
    # 80/20 train-test split
    total_size = len(tensor)
    rand_idx = np.random.permutation(total_size)
    # 1st 80%
    train_idx = set(rand_idx[:int(0.8 * total_size)])
    # Remaining (20%)
    test_idx = rand_idx[int(0.8 * total_size):]

    # Load images 1 at a time to prevent memory issues  
    for i, (hyperspectrals, masks) in enumerate(tensor):
        if i not in train_idx:
            continue
        channels = int(hyperspectrals.shape[0])
        # Convert (25, H, W) into (25, H*W) then transpose to (H*W, 25)
        # (band_x: [pixels] => pixel_x: [channels])
        X = hyperspectrals.numpy().reshape(channels, -1).T
        # No need to transpose masks because its already 1 dimension
        y = masks.numpy().reshape(-1)

        # Limit the number of decision trees to 100
        if random_forest.n_estimators < 100:
            random_forest.fit(X, y)
            random_forest.n_estimators += 1

        # Update SGD weights
        sgd_classifier.partial_fit(X, y, classes=[0, 1])


    with open(rf_model_path, "wb") as f:
        pickle.dump(random_forest, f)
    logger.success(f"Model saved to {rf_model_path}")

    with open(sgd_model_path, "wb") as f:
        pickle.dump(sgd_classifier, f)
    logger.success(f"Model saved to {sgd_model_path}")

    # Save for predict.py
    np.save(MODELS_DIR / "baseline_test_idx.npy", test_idx)


if __name__ == "__main__":
    app()
