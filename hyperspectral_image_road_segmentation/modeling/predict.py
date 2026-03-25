from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from torch.utils.data import DataLoader, random_split

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
):
    # # ---- REPLACE THIS WITH YOUR OWN CODE ----
    # logger.info("Performing inference for model...")
    # for i in tqdm(range(10), total=10):
    #     if i == 5:
    #         logger.info("Something happened for iteration 5.")
    # logger.success("Inference complete.")
    # # -----------------------------------------


    tensor = LoadHyperspectralImages()
    total_size = len(tensor)
    train_size = int(0.8 * total_size)
    validation_size = int(0.1 * total_size)
    # ^ boilerplate code so we can get the exact test dataset size
    test_size = int(total_size - train_size - validation_size) # aka the remainder

    test_ds = random_split(tensor, [test_size])


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
