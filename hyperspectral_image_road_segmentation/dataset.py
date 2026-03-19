# from pathlib import Path

import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset
from loguru import logger
# from tqdm import tqdm
import typer

from hyperspectral_image_road_segmentation.config import HYPERSPECTRAL_DIR, MASKS_DIR

app = typer.Typer()


class LoadHyperspectralImages(Dataset):

    def __init__(self):

        # Initialize both image types sorted, so they match 1-1 with each other.
        self.hyperspectral_images = sorted(HYPERSPECTRAL_DIR.glob("*_nir.tiff"))
        self.mask_images  = sorted(MASKS_DIR.glob("*_nif.tiff"))

        # Crash program if file count does not match.
        assert len(self.hyperspectral_images) == len(self.mask_images), (
            f"Mismatch: {len(self.image_files)} images, {len(self.mask_files)} masks"
        )

        # At this point both image counts should be equal
        logger.info(f"{len(self.hyperspectral_images)} samples found")

    def __len__(self):
        return len(self.hyperspectral_images)

    def __getitem__(self, i):
        
        hyperspectral_image = tiff.imread(self.hyperspectral_images[i]).astype(np.float32)  
        mask_image = tiff.imread(self.mask_images[i]).astype(np.float32) 
        
        # shape: (channels, height, width) dtype: uint8
        # Feature scaling / data normalization for each of the 25 channels.
        for c in range(hyperspectral_image.shape[0]):
            channel = hyperspectral_image[c]
            hyperspectral_image[c] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)

        # shape: (height, width) dtype: uint8
        # Channels are not stored for masks, therefore we just check if an image
        # is >0 / white (is road) or =0 / black (not road).
        mask_image = (mask_image > 0).astype(np.float32)

        # Pytorch and U-net expect a rank 3 tensor, so pad a value for mask images.
        return torch.tensor(hyperspectral_image), torch.tensor(mask_image).unsqueeze(0)


@app.command()
def main():
    return 

if __name__ == "__main__":
    app()
