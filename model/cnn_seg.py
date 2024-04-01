import torch
import torch.nn as nn
import numpy as np
import skimage.draw
from pathlib import Path
import tifffile as tif
import matplotlib.pyplot as plt
import click
import json


# PRE: image information from coco json file('image', and 'annotations'), output directory for segmentations
# POST: returns mask path, creates segmetation masks and saves in output directory as tiff file
def create_mask(image: dict, img_annotations: list, out_dir: Path):
    """creates Ground Truth for semantic Image Segmentation and Classification"""
    # Initialize the mask with zeros
    mask = np.zeros((image["height"], image["width"]), dtype=np.uint8)

    # Process each annotation for the image
    for annotation in img_annotations:
        if annotation["image_id"] == image["id"]:
            print(f'Processing annotation for image {image["file_name"]}: {annotation}')

            for seg in annotation["segmentation"]:
                # Get the y (row) and x (column) coordinates of the polygon
                # segmentation vertices are y1,x1,y2,x2...yn,xn
                ycoord, xcoord = skimage.draw.polygon(seg[1::2], seg[0::2], mask.shape)

                # cancerous location is white
                mask[ycoord, xcoord] = 255

    mask_out_path = Path.joinpath(out_dir, f"{Path(image['file_name']).stem}_mask.tif")
    tif.imwrite(file=mask_out_path, data=mask)

    return mask_out_path


@click.command()
@click.option("--coco", required=True, type=str)
@click.option("--dataset-root-dir", "-ds-root", "ds_root", required=True, type=str)
def main(coco, ds_root):
    with open(coco, "r") as f:
        data = json.load(f)
        images = data["images"]
        annotations = data["annotations"]

        ds_root_path = Path(ds_root)
        mask_out_dir = ds_root_path.joinpath("mask")
        imgs_out = ds_root_path.joinpath("images_out")

        if not mask_out_dir.exists():
            mask_out_dir.mkdir(parents=True, exist_ok=True)
        if not imgs_out.exists():
            imgs_out.mkdir(parents=True, exist_ok=True)

    for img in images:
        create_mask(image=img, img_annotations=annotations, out_dir=imgs_out)
        print("creating mask")

    # Module channels
    module_in = 1  # gray scale images
    module_out = 1  # binary segmentation

    # shape of data (batchsize,)


class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # The following convolutions will be structured to follow the UNET Model Architecture
        # Three contracting blocks
        # self.conv1 =
        # self.conv2 =
        # self.conv3 =

        # Three expanding blocks
        # self.upconv3 =
        # self.upconv2 =
        # self.upconv1 =

    def __call__(self, x):
        pass

    def contract_block(in_channels, out_channels, kernel_size, padding):
        """
        This block downsizes the image resolution via pooling, and extracts features through its convolutions.
        A contract block consists of:
            2 convolution layers, each of which are normalized and activated by the reLu function
            1 pooling layer
        """
        contract = nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        """
        This block upscales the image to increase resolution.
        This block consists of:
            2 convolutions which are normalized and activated with reLu
            1 transpose layer (broadcasting to larger pixel resolution)
        """
        expand = nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=1, padding=padding
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_channels, out_channels, kernel_size, stride=1, padding=padding
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        )
        return expand


if __name__ == "__main__":
    main()
