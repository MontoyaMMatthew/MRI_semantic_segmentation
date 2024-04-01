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

    # UNET MODEL
    model = nn.Sequential()


if __name__ == "__main__":
    main()
