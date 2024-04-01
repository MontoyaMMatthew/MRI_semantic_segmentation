import torch
import numpy as np
import skimage
from pathlib import Path
import tifffile as tif
import matplotlib.pyplot as plt
import click
import json


def create_mask(image: dict, img_annotations: dict, out_dir, max_print=3):
    mask = np.zeros(
        (image["height"], image["width"]), dtype=np.uint8
    )  # mask is all 0s # all "non-tumor" (white)

    for annotation in img_annotations:
        if annotation["image_id"] == image["id"]:
            print(f'Processing annotation for image {image["file_name"]}: {annotation}')

    printed_masks = 0
    for seg_idx, seg in enumerate(annotation["segmentation"]):

        ycoord, xcoord = skimage.draw.polygon(
            seg[1::2], seg[0::2], mask.shape
        )  # row->ycoord, col->xcoord
        seg_mask = np.zeros_like(mask, dtype=np.uint8)
        seg_mask[ycoord, xcoord] = 255  # segmentation is cancerous (black)
        mask_out_path = Path.joinpath(
            out_dir, f"{image['file_name'].replace('.jpg', '')}_seg_{seg_idx}.tif"
        )
        tif.imwrite(
            file=mask_out_path, data=seg_mask
        )  # creating images of the segmentation

        # Display the segmentation mask
        if printed_masks < max_print:
            plt.figure(figsize=(8, 8))
            plt.imshow(seg_mask, cmap="gray")
            plt.title(f"Segmentation Mask for {image['file_name']} Segment {seg_idx}")
            plt.axis("off")  # Hide axis labels
            plt.show()
            printed_masks += 1


@click.command()
@click.option("--coco", required=True, type=str)
@click.option("--dataset-root-dir", "-ds-root", required=True, type=str)
def main(coco, ds_root):
    with open(coco, "r") as f:
        data = json.load(f)
        images = data["images"]
        annotations = data["annotations"]

        ds_root_path = Path(ds_root)
        mask_out_dir = ds_root_path.joinpath("mask")
        imgs_out = ds_root_path.joinpath("images_out")

        if not Path.exists(mask_out_dir):
            ds_root_path.mkdir("mask")
        if not Path.exists(imgs_out):
            ds_root_path.mkdir("images_out")

    for img in images:
        create_mask(image=img, img_annotations=annotations, out_dir=imgs_out)
        print("creating mask")


if __name__ == "__main__":
    main()
