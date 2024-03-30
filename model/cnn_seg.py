import torch
import numpy as np


def create_mask(image: dict, img_annotations: dict, out_dir):
    mask = np.zeros((image["height"], image["width"]), dtype=np.uint8) # mask is all 0s # all "non-tumor" (white)

    for annotation in img_annotations:
        if annotation['image_id'] == image['id']:
            print(f'Processing annotation for image {image['file_name']}: {annotation}')

    for seg_idx, seg in enumerate(annotation['segmentation']):

        # ycoord,xcoord = = skimage.draw.polygon(seg[1::2], seg[0::2], mask_np.shape)
        seg_mask = np.zeros_like(mask, dtype=np.uint8)
        # seg_mask[ycoord, xcoord] = 255 # segmentation is cancerous (black)
def main():
    pass


if __name__ == "__main__":
    main()
