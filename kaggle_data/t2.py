from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json

with open('dataset/valid/_annotations.coco.json', 'r') as file:
    data = json.load(file)

# for img in data['images'][:10]:
#     print(img['file_name'])

# for img in data['annotations'][:10]:
#     print(img['segmentation'])


def print_struct(d, indent=0): # print the overall structure of the data
    if isinstance(d, dict):
        for key, value in d.items():
            print(' ' * indent + str(key))
            print_struct(value,indent+1)
    elif isinstance(d, list):
        print(' ' * indent + "list of length {} containing : ".format(len(d)))
        if d:
            print_struct(d[0], indent+1)


#print_struct(data)
            
import cv2
import os
import matplotlib.patches as patches

def display_image_with_annotations(ax, image, annotations, display_type = 'both', colors=None):
    ax.imshow(image)
    ax.axis('off') # turn off axes

    if colors is None:
        colors = plt.cm.tab10

    for ann in annotations:
        category_id = ann['category_id']
        color = colors(category_id % 10)


        if display_type in ['bbox', 'both']:
            bbox = ann['bbox']
            #print(bbox)
            rect = patches.CirclePolygon((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        if display_type in ['seg', 'both']:
            for seg in ann['segmentation']:
                poly = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                #print(poly)
                polygon = patches.Polygon(poly, closed=True, edgecolor=color, fill=False)
                ax.add_patch(polygon)
"""
    create a plot to display images with bbox annotations

"""
def display_images_with_coco_annotations(image_paths, annotations, display_type='both', colors=None):
    fig, axs = plt.subplots(2, 2, figsize=(10,10))

    for ax, img_pth in zip(axs.ravel(), image_paths):
        image = cv2.imread(img_pth)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        img_filename = os.path.basename(img_pth)
        img_id = next(item for item in annotations['images'] if item["file_name"] == img_filename)['id']

        img_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == img_id]

        display_image_with_annotations(ax, image, img_annotations, display_type, colors)

    plt.tight_layout()
    plt.show()



"""

    printing random images with bbox based on annotations

"""

import random
with open('dataset/valid/_annotations.coco.json', 'r') as file:
    annotations = json.load(file)


image_dir = 'dataset/valid'

all_image_files = [os.path.join(image_dir, img['file_name']) for img in annotations['images']]

random_image_files = random.sample(all_image_files, 4)

display_type = 'seg'
display_images_with_coco_annotations(random_image_files, annotations, display_type)



"""  create the mask for the tumor segmentation """

import json
import numpy as np
import skimage.draw
import tifffile
import shutil


def create_mask(image_info, annotations, output_folder, max_print=3):

    mask_np = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

    object_number = 1

    printed_masks = 0

    for ann in annotations:
        if ann['image_id'] == image_info['id']:
            print(f'Processing annotation for image {image_info['file_name']}: {ann}')

            for seg_idx, seg in enumerate(ann['segmentation']):
                #print(f'Segmentation points: {seg}')

                rr, cc = skimage.draw.polygon(seg[1::2], seg[0::2], mask_np.shape)
                
                seg_mask = np.zeros_like(mask_np, dtype=np.uint8)

                seg_mask[rr, cc] = 255

                mask_path = os.path.join(output_folder, f"{image_info['file_name'].replace('.jpg', '')}_seg_{seg_idx}.tif")
                tifffile.imwrite(mask_path, seg_mask)

                #print(f'Saved segmentation mask for {image_info['file_name']} segmenation {seg_idx} to {mask_path}')

                # plt.imshow(seg_mask, cmap='gray')
                # plt.title(f"Segmentation Mask for {image_info['file_name']} Segment {seg_idx}")
                # plt.show()

                printed_masks += 1
                if printed_masks >= max_print:
                    return
        #print("All segmentation masks saved.")




""" def main"""


def main(json_file, mask_output_folder, image_output_folder, original_image_dir):

    with open(json_file, 'r') as f:
        data = json.load(f)

        images = data['images']
        annotations = data['annotations']


        if not os.path.exists(mask_output_folder):
            os.makedirs(mask_output_folder)

        if not os.path.exists(image_output_folder):
            os.makedirs(image_output_folder)

        for img in images:
            create_mask(img, annotations, mask_output_folder)

            original_image_path = os.path.join(original_image_dir, img['file_name'])

            new_image_path = os.path.join(image_output_folder, os.path.basename(original_image_path))
            shutil.copy2(original_image_path, new_image_path)



## creating new test directory with corresponding masks
# original_image_dir = 'dataset/test'

# json_file = 'dataset/test/_annotations.coco.json'

# mask_output_folder = 'test2/masks'

# image_output_folder = 'test2/images'

# main(json_file, mask_output_folder, image_output_folder, original_image_dir)



# ## creating new training directory with corresponding masks
# original_image_dir = 'dataset/train'

# json_file = 'dataset/train/_annotations.coco.json'

# mask_output_folder = 'train2/masks'

# image_output_folder = 'train2/images'

# main(json_file, mask_output_folder, image_output_folder, original_image_dir)



# ## creating new validation directory with corresponding masks
# original_image_dir = 'dataset/valid'

# json_file = 'dataset/valid/_annotations.coco.json'

# mask_output_folder = 'valid2/masks'

# image_output_folder = 'valid2/images'

# main(json_file, mask_output_folder, image_output_folder, original_image_dir)



"""
    Possibly need to remove an image that has no corresponding mask?
"""


def compare_folder_and_delete(folder1_path, folder2_path):
    folder1_items = os.listdir(folder1_path)
    folder2_items = os.listdir(folder2_path)

    for item1 in folder1_items:
        found = False
        for item2 in folder2_items:
            if item1[:4] == item2[:4]:
                found = True
                break
        if not found:
            print(f"Corresponding item for {item1} not found.")
            item1_path = os.path.join(folder1_path, item1)
            os.remove(item1_path)
            print(f"Deleted {item1}")


    for item2 in folder2_items:
        found = False
        for item1 in folder1_items:
            if item2[:4] == item1[:4]:
                found = True
                break
        if not found:
            print(f"Corresponding item for {item2} not found.")
            item2_path = os.path.join(folder2_path, item2)
            os.remove(item2_path)
            print(f"Deleted {item2}")



folder1_path = 'train2/images'
folder2_path = 'train2/masks'

compare_folder_and_delete(folder1_path, folder2_path)
            

class CustomDataset:
    def __init__(self, root_dir):
       self.root_dir = root_dir
       self.image_folder = os.path.join(root_dir, 'images')
       self.mask_folder = os.path.join(root_dir, 'masks')
       self.image_files = sorted(os.listdir(self.image_folder))
       self.mask_files = sorted(os.listdir(self.mask_folder))


    def __len__(self):
       return len(self.image_files)
    

    def __getitem__(self, idx):
        # read image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #read corresponding mask
        mask_name = self.mask_files[idx]
        mask_path = os.path.join(self.mask_folder, mask_name)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        return image, mask
    

dataset = CustomDataset('test2')
image, mask = dataset[0]
#print(image.shape, mask.shape)


fig, axs = plt.subplots(3, 2, figsize=(10, 15))

for i in range(3):
    image, mask = dataset[i]
    axs[i, 0].imshow(image)
    axs[i, 0].set_title('Image')
    axs[i, 0].axis('off')
    axs[i, 1].imshow(mask, cmap='gray')
    axs[i, 1].set_title('Mask')
    axs[i, 1].axis('off')

# plt.tight_layout()
# plt.show()



from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class CustomDataset_general(Dataset):
    def __init__(self, root_dir, transform=None):
       self.root_dir = root_dir
       self.image_folder = os.path.join(root_dir, 'images')
       self.mask_folder = os.path.join(root_dir, 'masks')
       self.image_files = sorted(os.listdir(self.image_folder))
       self.mask_files = sorted(os.listdir(self.mask_folder))
       self.transform = transform


    def __len__(self):
       return len(self.image_files)
    

    def __getitem__(self, idx):
        # read image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        image_gray = image.convert('L') # convert to grayscale

        #read corresponding mask
        mask_name = self.mask_files[idx]
        mask_path = os.path.join(self.mask_folder, mask_name)

        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image_gray = self.transform(image_gray)
            mask = self.transform(mask)


        return image_gray, mask
    

train_path = 'train2'
test_path = 'test2'
valid_path = 'valid2'


image_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
    transforms.Lambda(lambda x: x.clamp(0, 1))
])


train_dataset = CustomDataset_general(train_path, transform=image_transform)
valid_dataset = CustomDataset_general(valid_path, transform=image_transform)
test_dataset = CustomDataset_general(test_path, transform=image_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


x, y = next (iter(train_loader))
#print(x.shape, y.shape, type(x), type(y))


import torch

# num_ones = torch.eq(x, 1).sum().item()
# num_zeros = x.numel() - num_ones

# print(f"Number of ones in x: {num_ones}")
# print(f"Number of zeroes in x: {num_zeros}")


# num_ones = torch.eq(y, 1).sum().item()
# num_zeros = y.numel() - num_ones

# print(f"Number of ones in y: {num_ones}")
# print(f"Number of zeroes in y: {num_zeros}")

# convert tensors to numpy arrays and squeeze channel dimension
x_np = x.numpy().squeeze(1)
y_np = y.numpy().squeeze(1)

# plt.figure(figsize=(10, 5))

# for i in range(4):
#     plt.subplot(2, 4, i + 1)
#     plt.imshow(x_np[i], cmap='gray')
#     plt.title('Original Image')
#     plt.axis('off')


# for i in range(4):
#     plt.subplot(2, 4, i + 5)
#     plt.imshow(y_np[i], cmap='gray')
#     plt.title('Mask')
#     plt.axis('off')

# plt.tight_layout()
# plt.show()



from torch import nn

C = 1
n_filters = 32

loss_func = nn.BCEWithLogitsLoss()


def cnnLayer(in_filters, out_filters, kernel_size = 3):
    padding = kernel_size // 2
    return nn.Sequential(
                        nn.Conv2d(in_filters, out_filters, kernel_size, padding=padding), 
                         nn.BatchNorm2d(out_filters),
                         nn.LeakyReLU()
                         )

model = nn.Sequential(
    cnnLayer(C, n_filters),
    *[cnnLayer(n_filters, n_filters) for _ in range(5)],
    nn.Conv2d(n_filters, 1, (3, 3), padding=1)

)

from torch.optim import lr_scheduler

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)



class EarlyStopping:
    def __init__(self, patience:int=10, delta:float=0.001, path = 'best_model.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)

        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        else:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

early_stopping = EarlyStopping(patience=10, delta=0.)

from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

def train_step(model:torch.nn.Module, dataloader:DataLoader, loss_fn:torch.nn.Module, optimizer:torch.optim.Optimizer, device='cpu'):
    model.train()
    train_loss = 0
    train_accuracy = 0

    for batch, (X,y) in enumerate(dataloader):
        X = X.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        y_pred_logits = model(X)

        loss = loss_fn(y_pred_logits, y)

        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        # y_pred_prob = y_pred_logits.softmax(dim=1)

        # y_pred_target = y_pred_prob.argmax(dim=1)

        # print(f'This is true label: {y}')
        # print(f'This is predicted label: {y_pred_target}')

        # train_accuracy += balanced_accuracy_score(y.cpu().numpy(), y_pred_target.detach().cpu().numpy(), adjusted=True)

        lr_scheduler.step()

        train_loss = train_loss / len(dataloader)

        # train_accuracy = train_accuracy / len(dataloader)


        return train_loss, train_accuracy


def val_step(model:torch.nn.Module, dataloader:DataLoader, loss_fn:torch.nn.Module, device='cpu'):
    model.eval()

    val_loss = 0
    val_accuracy = 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate (dataloader):
            X = X.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred_logits = model(X)

            loss = loss_fn(y_pred_logits, y)
            val_loss += loss.item()


            # y_pred_prob = y_pred_logits.softmax(dim=1)
            # y_pred_target = y_pred_prob.argmax(dim=1)

            # val_accuracy += balanced_accuracy_score(y.cpu().numpy(), y_pred_target.detach().cpu().numpy(), adjusted=True)

            val_loss = val_loss / len(dataloader)
            # val_accuracy = val_accuracy / len(dataloader)

            return val_loss, val_accuracy


from tqdm.autonotebook import tqdm

def Train(model:torch.nn.Module, train_dataloader:DataLoader, val_dataloader:DataLoader, loss_fn:torch.nn.Module, optimizer:torch.optim.Optimizer, early_stopping, epochs:int=10, device='cpu'):
    
    results = {
        'train_loss' : [],
        'train_accuracy': [],
        'val_loss' : [],
        'val_accuracy' : []
    }

    for epoch in tqdm(range(epochs)):

        train_loss, train_accuracy = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device)
        val_loss, val_accuracy = train_step(model=model, dataloader=val_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device)
        print(f"Epoch : {epoch+1} | train_loss : {train_loss:.4f} | train_accuracy : {train_accuracy:.4f} | val_loss : {val_loss:.4f} | val_accuracy : {val_accuracy:.4f}")

        early_stopping(val_loss, model)

        if early_stopping.early_stop == True:
            print('Early Stopping')
            break
        
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_accuracy)
        results['val_loss'].append(val_loss)
        results['val_accuracy'].append(val_accuracy)


    return results




def manual_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


epochs = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'

results = Train(model.to(device), train_dataloader=train_loader, val_dataloader=valid_loader, loss_fn=loss_fn, optimizer=optimizer, early_stopping=early_stopping, device=device, epochs=epochs)




def loss_and_metric_plot(results:dict):
    
    plt.style.use('ggplot')
    training_loss = results['train_loss']

    validation_loss = results['val_loss']

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize= (9,3.8))
    ax = ax.flat


    ax[0].plot(training_loss, 'o-', markersize = 4, label = 'Train')
    ax[0].plot(validation_loss, label = 'Val')
    ax[0].set_title('Binary Loss', fontsize = 12, fontweight='bold', color='black')
    ax[0].set_xlabel("Epoch", fontsize = 10, fontweight = 'bold', color='black')
    ax[0].set_ylabel("loss", fontsize = 10, fontweight='bold', color = 'black')

    ax[0].legend()
    fig.show()


loss_and_metric_plot(results)
plt.show()