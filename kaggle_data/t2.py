"""
    CS 487 Group Project Semantic Segmentation MRI image tumor prediction

    Note - can take awhile to run

    Mason Eiland
    Matthew Montoya
    
    stage 4

"""
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import json
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm.autonotebook import tqdm
import cv2
import os
import matplotlib.patches as patches
import random
import torch
import numpy as np
import skimage.draw
import tifffile
import shutil
from torch import nn


"""print the overall structure of the data"""
def print_struct(d, indent=0): 
    if isinstance(d, dict):
        for key, value in d.items():
            print(' ' * indent + str(key))
            print_struct(value,indent+1)
    elif isinstance(d, list):
        print(' ' * indent + "list of length {} containing : ".format(len(d)))
        if d:
            print_struct(d[0], indent+1)

        
""" Match images with corresponding annotation and display them"""
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
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        if display_type in ['seg', 'both']:
            for seg in ann['segmentation']:
                poly = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                #print(poly)
                polygon = patches.Polygon(poly, closed=True, edgecolor=color, fill=False)
                ax.add_patch(polygon)
"""
    create a plot to display images with coco annotations

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



"""  create the mask for the tumor segmentation """

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




""" creates new folders for data so we dont overwrite the original dataset"""
def create_files(json_file, mask_output_folder, image_output_folder, original_image_dir):

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




"""
    function to remove an image that has no corresponding mask
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
            


"""
    loads and returns image and mask pair data 
"""
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
    


"""
    loads image and mask pair data and does transformations
"""
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
    

""" Used in CustomDataset_general for image transformation, resizes, creates a tensor, normalizes, and makes sure data is between 0 and 1"""
image_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
    transforms.Lambda(lambda x: x.clamp(0, 1))
])


C = 1  # number input channels
n_filters = 32 

""" setting loss function"""
loss_fn = nn.BCEWithLogitsLoss()

""" defining Convolutional Neural Net helper function creates hidden layer"""
def cnnLayer(in_filters, out_filters, kernel_size = 3):
    padding = kernel_size // 2
    return nn.Sequential(
                        nn.Conv2d(in_filters, out_filters, kernel_size, padding=padding), 
                         nn.BatchNorm2d(out_filters),
                         nn.LeakyReLU()
                         )

"""creating model """
model = nn.Sequential(
    cnnLayer(C, n_filters), # first layer 
    *[cnnLayer(n_filters, n_filters) for _ in range(5)], # make 5 more layers
    nn.Conv2d(n_filters, 1, (3, 3), padding=1)

)


"""optimizer"""
from torch.optim import lr_scheduler

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


""" define conditions for early stop"""
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



"""performs a training step for the model -- iterates over data, calculates loss, and updates parameters, returns the average loss"""
def train_step(model:torch.nn.Module, dataloader:torch.utils.data.DataLoader, loss_fn:torch.nn.Module, optimizer:torch.optim.Optimizer, device='cpu'):
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

        """FIX"""
        #y_pred_prob = y_pred_logits.softmax(dim=1)

        #y_pred_target = y_pred_prob.argmax(dim=1)

        # print(f'This is true label: {y}')
        # print(f'This is predicted label: {y_pred_target}')


        #train_accuracy += (y_pred_target == y).sum().item()
    lr_scheduler.step()

    train_loss = train_loss / len(dataloader)

    #train_accuracy = train_accuracy / len(dataloader)


    return train_loss, train_accuracy

"""evaluates model on validation data and returns average loss"""
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

            """FIX"""
            #y_pred_prob = y_pred_logits.softmax(dim=1)
            #y_pred_target = y_pred_prob.argmax(dim=1)

            #val_accuracy += (y_pred_target == y).sum().item()
    val_loss = val_loss / len(dataloader)
    #val_accuracy = val_accuracy / len(dataloader)

    return val_loss, val_accuracy


""" trains the neural network for a given number of epochs for each epoch runs train_step updates parameters then runs val_step to see how model does on validation data"""
def Train(model:torch.nn.Module, train_dataloader:DataLoader, val_dataloader:DataLoader, loss_fn:torch.nn.Module, optimizer:torch.optim.Optimizer, early_stopping, epochs:int=10, device='cpu'):
    
    results = {
        'train_loss' : [],
        'train_accuracy': [],
        'val_loss' : [],
        'val_accuracy' : []
    }

    for epoch in tqdm(range(epochs)):

        train_loss, train_accuracy = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device)
        val_loss, val_accuracy = val_step(model=model, dataloader=val_dataloader, loss_fn=loss_fn, device=device)
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


"""plot the average loss for training and validation data"""
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





def main():
    
    #with open('dataset/valid/_annotations.coco.json', 'r') as file:
    #    data = json.load(file)

    """

    printing random images with bbox based on annotations

    """

    with open('dataset/valid/_annotations.coco.json', 'r') as file:
        annotations = json.load(file)


    image_dir = 'dataset/valid'

    all_image_files = [os.path.join(image_dir, img['file_name']) for img in annotations['images']]

    random_image_files = random.sample(all_image_files, 4)

    display_type = 'seg'
    display_images_with_coco_annotations(random_image_files, annotations, display_type)


    """creating new test directory with corresponding masks"""
    original_image_dir = 'dataset/test'

    json_file = 'dataset/test/_annotations.coco.json'

    mask_output_folder = 'test2/masks'

    image_output_folder = 'test2/images'

    create_files(json_file, mask_output_folder, image_output_folder, original_image_dir)



    """ creating new training directory with corresponding masks"""
    original_image_dir = 'dataset/train'

    json_file = 'dataset/train/_annotations.coco.json'

    mask_output_folder = 'train2/masks'

    image_output_folder = 'train2/images'

    create_files(json_file, mask_output_folder, image_output_folder, original_image_dir)



    """creating new validation directory with corresponding masks"""
    original_image_dir = 'dataset/valid'

    json_file = 'dataset/valid/_annotations.coco.json'

    mask_output_folder = 'valid2/masks'

    image_output_folder = 'valid2/images'

    create_files(json_file, mask_output_folder, image_output_folder, original_image_dir)

    """ removing the image with no corresponding mask"""
    folder1_path = 'train2/images'
    folder2_path = 'train2/masks'
    compare_folder_and_delete(folder1_path, folder2_path)

    train_path = 'train2'
    test_path = 'test2'
    valid_path = 'valid2'

    """transforming data"""
    train_dataset = CustomDataset_general(train_path, transform=image_transform)
    valid_dataset = CustomDataset_general(valid_path, transform=image_transform)
    test_dataset = CustomDataset_general(test_path, transform=image_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    """setting early stopping condition"""
    early_stopping = EarlyStopping(patience=10, delta=0.)

    epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\nStarting Model Training...\n")

    results = Train(model.to(device), train_dataloader=train_loader, val_dataloader=valid_loader, loss_fn=loss_fn, optimizer=optimizer, early_stopping=early_stopping, device=device, epochs=epochs)


    loss_and_metric_plot(results)
    plt.show()


if __name__ == "__main__":
    main()