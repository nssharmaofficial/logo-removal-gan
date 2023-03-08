import os

import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from setup import * 

def get_paths():
    setup = Setup()
    clean_paths = []
    logo_paths = []
    
    for logo_path in os.listdir(setup.logo_dir):
            
        # Extract the i-j part of the logo image filename
        clean_filename = logo_path.split("-")[0] + "-" + logo_path.split("-")[1] + ".jpg"
            
        # Find the path to the corresponding clean image
        clean_path = os.path.join(setup.clean_dir, clean_filename)
            
        # Find the path to the corresponding logo image
        logo_path = os.path.join(setup.logo_dir, logo_path)
            
        # Add the mapping to the dictionary
        clean_paths.append(clean_path)
        logo_paths.append(logo_path)
        

    # Divide into train/val set
    split_point = int(len(logo_paths)*0.7)
    train_clean_paths, val_clean_paths = clean_paths[:split_point], clean_paths[split_point:]
    train_logo_paths, val_logo_paths = logo_paths[:split_point], logo_paths[split_point:]
    
    return train_logo_paths, val_logo_paths, train_clean_paths, val_clean_paths


class No_Patch_Dataset(data.Dataset):
    
    def __init__(self, logo_paths, clean_paths, size=(512,512)):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]) 
        self.clean_paths = clean_paths
        self.logo_paths = logo_paths
        
        
    def __len__(self):
        return len(self.logo_paths)

 
    def __getitem__(self, idx):
        
        logo_path = self.logo_paths[idx]
        clean_path = self.clean_paths[idx]

        logo_image = Image.open(logo_path).convert('RGB')
        clean_image = Image.open(clean_path).convert('RGB')
        
        logo_tensor = self.transform(logo_image)
        clean_tensor = self.transform(clean_image)
        
        return logo_tensor, clean_tensor
        

def denormalize(image: torch.Tensor):
    inv_normalize = transforms.Normalize(
        mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
        std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
    )
    return (inv_normalize(image) * 255.).type(torch.uint8).permute(1, 2, 0).numpy()

    
def get_data_loader(dataset, batch_size=32):
    return DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        num_workers = 3,
        pin_memory = True,
        shuffle = True
    )
    
    
if __name__ == '__main__':
    
    setup = Setup()
        
    print('Loading dataset...')
    train_logo_paths, val_logo_paths, train_clean_paths, val_clean_paths = get_paths()
    val_dataset = No_Patch_Dataset(val_logo_paths, val_clean_paths, size = setup.whole_size)
    val_loader = get_data_loader(val_dataset, batch_size=setup.BATCH_show)
        
    x, y = next(iter(val_loader))
    
    for logo, clean in zip(x, y):
        logo = denormalize(logo)
        clean = denormalize(clean)
            
        _, ax = plt.subplots(1,2, figsize=(20,10))
        ax[0].imshow(logo)
        ax[0].title.set_text('Logo')
        ax[1].imshow(clean)
        ax[1].title.set_text('Clean')
        plt.show()
        plt.pause(1)