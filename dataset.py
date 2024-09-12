from torch.utils.data import DataLoader,Dataset
import os
import cv2
import torch
from PIL import Image
class CustomData(Dataset):
    def __init__(self,data_dir,transform = None) -> None:
        self.dictionary = {
            'Bicycle':0,
            'Car':1,
            'Motorcycle':2,
            'Bus':3,
            'Truck':4
        }
        self.data_dir = data_dir
        self.file_paths = []
        self.labels = []
        self.transform = transform
        for cls in os.listdir(data_dir):
            for filename in os.listdir(os.path.join(data_dir,cls)):
                filepath = os.path.join(os.path.join(data_dir,cls),filename)
                self.file_paths.append(filepath)
                self.labels.append(self.dictionary[cls])
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.file_paths[index]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.labels[index])
        return image,label
# data = CustomData(data_dir='G:/My Drive/Data/vehicledataset/train')

# print(data[0])