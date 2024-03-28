import warnings
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import csv
import numpy as np
from torch.utils.data import Dataset
import random
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define transform for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to fit ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet mean and std
    transforms.Lambda(lambda x: x.unsqueeze(0)) 
])


class Patient:
    def __init__(self, ID, age, gender, lymph_count, images_path, label=None, model=None, embedding_dim=512):
        self.ID = ID
        self.age = age
        self.gender = gender
        self.lymph_count = lymph_count
        self.images_path = images_path  
        self.label = label  
        self.model = model
        self.embedding_dim = embedding_dim
        self.image_list = self.load_images()  
        self.features = self.extract_features()

    def load_images(self):
        if self.model is None:
            return(None) # no need for the images
        image_list = []
        for image_file in os.listdir(self.images_path):
            image_path = os.path.join(self.images_path, image_file)
            image = Image.open(image_path).convert("RGB")
            image = transform(image)
            image_list.append(image)
        return image_list

    def extract_features(self):
        if self.model is not None:  # If a model is provided, extract features using the model
            features = []
            for image in self.image_list:
                with torch.no_grad():
                    image = torch.tensor(image).to(device)
                    feature = self.model(image).cpu().numpy()[0] 
                    features.append(feature)
            age_embedding = torch.zeros(self.embedding_dim)  # Create an embedding for age
            lymph_count_embedding = torch.zeros(self.embedding_dim)  # Create an embedding  for lymph_count
            age_embedding[int(self.age) % self.embedding_dim] = 1 
            lymph_count_embedding[int(self.lymph_count) % self.embedding_dim] = 1  
            features.append(age_embedding.numpy())
            features.append(lymph_count_embedding.numpy())
            self.features = features
            
        else:  # If no model is provided, load features from CSV file
            if self.label is not None:
                csv_path = os.path.join(os.path.join(os.path.join(os.path.dirname(self.images_path),"../"),"train_features"), f"{self.ID}_features.csv")
                self.features = self.load_features_from_csv(csv_path)
            else: 
                csv_path = os.path.join(os.path.join(os.path.join(os.path.dirname(self.images_path),"../"),"test_features"), f"{self.ID}_features.csv")
                self.features = self.load_features_from_csv(csv_path)
                
    def save_features_to_csv(self):
        if self.label is not None:
            feature_folder = "train_features"
        else:
            feature_folder = "test_features"
        
        feature_dir = os.path.join(os.path.dirname(self.images_path)+"/../../", f"{feature_folder}")
        os.makedirs(feature_dir, exist_ok=True)  # Create feature folder if it doesn't exist
        
        csv_path = os.path.join(feature_dir, f"{self.ID}_features.csv")
        if os.path.exists(csv_path):
            os.remove(csv_path)  # Remove the file if it already exists
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Feature"] * len(self.features[0])) 
            writer.writerows(self.features)
    
    def load_features_from_csv(self, csv_path):
        features = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            for row in reader:
                feature = [float(value) for value in row]
                features.append(feature)
        features = torch.tensor(features, dtype=torch.float32)
        return features
    
    def __str__(self):
        return f"Patient ID: {self.ID}, Age: {self.age}, Gender: {self.gender}, Lymphocyte Count: {self.lymph_count}, Label: {self.label}, Number of Images: {len(self.image_list)}"




class PatientDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.indices = list(range(len(data))) 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        idx = self.indices[idx]
        features, label = self.data[idx]
        return torch.from_numpy(np.array(features)), torch.from_numpy(np.array([label]))

    def shuffle(self):
        random.shuffle(self.indices)