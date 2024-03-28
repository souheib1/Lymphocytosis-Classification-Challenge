import pandas as pd
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.models as models
import csv
import cv2
from skimage.morphology import binary_dilation,disk
warnings.filterwarnings("ignore")



def preprocess_metadata(path_data='../dataset/'):
    """
    Preprocesses metadata from clinical annotation CSV file.
    """
    df = pd.read_csv(path_data+"/clinical_annotation.csv",index_col="ID")  
    df['DOB'] = pd.to_datetime(df['DOB'])
    current_year = 2024
    df['AGE'] = current_year - df['DOB'].dt.year
    sex_mapping = {'M': 0, 'F': 1,'f':1}
    df['GENDER'] = df['GENDER'].map(sex_mapping)
    df =  df.drop(df.columns[0], axis=1)
    df.drop('DOB', axis=1, inplace=True)
    train_set = df[df['LABEL'].isin([0, 1])]
    test_set = df[df['LABEL'] == -1]

    # Add a column for the train set to indicate the subgroup used for stratification
    thresh_lymph_count = train_set["LYMPH_COUNT"].median()
    thresh_age = train_set["AGE"].median()

    # Store the 3 binary variables
    train_meta_binary = pd.DataFrame(index=train_set.index)
    train_meta_binary["LYMPH_COUNT"] = train_set["LYMPH_COUNT"] > thresh_lymph_count
    train_meta_binary["AGE"] = train_set["AGE"] > thresh_age
    train_meta_binary["LABEL"] = train_set["LABEL"]
    train_set["SUBGROUP"] = train_meta_binary["LYMPH_COUNT"] + 2*train_meta_binary["AGE"] + 4*train_meta_binary["LABEL"]

    return train_set, test_set



def plot_statistics_by_label(df, columns=['GENDER', 'LYMPH_COUNT','AGE'], label_column='LABEL'):
    """
    Plots histograms of features.
    """
    # Filter data for labels 0 and 1
    label_df_0 = df[df['LABEL'] == 0]
    label_df_1 = df[df['LABEL'] == 1]

    for column in columns:
        plt.figure(figsize=(15, 5))

        # Plot histogram for label 0
        plt.subplot(1, 2, 1)
        sns.histplot(data=label_df_0[column], kde=True, color='blue')
        plt.title(f"Histogram of {column} (Label 0)")

        # Plot histogram for label 1
        plt.subplot(1, 2, 2)
        sns.histplot(data=label_df_1[column], kde=True, color='orange')
        plt.title(f"Histogram of {column} (Label 1)")

        plt.tight_layout()
        plt.show()
        
def compute_correlation_heatmap(df, columns=['LABEL', 'LYMPH_COUNT', 'AGE','GENDER']):
    """
    Computes the correlation heatmap between features.
    """
    correlation_matrix = df[columns].corr()
    plt.figure(figsize=(5, 3))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
    plt.title("Correlation Heatmap")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()
    
    
def k_means_binary_seg(image):
    image = (image * 255).astype(np.uint8)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,1]
    _, binary_seg = cv2.threshold(image_hsv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #thresholding
    binary_seg = binary_dilation(binary_seg, disk(10)) #dilatation
    result_image = cv2.bitwise_and(image, image, mask=np.uint8(binary_seg))
    return result_image
