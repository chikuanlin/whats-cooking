import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.path.pardir))
sys.path.append(PARENT_DIR)
sys.path.append(BASE_DIR)
from dataset.dataset import WhatsCookingDataset, \
    WhatsCookingStemmedSeparatedDataset
import numpy as np
import matplotlib.pyplot as plt
import utilities.tSNE_util as tSNE_util
from processors.simple_ingredients_encoder import SimpleIngredientsEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


if __name__ == "__main__":
    """Feature visualization using various encoding methods."""

    # Ingredient separated dataset
    dataset = WhatsCookingDataset(file_path='../dataset/train.json')
    dataset_separated = WhatsCookingStemmedSeparatedDataset(stem=False, file_path='../dataset/train.json')
    x_as_text = [' '.join(cuisine.ingredients).lower() for cuisine in dataset_separated.cuisines]
    
    # Embedding with various encoders
    y = np.array([cuisine.cuisine for cuisine in dataset.cuisines])
    onehot_encoder = SimpleIngredientsEncoder()
    tfidf_encoder = TfidfVectorizer(binary=True)
    x = onehot_encoder.fit_transform(dataset)
    x_separated = onehot_encoder.fit_transform(dataset_separated)
    x_tfidf = tfidf_encoder.fit_transform(x_as_text).astype('float16')

    # Visualize using tSNE
    sample_num = 10000
    np.random.seed(0)
    smp_idx = np.random.choice(x.shape[0], size=sample_num, replace=False)

    fig_dir = 'feature_visualization'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    fig, ax = plt.subplots(figsize=(16,10))
    tSNE_util.plot_tSNE(
        ax, x[smp_idx, :], y[smp_idx],
        fig_path=os.path.join(fig_dir, 'tSNE-one hot.eps'))

    fig, ax = plt.subplots(figsize=(16,10))
    tSNE_util.plot_tSNE(
        ax, x_separated[smp_idx, :], y[smp_idx], 
        fig_path=os.path.join(fig_dir, 'tSNE-one hot separated data.eps'))

    fig, ax = plt.subplots(figsize=(16,10))
    tSNE_util.plot_tSNE(
        ax, x_separated[smp_idx, :], y[smp_idx], 
        fig_path=os.path.join(fig_dir, 'tSNE-tfidf separated data.eps'))
