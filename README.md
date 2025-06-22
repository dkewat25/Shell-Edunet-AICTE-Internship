# Shell-Edunet-AICTE-Internship
This project develops a garbage classification system using the EfficientNetV2B2 model to automate waste segregation. It aims to improve recycling, enhance waste management, and support environmental conservation.

Week 1: Data Acquisition & Initial Preparation
This week focused on setting up the foundational data pipeline for our image classification project. The primary goal was to acquire the dataset, understand its structure, and implement robust data loading and preliminary preprocessing steps.

Key Achievements:
Dataset Acquisition: Successfully downloaded the Trash Type Image Dataset from Kaggle.
Data Structure Understanding: Verified that the dataset is organized with subfolders representing each trash category (e.g., cardboard, glass, metal, paper, plastic, trash), which is ideal for direct use with TensorFlow's image data loaders.
Data Loading & Splitting: Implemented efficient data loading using tf.keras.utils.image_dataset_from_directory. The full dataset (2527 images across 6 classes) was automatically split into:
Training Set: 2022 images
Validation Set: 505 images This ensures a robust separation of data for model training and hyperparameter tuning.
Basic Preprocessing: Integrated essential preprocessing steps directly into the data pipeline:
All images are resized to a uniform 224x224 pixel dimension.
Pixel values are normalized from the [0, 255] range to [0, 1].
Performance Optimization: Applied .cache() and .prefetch() to the datasets to significantly improve data loading efficiency during subsequent training phases.
Environment Setup: Confirmed local environment setup (PyCharm, TensorFlow) and addressed common path-related errors, ensuring the project can run smoothly on the local machine.
