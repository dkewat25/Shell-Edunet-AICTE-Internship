# Shell-Edunet-AICTE-Internship
This project aims to build a garbage classification system using EfficientNetV2B2 to automate waste segregation, improve recycling, and support environmental conservation.

✅ Week 1: Dataset Setup & Preprocessing
Acquired the Trash Type Image Dataset from Kaggle (2527 images across 6 classes).

Implemented data loading using image_dataset_from_directory, splitting it into training and validation sets.

Resized all images to 224x224, normalized pixel values, and applied cache() and prefetch() for faster loading.

Ensured environment compatibility and resolved path issues.

🚀 Week 2: Model Optimization & Performance Boost
Increased input size to 260x260 to match EfficientNetV2B2 requirements.

Added preprocess_input() for accurate image normalization.

Optimized data pipelines across all datasets using cache(), shuffle(), and prefetch().

Introduced ModelCheckpoint, ReduceLROnPlateau, and EarlyStopping for efficient and stable training.

Enabled mixed_float16 for faster GPU training.

Optional upgrade to AdamW optimizer for better generalization.

📈 Result: Improved accuracy, faster training, better generalization, and a robust pipeline ready for deployment.


