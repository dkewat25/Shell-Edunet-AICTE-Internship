import tensorflow as tf
import os

# --- Configuration ---
# Path to your main dataset folder (adjust this path!)
# Make sure 'TrashType_Image_Dataset' is the folder containing your 'cardboard', 'glass', etc. subfolders.
DATA_DIR = 'C:/Users/Dishant/PycharmProjects/Capstone/Garbage_Collection/TrashType_Image_Dataset'

# Image dimensions that will be used for resizing
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
TARGET_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

# Batch size for training
BATCH_SIZE = 32

# Validation split percentage
VALIDATION_SPLIT = 0.2  # 20% of the data for validation

# --- Data Loading and Splitting ---

print(f"Loading data from: {DATA_DIR}")

# Create training dataset
train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='int', # 'int' for integer labels (0, 1, 2...), 'categorical' for one-hot
    validation_split=VALIDATION_SPLIT,
    subset='training',
    seed=42, # Set a seed for reproducibility
    image_size=TARGET_SIZE,
    batch_size=BATCH_SIZE
)

# Create validation dataset
val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='int',
    validation_split=VALIDATION_SPLIT,
    subset='validation',
    seed=42, # Use the same seed as training for consistent split
    image_size=TARGET_SIZE,
    batch_size=BATCH_SIZE
)

# --- IMPORTANT FIX: Get class names BEFORE mapping/caching/prefetching ---
class_names = train_ds_raw.class_names


# --- Basic Preprocessing (Normalization) ---
# Rescaling layer to normalize pixel values from [0, 255] to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Apply normalization to both datasets
train_ds = train_ds_raw.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds_raw.map(lambda x, y: (normalization_layer(x), y))

# --- Cache and Prefetch for Performance ---
# Use .cache() to keep images in memory after first epoch if dataset fits
# Use .prefetch() to overlap data preprocessing and model execution
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- Verification ---
print("\n--- Dataset Information ---")
print(f"Number of training batches: {len(train_ds)}")
print(f"Number of validation batches: {len(val_ds)}")
print(f"Detected classes: {class_names}") # Use the 'class_names' variable directly

# Get a sample batch to check shapes
for image_batch, labels_batch in train_ds.take(1):
    print(f"Image batch shape: {image_batch.shape}")
    print(f"Labels batch shape: {labels_batch.shape}")
    print(f"Sample labels (first 5): {labels_batch[:5].numpy()}")
    break
