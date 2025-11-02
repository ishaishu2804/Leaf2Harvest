import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
from PIL import Image
import pandas as pd
import json

# Define the model architecture
def create_model(num_classes):
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Fourth Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')  # Use num_classes
    ])
    
    return model

def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return img_array

def main():
    print("[INFO] Setting up dataset...")
    # Define dataset path relative to the project root
    dataset_dir = os.path.join('data', 'plant_disease', 'Plant_leave_diseases_dataset_without_augmentation') # Corrected path

    # Check if the dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Dataset directory not found at {dataset_dir}.")
        print("Please ensure prepare_dataset.py was run and extracted the data correctly.")
        print("Expected structure: data/plant_disease/YOUR_DATASET_FOLDER/class1/... , etc.")
        return

    # Create training and validation datasets
    # Assuming the dataset is organized into subdirectories where each subdirectory is a class label
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32

    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
          dataset_dir,
          validation_split=0.2,
          subset="training",
          seed=123,
          image_size=(IMG_HEIGHT, IMG_WIDTH),
          batch_size=BATCH_SIZE)

        val_ds = tf.keras.utils.image_dataset_from_directory(
          dataset_dir,
          validation_split=0.2,
          subset="validation",
          seed=123,
          image_size=(IMG_HEIGHT, IMG_WIDTH),
          batch_size=BATCH_SIZE)

        class_names = train_ds.class_names
        num_classes = len(class_names)
        print(f"[INFO] Found {num_classes} classes: {class_names}")
        
        # Print a sample batch shape to verify data loading
        for image_batch, label_batch in train_ds.take(1):
            print(f"[INFO] Training batch image shape: {image_batch.shape}")
            print(f"[INFO] Training batch label shape: {label_batch.shape}")

    except Exception as e:
        print(f"[ERROR] Error loading dataset: {e}")
        print("Please check the dataset directory structure and ensure it contains subfolders for each class.")
        return


    print("[INFO] Creating model...")
    model = create_model(num_classes)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy', # Use sparse_categorical_crossentropy for integer labels
        metrics=['accuracy']
    )
    
    print("[INFO] Model created and compiled.")
    model.summary()

    print("[INFO] Training model...")
    # Add training steps
    EPOCHS = 10 # You can adjust the number of epochs
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    print("[INFO] Model training finished.")
    
    # Print training history
    print("[INFO] Training History:")
    print(history.history)

    # Save the model
    model_path = os.path.join('app', 'models', 'disease_model.h5')
    model.save(model_path)
    print(f"[INFO] Trained model saved to {model_path}")

    # Save class_names list
    class_names_path = os.path.join('app', 'models', 'class_names.json')
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f)
    print(f"[INFO] Class names saved to {class_names_path}")


if __name__ == "__main__":
    main() 