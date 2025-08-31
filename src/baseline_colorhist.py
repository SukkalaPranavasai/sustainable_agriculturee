#!/usr/bin/env python3
"""
Baseline Color Histogram Model for Leaf Disease Detection

This script implements a baseline approach using color histogram features
and logistic regression for the PlantVillage dataset.
"""

import os
import argparse
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def load_images_and_labels(data_dir, max_images=None):
    """
    Load images and labels from the PlantVillage dataset.
    
    Args:
        data_dir (str): Path to the PlantVillage dataset
        max_images (int): Maximum number of images to load per class
    
    Returns:
        tuple: (images, labels, class_names)
    """
    images = []
    labels = []
    class_names = []
    
    # Get all class folders
    class_folders = [d for d in os.listdir(data_dir) 
                    if os.path.isdir(os.path.join(data_dir, d))]
    class_folders.sort()
    
    print(f"Found {len(class_folders)} classes")
    
    for class_idx, class_name in enumerate(class_folders):
        class_path = os.path.join(data_dir, class_name)
        
        # Get all image files
        image_files = (glob.glob(os.path.join(class_path, "*.jpg")) + 
                      glob.glob(os.path.join(class_path, "*.jpeg")) + 
                      glob.glob(os.path.join(class_path, "*.png")))
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"Loading {len(image_files)} images from class '{class_name}'")
        
        for image_file in tqdm(image_files, desc=f"Loading {class_name}"):
            try:
                # Load and resize image
                img = cv2.imread(image_file)
                if img is not None:
                    img = cv2.resize(img, (64, 64))  # Resize for consistency
                    images.append(img)
                    labels.append(class_idx)
                    if class_name not in class_names:
                        class_names.append(class_name)
            except Exception as e:
                print(f"Error loading {image_file}: {e}")
                continue
    
    return np.array(images), np.array(labels), class_names


def extract_color_histogram_features(images, bins=16):
    """
    Extract color histogram features from images.
    
    Args:
        images (np.array): Array of images
        bins (int): Number of bins per color channel
    
    Returns:
        np.array: Feature matrix
    """
    print("Extracting color histogram features...")
    features = []
    
    for img in tqdm(images, desc="Extracting features"):
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Calculate histograms for each channel
        hist_r = cv2.calcHist([img_rgb], [0], None, [bins], [0, 256]).flatten()
        hist_g = cv2.calcHist([img_rgb], [1], None, [bins], [0, 256]).flatten()
        hist_b = cv2.calcHist([img_rgb], [2], None, [bins], [0, 256]).flatten()
        
        # Normalize histograms
        hist_r = hist_r / (hist_r.sum() + 1e-8)
        hist_g = hist_g / (hist_g.sum() + 1e-8)
        hist_b = hist_b / (hist_b.sum() + 1e-8)
        
        # Concatenate features
        feature_vector = np.concatenate([hist_r, hist_g, hist_b])
        features.append(feature_vector)
    
    return np.array(features)


def train_baseline_model(X_train, X_test, y_train, y_test, class_names):
    """
    Train and evaluate the baseline logistic regression model.
    
    Args:
        X_train, X_test: Training and test feature matrices
        y_train, y_test: Training and test labels
        class_names: List of class names
    
    Returns:
        LogisticRegression: Trained model
    """
    print("\nTraining baseline model...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression
    model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nBaseline Model Results:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    return model, scaler


def main():
    """Main function to run the baseline experiment."""
    parser = argparse.ArgumentParser(
        description="Baseline Color Histogram Model for Leaf Disease Detection"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="../data/raw/PlantVillage",
        help="Path to PlantVillage dataset directory"
    )
    parser.add_argument(
        "--max_images", 
        type=int, 
        default=None,
        help="Maximum number of images to load per class"
    )
    parser.add_argument(
        "--test_size", 
        type=float, 
        default=0.2,
        help="Proportion of data to use for testing"
    )
    parser.add_argument(
        "--random_state", 
        type=int, 
        default=42,
        help="Random state for reproducibility"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Baseline Color Histogram Model for Leaf Disease Detection")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Max images per class: {args.max_images or 'All'}")
    print(f"Test size: {args.test_size}")
    print(f"Random state: {args.random_state}")
    print("=" * 60)
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist!")
        return
    
    # Load images and labels
    print("\nLoading dataset...")
    images, labels, class_names = load_images_and_labels(args.data_dir, args.max_images)
    
    if len(images) == 0:
        print("Error: No images loaded!")
        return
    
    print(f"\nDataset loaded successfully:")
    print(f"Total images: {len(images)}")
    print(f"Total classes: {len(class_names)}")
    print(f"Image shape: {images[0].shape}")
    
    # Extract features
    features = extract_color_histogram_features(images, bins=16)
    print(f"Feature matrix shape: {features.shape}")
    print(f"Features per image: {features.shape[1]}")
    
    # Split data
    print(f"\nSplitting data (train: {1-args.test_size:.1%}, test: {args.test_size:.1%})...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, 
        test_size=args.test_size, 
        random_state=args.random_state,
        stratify=labels
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train and evaluate model
    model, scaler = train_baseline_model(X_train, X_test, y_train, y_test, class_names)
    
    # Save model and scaler
    import joblib
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/baseline_colorhist_model.pkl")
    joblib.dump(scaler, "../models/baseline_colorhist_scaler.pkl")
    print(f"\nModel and scaler saved to ../models/")
    
    print("\n" + "=" * 60)
    print("Baseline experiment completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
