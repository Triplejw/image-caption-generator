import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

print("="*60)
print("CREATING TRAIN/VALIDATION/TEST SPLITS")
print("="*60)

# Load captions
df = pd.read_csv('data/captions.txt')
print(f"\nTotal captions: {len(df)}")

# Get unique images
unique_images = df['image'].unique()
print(f"Unique images: {len(unique_images)}")

# Standard Flickr8k split: 6000 train, 1000 val, 1000 test
train_images, temp_images = train_test_split(
    unique_images, test_size=2000, random_state=42
)
val_images, test_images = train_test_split(
    temp_images, test_size=1000, random_state=42
)

print(f"\nTrain images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
print(f"Test images: {len(test_images)}")

# Create splits for captions
train_df = df[df['image'].isin(train_images)]
val_df = df[df['image'].isin(val_images)]
test_df = df[df['image'].isin(test_images)]

print(f"\nTrain captions: {len(train_df)}")
print(f"Validation captions: {len(val_df)}")
print(f"Test captions: {len(test_df)}")

# Save splits
os.makedirs('data/splits', exist_ok=True)
train_df.to_csv('data/splits/train_captions.txt', index=False)
val_df.to_csv('data/splits/val_captions.txt', index=False)
test_df.to_csv('data/splits/test_captions.txt', index=False)

print("\n✓ Splits saved to data/splits/")
print("="*60)
