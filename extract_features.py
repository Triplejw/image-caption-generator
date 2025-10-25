import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============ Feature Extractor with Spatial Features ============

class FeatureExtractor(torch.nn.Module):
    """Extract spatial features from ResNet for attention"""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Load pretrained ResNet50
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        
        # Remove average pooling and FC layer to keep spatial features
        # We want 7x7 spatial features instead of 1x1
        modules = list(resnet.children())[:-2]  # Remove avgpool and fc
        self.resnet = torch.nn.Sequential(*modules)
        
        # Adaptive pooling to ensure 7x7 output
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((7, 7))
        
    def forward(self, images):
        """
        Extract spatial features
        Output shape: (batch_size, 2048, 7, 7)
        """
        with torch.no_grad():
            features = self.resnet(images)
            features = self.adaptive_pool(features)
        return features


# ============ Extract and Save Features ============

def extract_all_features():
    """Extract features for all images and save to disk"""
    
    print("="*60)
    print("EXTRACTING IMAGE FEATURES WITH RESNET-50")
    print("="*60)
    
    # Load captions file to get all image names
    df = pd.read_csv('data/captions.txt')
    unique_images = df['image'].unique()
    
    print(f"\nTotal unique images: {len(unique_images)}")
    
    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Initialize feature extractor
    extractor = FeatureExtractor().to(device)
    extractor.eval()
    
    # Create directory for features
    os.makedirs('data/features', exist_ok=True)
    
    # Extract features for each image
    print("\nExtracting features...")
    features_dict = {}
    
    for img_name in tqdm(unique_images):
        img_path = os.path.join('data/Images', img_name)
        
        try:
            # Load and transform image
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            
            # Extract features
            with torch.no_grad():
                features = extractor(image)
                # Shape: (1, 2048, 7, 7)
                # Reshape to (49, 2048) - 49 spatial locations
                features = features.squeeze(0).permute(1, 2, 0)  # (7, 7, 2048)
                features = features.reshape(-1, 2048)  # (49, 2048)
                
            features_dict[img_name] = features.cpu().numpy()
            
        except Exception as e:
            print(f"\nError processing {img_name}: {e}")
            continue
    
    # Save features
    print("\nSaving features to disk...")
    with open('data/features/resnet_features.pkl', 'wb') as f:
        pickle.dump(features_dict, f)
    
    print(f"\n✓ Features extracted and saved!")
    print(f"Feature shape per image: (49, 2048)")
    print(f"Total size: ~{len(features_dict) * 49 * 2048 * 4 / (1024**2):.1f} MB")
    print("="*60)


if __name__ == "__main__":
    extract_all_features()
