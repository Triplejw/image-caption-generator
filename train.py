import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import pickle
from tqdm import tqdm
import os
import random

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ============ STEP 1: Vocabulary Builder ============

class Vocabulary:
    def __init__(self, freq_threshold=5):
        """Build vocabulary from captions"""
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
        
    def __len__(self):
        return len(self.itos)
    
    def build_vocabulary(self, sentence_list):
        """Build vocabulary from list of sentences"""
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            for word in word_tokenize(sentence.lower()):
                frequencies[word] += 1
                
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
        
        print(f"Vocabulary size: {len(self.itos)}")
    
    def numericalize(self, text):
        """Convert text to numerical representation"""
        tokenized_text = word_tokenize(text.lower())
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]


# ============ STEP 2: Dataset Class ============

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5, vocab=None):
        """
        Dataset for Flickr8k
        Args:
            root_dir: Directory with images
            captions_file: Path to captions.txt
            transform: Image transformations
            freq_threshold: Minimum word frequency for vocabulary
            vocab: Pre-built vocabulary (for validation/test sets)
        """
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        
        # Build or use existing vocabulary
        if vocab is None:
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocabulary(self.df["caption"].tolist())
        else:
            self.vocab = vocab
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Get image and caption pair"""
        caption = self.df.iloc[idx]["caption"]
        img_name = self.df.iloc[idx]["image"]
        img_path = os.path.join(self.root_dir, img_name)
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        # Convert caption to numerical representation
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return img, torch.tensor(numericalized_caption)


# ============ STEP 3: Collate Function for Batching ============

class PadCollate:
    """Pad captions to same length in a batch"""
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        
        targets = [item[1] for item in batch]
        targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        
        return imgs, targets


# ============ STEP 4: Encoder (CNN - ResNet50) ============

class EncoderCNN(nn.Module):
    """CNN Encoder using pretrained ResNet50"""
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=True)
        
        # Remove last FC layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Freeze ResNet parameters (we'll use it as feature extractor)
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Linear layer to project features to embedding size
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images):
        """Extract features from images"""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        features = self.dropout(features)
        return features


# ============ STEP 5: Decoder (LSTM) ============

class DecoderLSTM(nn.Module):
    """LSTM Decoder for caption generation"""
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super(DecoderLSTM, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.5 if num_layers > 1 else 0)
        
        # Fully connected layer
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, features, captions):
        """
        Generate caption from image features
        Args:
            features: Image features from encoder
            captions: Ground truth captions (for teacher forcing)
        """
        # Embed captions
        embeddings = self.embed(captions)
        
        # Concatenate image features with caption embeddings
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        # LSTM forward pass
        hiddens, _ = self.lstm(embeddings)
        
        # Output layer
        outputs = self.linear(self.dropout(hiddens))
        
        return outputs


# ============ STEP 6: Training Function ============
def train_epoch(encoder, decoder, train_loader, criterion, optimizer, epoch, num_epochs):
    """Train for one epoch"""
    encoder.train()
    decoder.train()
    
    loop = tqdm(train_loader, leave=True)
    total_loss = 0
    
    for idx, (imgs, captions) in enumerate(loop):
        imgs = imgs.to(device)
        captions = captions.to(device)
        
        # Forward pass
        features = encoder(imgs)
        outputs = decoder(features, captions[:, :-1])
        
        # Calculate loss (ignore padding)
        loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions[:, 1:].reshape(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        loop.set_descr
def train_epoch(encoder, decoder, train_loader, criterion, optimizer, epoch, num_epochs):
    """Train for one epoch"""
    encoder.train()
    decoder.train()
    
    loop = tqdm(train_loader, leave=True)
    total_loss = 0
    
    for idx, (imgs, captions) in enumerate(loop):
        imgs = imgs.to(device)
        captions = captions.to(device)
        
        # Forward pass
        features = encoder(imgs)
        outputs = decoder(features, captions[:, :-1])
        
        # Calculate loss (ignore padding)
        # Skip the first output (which corresponds to image features)
        loss = criterion(outputs[:, 1:, :].reshape(-1, outputs.shape[2]), 
                        captions[:, 1:].reshape(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

# ============ STEP 7: Caption Generation ============

def generate_caption(encoder, decoder, image_path, vocabulary, transform, max_length=50):
    """Generate caption for a single image"""
    encoder.eval()
    decoder.eval()
    
    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    # Generate caption
    with torch.no_grad():
        features = encoder(image)
        caption = []
        
        # Initialize with <SOS> token
        inputs = torch.tensor([vocabulary.stoi["<SOS>"]]).unsqueeze(0).to(device)
        hidden = None
        
        for _ in range(max_length):
            # Embed input
            embeddings = decoder.embed(inputs)
            
            # Concatenate with features for first step
            if hidden is None:
                embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
            
            # LSTM step
            hiddens, hidden = decoder.lstm(embeddings, hidden) if hidden else decoder.lstm(embeddings)
            
            # Predict next word
            outputs = decoder.linear(hiddens[:, -1, :])
            predicted = outputs.argmax(1)
            
            # Get predicted word
            word_idx = predicted.item()
            
            # Stop if <EOS> token
            if word_idx == vocabulary.stoi["<EOS>"]:
                break
                
            caption.append(vocabulary.itos[word_idx])
            
            # Use predicted word as next input
            inputs = predicted.unsqueeze(0)
        
        return ' '.join(caption)


# ============ MAIN EXECUTION ============

if __name__ == "__main__":
    print("="*60)
    print("IMAGE CAPTION GENERATOR - TRAINING")
    print("="*60)
    
    # Hyperparameters
    embed_size = 256
    hidden_size = 512
    num_layers = 2
    learning_rate = 3e-4
    num_epochs = 10
    batch_size = 32
    freq_threshold = 5
    
    print(f"\nHyperparameters:")
    print(f"  Embedding size: {embed_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers: {num_layers}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num epochs: {num_epochs}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.3),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load dataset
    print("\n" + "="*60)
    print("Loading Flickr8k dataset...")
    print("="*60)
    
    dataset = FlickrDataset(
        root_dir="data/Images",
        captions_file="data/captions.txt",
        transform=transform,
        freq_threshold=freq_threshold
    )
    
    print(f"Total samples: {len(dataset)}")
    
    # Save vocabulary
    os.makedirs('models', exist_ok=True)
    with open('models/vocabulary.pkl', 'wb') as f:
        pickle.dump(dataset.vocab, f)
    print("Vocabulary saved to models/vocabulary.pkl")
    
    # Create dataloader
    pad_idx = dataset.vocab.stoi["<PAD>"]
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=PadCollate(pad_idx=pad_idx)
    )
    
    print(f"Number of batches: {len(train_loader)}")
    
    # Initialize models
    print("\n" + "="*60)
    print("Initializing models...")
    print("="*60)
    
    vocab_size = len(dataset.vocab)
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, num_layers).to(device)
    
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters() if p.requires_grad):,}")
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    params = list(encoder.linear.parameters()) + list(encoder.bn.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Train the model
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        avg_loss = train_epoch(encoder, decoder, train_loader, criterion, optimizer, epoch, num_epochs)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'vocab_size': vocab_size,
            'embed_size': embed_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers
        }
        
        torch.save(checkpoint, f'models/checkpoint_epoch_{epoch+1}.pth')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, 'models/best_model.pth')
            print(f"✓ Best model saved! Loss: {best_loss:.4f}")
        
        print("-"*60)
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print("Models saved in 'models/' directory")
    print("="*60)
