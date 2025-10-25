import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============ VOCABULARY ============

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
        
    def __len__(self):
        return len(self.itos)
    
    def build_vocabulary(self, sentence_list):
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
        tokenized_text = word_tokenize(text.lower())
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]


# ============ DATASET WITH PRE-EXTRACTED FEATURES ============

class FlickrDatasetPreExtracted(Dataset):
    def __init__(self, captions_file, features_dict, vocab=None, freq_threshold=5):
        self.df = pd.read_csv(captions_file)
        self.features_dict = features_dict
        
        if vocab is None:
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocabulary(self.df["caption"].tolist())
        else:
            self.vocab = vocab
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        caption = self.df.iloc[idx]["caption"]
        img_name = self.df.iloc[idx]["image"]
        
        # Load pre-extracted features
        features = torch.FloatTensor(self.features_dict[img_name])
        
        # Numericalize caption
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return features, torch.tensor(numericalized_caption)


class PadCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        features = [item[0] for item in batch]
        features = torch.stack(features, dim=0)
        
        targets = [item[1] for item in batch]
        targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, 
                                           padding_value=self.pad_idx)
        
        return features, targets


# ============ BAHDANAU ATTENTION MECHANISM ============

class BahdanauAttention(nn.Module):
    """Bahdanau (Additive) Attention"""
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        
    def forward(self, encoder_out, decoder_hidden):
        """
        encoder_out: (batch_size, num_pixels=49, encoder_dim=2048)
        decoder_hidden: (batch_size, decoder_dim)
        
        Returns:
        attention_weights: (batch_size, num_pixels)
        context_vector: (batch_size, encoder_dim)
        """
        # Transform encoder outputs
        att1 = self.encoder_att(encoder_out)  # (batch, 49, attention_dim)
        
        # Transform decoder hidden state
        att2 = self.decoder_att(decoder_hidden)  # (batch, attention_dim)
        
        # Add and apply tanh
        att = torch.tanh(att1 + att2.unsqueeze(1))  # (batch, 49, attention_dim)
        
        # Calculate attention scores
        att = self.full_att(att).squeeze(2)  # (batch, 49)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(att, dim=1)  # (batch, 49)
        
        # Calculate context vector
        context_vector = (encoder_out * attention_weights.unsqueeze(2)).sum(dim=1)
        # (batch, encoder_dim)
        
        return context_vector, attention_weights


# ============ ATTENTION-BASED LSTM DECODER ============

class AttentionDecoder(nn.Module):
    """LSTM Decoder with Bahdanau Attention"""
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, 
                 encoder_dim=2048, dropout=0.5):
        super(AttentionDecoder, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        
        # Attention mechanism
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # LSTM cell (takes concatenated context + embedding as input)
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        
        # Linear layers
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # Gate for attention
        self.fc = nn.Linear(decoder_dim, vocab_size)
        
    def init_hidden_state(self, encoder_out):
        """Initialize LSTM hidden and cell states"""
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c
    
    def forward(self, encoder_out, captions):
        """
        encoder_out: (batch_size, 49, 2048)
        captions: (batch_size, max_caption_length)
        """
        batch_size = encoder_out.size(0)
        vocab_size = self.vocab_size
        
        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)
        
        # Embedding captions (includes <SOS> but not <EOS>)
        embeddings = self.embedding(captions)  # (batch, caption_len, embed_dim)
        
        # Get the actual caption length
        caption_len = captions.size(1)
        
        # Tensors to hold outputs
        predictions = torch.zeros(batch_size, caption_len, vocab_size).to(device)
        alphas = torch.zeros(batch_size, caption_len, encoder_out.size(1)).to(device)
        
        # Decode step by step
        for t in range(caption_len):
            # Get attention-weighted encoding
            context_vector, alpha = self.attention(encoder_out, h)
            
            # Gate the context vector
            gate = torch.sigmoid(self.f_beta(h))
            gated_context = gate * context_vector
            
            # Get embedding for current time step
            embedding_t = embeddings[:, t, :]
            
            # Concatenate context + embedding
            lstm_input = torch.cat([gated_context, embedding_t], dim=1)
            
            # LSTM step
            h, c = self.lstm_cell(lstm_input, (h, c))
            
            # Predict next word
            preds = self.fc(self.dropout(h))
            
            predictions[:, t, :] = preds
            alphas[:, t, :] = alpha
        
        return predictions, alphas


# ============ TRAINING FUNCTION ============

def train_epoch(decoder, train_loader, criterion, optimizer, epoch, num_epochs):
    decoder.train()
    loop = tqdm(train_loader, leave=True)
    total_loss = 0
    
    for idx, (features, captions) in enumerate(loop):
        features = features.to(device)
        captions = captions.to(device)
        
        # Forward pass
        predictions, alphas = decoder(features, captions[:, :-1])
        
        # Calculate loss - need to handle variable lengths properly
        targets = captions[:, 1:]
        
        # Create mask for non-padded tokens
        mask = (targets != 0).float()  # 0 is PAD token
        
        # Flatten predictions and targets
        predictions = predictions.reshape(-1, predictions.shape[-1])
        targets = targets.reshape(-1)
        mask = mask.reshape(-1)
        
        # Calculate loss only on non-padded tokens
        loss = criterion(predictions, targets)
        
        # Manually apply mask
        loss = (loss.view(-1) * mask).sum() / mask.sum()
        
        # Add doubly stochastic attention regularization
        # alphas shape: (batch, seq_len, 49)
        batch_size = alphas.size(0)
        seq_len = alphas.size(1)
        
        # Reshape mask to match alphas
        alpha_mask = mask.view(batch_size, seq_len)
        
        # Sum alphas over sequence dimension: (batch, 49)
        alpha_sum = (alphas * alpha_mask.unsqueeze(2)).sum(dim=1)
        
        # Penalize if sum != 1 for each pixel
        alpha_loss = ((1. - alpha_sum) ** 2).mean()
        
        total_loss_val = loss + alpha_loss
        
        # Backward
        optimizer.zero_grad()
        total_loss_val.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += total_loss_val.item()
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(loss=total_loss_val.item())
    
    return total_loss / len(train_loader)


# ============ VALIDATION FUNCTION ============

def validate(decoder, val_loader, criterion):
    decoder.eval()
    total_loss = 0
    
    with torch.no_grad():
        for features, captions in val_loader:
            features = features.to(device)
            captions = captions.to(device)
            
            predictions, alphas = decoder(features, captions[:, :-1])
            
            targets = captions[:, 1:]
            
            # Create mask for non-padded tokens
            mask = (targets != 0).float()
            
            predictions = predictions.reshape(-1, predictions.shape[-1])
            targets = targets.reshape(-1)
            mask = mask.reshape(-1)
            
            loss = criterion(predictions, targets)
            loss = (loss.view(-1) * mask).sum() / mask.sum()
            
            # Alpha regularization
            batch_size = alphas.size(0)
            seq_len = alphas.size(1)
            
            alpha_mask = mask.view(batch_size, seq_len)
            alpha_sum = (alphas * alpha_mask.unsqueeze(2)).sum(dim=1)
            alpha_loss = ((1. - alpha_sum) ** 2).mean()
            
            total_loss += (loss + alpha_loss).item()
    
    return total_loss / len(val_loader)


# ============ MAIN TRAINING ============

if __name__ == "__main__":
    print("="*60)
    print("IMAGE CAPTION GENERATOR V2 - WITH ATTENTION")
    print("="*60)
    
    # Hyperparameters
    attention_dim = 512
    embed_dim = 512
    decoder_dim = 512
    encoder_dim = 2048
    learning_rate = 4e-4
    num_epochs = 15
    batch_size = 32
    
    print(f"\nHyperparameters:")
    print(f"  Attention dim: {attention_dim}")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Decoder dim: {decoder_dim}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    
    # Load pre-extracted features
    print("\n" + "="*60)
    print("Loading pre-extracted features...")
    with open('data/features/resnet_features.pkl', 'rb') as f:
        features_dict = pickle.load(f)
    print(f"✓ Loaded features for {len(features_dict)} images")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = FlickrDatasetPreExtracted(
        'data/splits/train_captions.txt',
        features_dict,
        vocab=None
    )
    
    val_dataset = FlickrDatasetPreExtracted(
        'data/splits/val_captions.txt',
        features_dict,
        vocab=train_dataset.vocab
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Save vocabulary
    os.makedirs('models_v2', exist_ok=True)
    with open('models_v2/vocabulary.pkl', 'wb') as f:
        pickle.dump(train_dataset.vocab, f)
    
    # Create dataloaders
    pad_idx = train_dataset.vocab.stoi["<PAD>"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=PadCollate(pad_idx)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=PadCollate(pad_idx)
    )
    
    # Initialize model
    print("\n" + "="*60)
    print("Initializing attention-based decoder...")
    vocab_size = len(train_dataset.vocab)
    
    decoder = AttentionDecoder(
        attention_dim=attention_dim,
        embed_dim=embed_dim,
        decoder_dim=decoder_dim,
        vocab_size=vocab_size,
        encoder_dim=encoder_dim
    ).to(device)
    
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none')  # Manual masking
    optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(decoder, train_loader, criterion, optimizer, 
                                epoch, num_epochs)
        val_loss = validate(decoder, val_loader, criterion)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'vocab_size': vocab_size,
            'attention_dim': attention_dim,
            'embed_dim': embed_dim,
            'decoder_dim': decoder_dim,
            'encoder_dim': encoder_dim
        }
        
        torch.save(checkpoint, f'models_v2/checkpoint_epoch_{epoch+1}.pth')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, 'models_v2/best_model.pth')
            print(f"  ✓ Best model saved! Val Loss: {best_val_loss:.4f}")
        
        print("-"*60)
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)
