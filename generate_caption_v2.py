import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pickle
import sys
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import classes from training script
from train_v2_attention import BahdanauAttention, AttentionDecoder, Vocabulary

# Define FeatureExtractor here
class FeatureExtractor(nn.Module):
    """Extract spatial features from ResNet for attention"""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
            features = self.adaptive_pool(features)
        return features


def load_model(checkpoint_path='models_v2/best_model.pth'):
    """Load trained attention model"""
    print("Loading model...")
    
    # Load vocabulary
    with open('models_v2/vocabulary.pkl', 'rb') as f:
        vocabulary = pickle.load(f)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model parameters
    attention_dim = checkpoint['attention_dim']
    embed_dim = checkpoint['embed_dim']
    decoder_dim = checkpoint['decoder_dim']
    vocab_size = checkpoint['vocab_size']
    encoder_dim = checkpoint['encoder_dim']
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()
    
    # Initialize decoder
    decoder = AttentionDecoder(
        attention_dim=attention_dim,
        embed_dim=embed_dim,
        decoder_dim=decoder_dim,
        vocab_size=vocab_size,
        encoder_dim=encoder_dim
    ).to(device)
    
    # Load weights
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.eval()
    
    print(f"✓ Model loaded (Val Loss: {checkpoint['val_loss']:.4f})")
    
    return feature_extractor, decoder, vocabulary


def generate_caption(feature_extractor, decoder, image_path, vocabulary, max_length=50):
    """Generate caption for an image using beam search"""
    
    # Transform for images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        features = feature_extractor(image)  # (1, 2048, 7, 7)
        features = features.squeeze(0).permute(1, 2, 0)  # (7, 7, 2048)
        features = features.reshape(-1, 2048)  # (49, 2048)
        features = features.unsqueeze(0)  # (1, 49, 2048)
        
        # Initialize decoder hidden state
        h, c = decoder.init_hidden_state(features)
        
        # Start with <SOS> token
        word = torch.tensor([vocabulary.stoi["<SOS>"]]).to(device)
        caption = []
        
        for _ in range(max_length):
            # Get embedding
            embedding = decoder.embedding(word)  # (1, embed_dim)
            
            # Get attention-weighted context
            context_vector, alpha = decoder.attention(features, h)
            
            # Gate context
            gate = torch.sigmoid(decoder.f_beta(h))
            gated_context = gate * context_vector
            
            # Concatenate context + embedding
            lstm_input = torch.cat([gated_context, embedding], dim=1)
            
            # LSTM step
            h, c = decoder.lstm_cell(lstm_input, (h, c))
            
            # Predict next word
            scores = decoder.fc(h)
            predicted = scores.argmax(1)
            
            # Get word
            word_idx = predicted.item()
            
            # Stop if <EOS>
            if word_idx == vocabulary.stoi["<EOS>"]:
                break
            
            # Avoid <PAD>, <SOS>, <UNK>
            if word_idx in [vocabulary.stoi["<PAD>"], vocabulary.stoi["<SOS>"]]:
                word = predicted
                continue
                
            caption.append(vocabulary.itos[word_idx])
            
            # Use predicted word as next input
            word = predicted
        
        return ' '.join(caption)


def generate_with_beam_search(feature_extractor, decoder, image_path, vocabulary, beam_size=3, max_length=50):
    """Generate caption using beam search for better results"""
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Extract features
        features = feature_extractor(image)
        features = features.squeeze(0).permute(1, 2, 0)
        features = features.reshape(-1, 2048).unsqueeze(0)
        
        # Expand features for beam search
        features = features.expand(beam_size, -1, -1)
        
        # Initialize
        h, c = decoder.init_hidden_state(features)
        
        # Start sequences
        sequences = [[vocabulary.stoi["<SOS>"]]]
        scores = [0.0]
        
        for _ in range(max_length):
            all_candidates = []
            
            for i, seq in enumerate(sequences):
                if seq[-1] == vocabulary.stoi["<EOS>"]:
                    all_candidates.append((seq, scores[i]))
                    continue
                
                # Get last word
                word = torch.tensor([seq[-1]]).to(device)
                embedding = decoder.embedding(word)
                
                # Attention and LSTM
                context_vector, _ = decoder.attention(features[i:i+1], h[i:i+1])
                gate = torch.sigmoid(decoder.f_beta(h[i:i+1]))
                gated_context = gate * context_vector
                lstm_input = torch.cat([gated_context, embedding], dim=1)
                h_new, c_new = decoder.lstm_cell(lstm_input, (h[i:i+1], c[i:i+1]))
                
                # Get scores
                output = decoder.fc(h_new)
                log_probs = F.log_softmax(output, dim=1)
                top_log_probs, top_indices = log_probs.topk(beam_size)
                
                for j in range(beam_size):
                    candidate_seq = seq + [top_indices[0][j].item()]
                    candidate_score = scores[i] + top_log_probs[0][j].item()
                    all_candidates.append((candidate_seq, candidate_score))
            
            # Sort and keep top beam_size
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = [seq for seq, score in ordered[:beam_size]]
            scores = [score for seq, score in ordered[:beam_size]]
            
            # Check if all sequences ended
            if all(seq[-1] == vocabulary.stoi["<EOS>"] for seq in sequences):
                break
        
        # Get best sequence
        best_sequence = sequences[0]
        caption = []
        for idx in best_sequence[1:]:  # Skip <SOS>
            if idx == vocabulary.stoi["<EOS>"]:
                break
            if idx not in [vocabulary.stoi["<PAD>"], vocabulary.stoi["<SOS>"]]:
                caption.append(vocabulary.itos[idx])
        
        return ' '.join(caption)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_caption_v2.py <image_path> [--beam]")
        print("Example: python generate_caption_v2.py data/Images/1000268201_693b08cb0e.jpg")
        print("         python generate_caption_v2.py test.jpg --beam  # Use beam search")
        sys.exit(1)
    
    image_path = sys.argv[1]
    use_beam = '--beam' in sys.argv
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    # Load model
    feature_extractor, decoder, vocabulary = load_model()
    
    print(f"\nGenerating caption for: {image_path}")
    print(f"Method: {'Beam Search (beam_size=3)' if use_beam else 'Greedy Search'}")
    print("-" * 60)
    
    # Generate caption
    if use_beam:
        caption = generate_with_beam_search(feature_extractor, decoder, image_path, vocabulary)
    else:
        caption = generate_caption(feature_extractor, decoder, image_path, vocabulary)
    
    print("\n" + "=" * 60)
    print("GENERATED CAPTION:")
    print("=" * 60)
    print(caption)
    print("=" * 60)

