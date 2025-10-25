import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import sys
import os

# Import model classes
from train import EncoderCNN, DecoderLSTM
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import sys
import os

# Import model classes and Vocabulary from train.py
from train import EncoderCNN, DecoderLSTM, Vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(checkpoint_path='models/best_model.pth'):
    """Load trained model"""
    # Load vocabulary
    with open('models/vocabulary.pkl', 'rb') as f:
        vocabulary = pickle.load(f)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model parameters
    embed_size = checkpoint['embed_size']
    hidden_size = checkpoint['hidden_size']
    vocab_size = checkpoint['vocab_size']
    num_layers = checkpoint['num_layers']
    
    # Initialize models
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, num_layers).to(device)
    
    # Load weights
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder, vocabulary

def generate_caption(encoder, decoder, image_path, vocabulary, max_length=50):
    """Generate caption for an image"""
    # Transform for test images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_caption.py <image_path>")
        print("Example: python generate_caption.py data/Images/1000268201_693b08cb0e.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    print("Loading model...")
    encoder, decoder, vocabulary = load_model()
    
    print(f"Generating caption for: {image_path}")
    caption = generate_caption(encoder, decoder, image_path, vocabulary)
    
    print("\n" + "="*60)
    print("GENERATED CAPTION:")
    print("="*60)
    print(caption)
    print("="*60)
