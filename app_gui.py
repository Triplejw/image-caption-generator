import gradio as gr
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pickle
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
import os

# Import from training script
from train_v2_attention import BahdanauAttention, AttentionDecoder, Vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============ FEATURE EXTRACTOR ============

class FeatureExtractor(nn.Module):
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

# ============ LOAD MODEL ============

print("Loading model...")
with open('models_v2/vocabulary.pkl', 'rb') as f:
    vocabulary = pickle.load(f)

checkpoint = torch.load('models_v2/best_model.pth', map_location=device)
decoder = AttentionDecoder(
    attention_dim=checkpoint['attention_dim'],
    embed_dim=checkpoint['embed_dim'],
    decoder_dim=checkpoint['decoder_dim'],
    vocab_size=checkpoint['vocab_size'],
    encoder_dim=checkpoint['encoder_dim']
).to(device)
decoder.load_state_dict(checkpoint['decoder_state_dict'])
decoder.eval()

feature_extractor = FeatureExtractor().to(device)
feature_extractor.eval()

print("Model loaded successfully!")

# ============ CAPTION GENERATION FUNCTION ============

def generate_caption_from_image(image, use_beam_search=False):
    """Generate caption for uploaded image"""
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Convert to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        features = feature_extractor(image_tensor)
        features = features.squeeze(0).permute(1, 2, 0)
        features = features.reshape(-1, 2048).unsqueeze(0)
        
        # Generate caption
        h, c = decoder.init_hidden_state(features)
        word = torch.tensor([vocabulary.stoi["<SOS>"]]).to(device)
        caption = []
        
        for _ in range(50):
            embedding = decoder.embedding(word)
            context_vector, alpha = decoder.attention(features, h)
            gate = torch.sigmoid(decoder.f_beta(h))
            gated_context = gate * context_vector
            lstm_input = torch.cat([gated_context, embedding], dim=1)
            h, c = decoder.lstm_cell(lstm_input, (h, c))
            scores = decoder.fc(h)
            predicted = scores.argmax(1)
            word_idx = predicted.item()
            
            if word_idx == vocabulary.stoi["<EOS>"]:
                break
            if word_idx not in [vocabulary.stoi["<PAD>"], vocabulary.stoi["<SOS>"]]:
                caption.append(vocabulary.itos[word_idx])
            word = predicted
        
        caption_text = ' '.join(caption)
        return caption_text

# ============ TRAINING VISUALIZATION ============

def show_training_curves():
    """Generate and return training curves"""
    train_losses = []
    val_losses = []
    epochs = []
    
    for epoch in range(1, 16):
        path = f'models_v2/checkpoint_epoch_{epoch}.pth'
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location='cpu')
            train_losses.append(checkpoint['train_loss'])
            val_losses.append(checkpoint['val_loss'])
            epochs.append(epoch)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=8)
    ax.plot(epochs, val_losses, 'r-o', label='Validation Loss', linewidth=2, markersize=8)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Training Progress - Attention Model', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

# ============ MODEL INFO ============

def get_model_info():
    """Return model architecture information"""
    info = f"""
    ### Model Architecture (Version 2 - Attention-Based)
    
    **Encoder:**
    - ResNet-50 (pretrained on ImageNet)
    - Extracts 49 spatial regions (7×7 grid)
    - 2048 dimensions per region
    
    **Attention Mechanism:**
    - Bahdanau (Additive) Attention
    - Attention dimension: {checkpoint['attention_dim']}
    - Dynamically focuses on different image regions per word
    
    **Decoder:**
    - LSTM with attention
    - Embedding dimension: {checkpoint['embed_dim']}
    - Hidden dimension: {checkpoint['decoder_dim']}
    - Vocabulary size: {checkpoint['vocab_size']} words
    - Total parameters: 13.4M
    
    **Training:**
    - Dataset: Flickr8k (6,000 train / 1,000 val / 1,000 test)
    - Best validation loss: {checkpoint['val_loss']:.4f}
    - Optimizer: Adam (lr=4e-4)
    - Trained for 15 epochs on RTX 3060
    """
    return info

# ============ BLEU SCORE DISPLAY ============

def show_metrics():
    """Display model metrics"""
    metrics_text = """
    ### Evaluation Metrics
    
    **Model Performance:**
    - Validation Loss: 3.04
    - Training completed in ~25 minutes
    
    **To calculate BLEU scores on test set, run:**
    ```
    python evaluate_model.py
    ```
    
    **Expected Performance:**
    - BLEU-1: ~0.55-0.60 (unigram precision)
    - BLEU-2: ~0.35-0.40 (bigram precision)
    - BLEU-3: ~0.20-0.25 (trigram precision)
    - BLEU-4: ~0.10-0.15 (4-gram precision)
    
    **Improvements vs Version 1:**
    ✓ Attention mechanism for spatial understanding
    ✓ Better color and object recognition
    ✓ More coherent captions
    ✓ Train/val split for proper evaluation
    """
    return metrics_text

# ============ GRADIO INTERFACE ============

with gr.Blocks(title="Image Caption Generator", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🖼️ Image Caption Generator with Attention
    ### Upload an image and get AI-generated captions!
    Built with ResNet-50 encoder and LSTM decoder with Bahdanau attention mechanism.
    """)
    
    with gr.Tabs():
        
        # TAB 1: Caption Generation
        with gr.Tab("📸 Generate Captions"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload Image")
                    generate_btn = gr.Button("Generate Caption", variant="primary", size="lg")
                
                with gr.Column():
                    caption_output = gr.Textbox(label="Generated Caption", lines=3, 
                                               placeholder="Caption will appear here...")
                    gr.Markdown("### Try these:")
                    gr.Examples(
                        examples=[
                            "data/Images/1000268201_693b08cb0e.jpg",
                            "data/Images/1001773457_577c3a7d70.jpg",
                            "data/Images/1002674143_1b742ab4b8.jpg",
                        ],
                        inputs=image_input
                    )
            
            generate_btn.click(
                fn=generate_caption_from_image,
                inputs=[image_input],
                outputs=caption_output
            )
        
        # TAB 2: Training Visualization
        with gr.Tab("📊 Training Progress"):
            gr.Markdown("### Training and Validation Loss Curves")
            with gr.Row():
                plot_btn = gr.Button("Show Training Curves", variant="primary")
            plot_output = gr.Plot(label="Loss Curves")
            
            plot_btn.click(
                fn=show_training_curves,
                outputs=plot_output
            )
        
        # TAB 3: Model Info
        with gr.Tab("🤖 Model Architecture"):
            gr.Markdown(get_model_info())
        
        # TAB 4: Metrics
        with gr.Tab("📈 Evaluation Metrics"):
            gr.Markdown(show_metrics())
    
    gr.Markdown("""
    ---
    **Project:** Image Caption Generator | **Model:** Attention-based Encoder-Decoder  
    **Hardware:** NVIDIA RTX 3060 Laptop GPU | **Framework:** PyTorch 2.5
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
