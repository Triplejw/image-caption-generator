import torch
import pickle
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from tqdm import tqdm
import nltk
nltk.download('punkt', quiet=True)

from train_v2_attention import AttentionDecoder, Vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_bleu(checkpoint_path='models_v2/best_model.pth'):
    """Calculate BLEU scores on test set"""
    
    print("Loading model and data...")
    
    # Load vocabulary
    with open('models_v2/vocabulary.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    # Load features
    with open('data/features/resnet_features.pkl', 'rb') as f:
        features_dict = pickle.load(f)
    
    # Load test captions
    test_df = pd.read_csv('data/splits/test_captions.txt')
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    decoder = AttentionDecoder(
        attention_dim=checkpoint['attention_dim'],
        embed_dim=checkpoint['embed_dim'],
        decoder_dim=checkpoint['decoder_dim'],
        vocab_size=checkpoint['vocab_size'],
        encoder_dim=checkpoint['encoder_dim']
    ).to(device)
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.eval()
    
    # Group captions by image
    image_captions = test_df.groupby('image')['caption'].apply(list).to_dict()
    
    references = []
    hypotheses = []
    
    print(f"\nEvaluating on {len(image_captions)} test images...")
    
    for img_name, ref_captions in tqdm(image_captions.items()):
        # Get features
        features = torch.FloatTensor(features_dict[img_name]).unsqueeze(0).to(device)
        
        # Generate caption
        with torch.no_grad():
            h, c = decoder.init_hidden_state(features)
            word = torch.tensor([vocab.stoi["<SOS>"]]).to(device)
            caption = []
            
            for _ in range(50):
                embedding = decoder.embedding(word)
                context_vector, _ = decoder.attention(features, h)
                gate = torch.sigmoid(decoder.f_beta(h))
                gated_context = gate * context_vector
                lstm_input = torch.cat([gated_context, embedding], dim=1)
                h, c = decoder.lstm_cell(lstm_input, (h, c))
                scores = decoder.fc(h)
                predicted = scores.argmax(1)
                word_idx = predicted.item()
                
                if word_idx == vocab.stoi["<EOS>"]:
                    break
                if word_idx not in [vocab.stoi["<PAD>"], vocab.stoi["<SOS>"]]:
                    caption.append(vocab.itos[word_idx])
                word = predicted
        
        # Tokenize references and hypothesis
        references.append([ref.lower().split() for ref in ref_captions])
        hypotheses.append(caption)
    
    # Calculate BLEU scores
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    results = {
        'BLEU-1': bleu1,
        'BLEU-2': bleu2,
        'BLEU-3': bleu3,
        'BLEU-4': bleu4
    }
    
    print("\n" + "="*60)
    print("BLEU SCORES ON TEST SET")
    print("="*60)
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
    print("="*60)
    
    return results

if __name__ == "__main__":
    evaluate_bleu()
