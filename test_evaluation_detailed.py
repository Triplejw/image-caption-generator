import torch
import pickle
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import numpy as np
import random

from train_v2_attention import AttentionDecoder, Vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_test_set_detailed():
    """Comprehensive evaluation on test set"""
    
    print("="*70)
    print("DETAILED TEST SET EVALUATION - UNSEEN DATA")
    print("="*70)
    
    # Load model
    print("\nLoading model and test data...")
    with open('models_v2/vocabulary.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    with open('data/features/resnet_features.pkl', 'rb') as f:
        features_dict = pickle.load(f)
    
    test_df = pd.read_csv('data/splits/test_captions.txt')
    
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
    
    # Get unique test images
    test_images = test_df['image'].unique()
    print(f"✓ Test set: {len(test_images)} images (never seen during training)")
    print(f"✓ Total test captions: {len(test_df)} (5 per image)")
    
    # Evaluate
    image_scores = []
    all_predictions = []
    all_references = []
    
    print("\nEvaluating on test images...")
    for img_name in tqdm(test_images):
        # Get all reference captions for this image
        ref_captions = test_df[test_df['image'] == img_name]['caption'].tolist()
        
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
        
        # Calculate BLEU for this image
        references = [ref.lower().split() for ref in ref_captions]
        bleu4 = sentence_bleu(references, caption, weights=(0.25, 0.25, 0.25, 0.25))
        
        image_scores.append({
            'image': img_name,
            'prediction': ' '.join(caption),
            'references': ref_captions,
            'bleu4': bleu4
        })
        
        all_predictions.append(caption)
        all_references.append(references)
    
    # Calculate statistics
    bleu_scores = [item['bleu4'] for item in image_scores]
    mean_bleu = np.mean(bleu_scores)
    std_bleu = np.std(bleu_scores)
    median_bleu = np.median(bleu_scores)
    
    # Print results
    print("\n" + "="*70)
    print("TEST SET PERFORMANCE SUMMARY")
    print("="*70)
    print(f"\nTotal test images evaluated: {len(test_images)}")
    print(f"Model never saw these images during training!")
    print(f"\nBLEU-4 Statistics on Test Set:")
    print(f"  Mean:   {mean_bleu:.4f}")
    print(f"  Median: {median_bleu:.4f}")
    print(f"  Std:    {std_bleu:.4f}")
    print(f"  Min:    {min(bleu_scores):.4f}")
    print(f"  Max:    {max(bleu_scores):.4f}")
    
    # Performance categories
    excellent = sum(1 for s in bleu_scores if s >= 0.3)
    good = sum(1 for s in bleu_scores if 0.2 <= s < 0.3)
    fair = sum(1 for s in bleu_scores if 0.1 <= s < 0.2)
    poor = sum(1 for s in bleu_scores if s < 0.1)
    
    print(f"\nPerformance Distribution:")
    print(f"  Excellent (BLEU ≥ 0.30): {excellent} images ({excellent/len(test_images)*100:.1f}%)")
    print(f"  Good (0.20 ≤ BLEU < 0.30): {good} images ({good/len(test_images)*100:.1f}%)")
    print(f"  Fair (0.10 ≤ BLEU < 0.20): {fair} images ({fair/len(test_images)*100:.1f}%)")
    print(f"  Poor (BLEU < 0.10): {poor} images ({poor/len(test_images)*100:.1f}%)")
    
    # Show best and worst predictions
    sorted_results = sorted(image_scores, key=lambda x: x['bleu4'], reverse=True)
    
    print("\n" + "="*70)
    print("BEST PREDICTIONS (Top 5)")
    print("="*70)
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"\n{i}. Image: {result['image']}")
        print(f"   BLEU-4: {result['bleu4']:.4f}")
        print(f"   Generated: {result['prediction']}")
        print(f"   Reference: {result['references'][0]}")
    
    print("\n" + "="*70)
    print("WORST PREDICTIONS (Bottom 5)")
    print("="*70)
    for i, result in enumerate(sorted_results[-5:], 1):
        print(f"\n{i}. Image: {result['image']}")
        print(f"   BLEU-4: {result['bleu4']:.4f}")
        print(f"   Generated: {result['prediction']}")
        print(f"   Reference: {result['references'][0]}")
    
    print("\n" + "="*70)
    print("RANDOM SAMPLE PREDICTIONS (5 random)")
    print("="*70)
    random_samples = random.sample(image_scores, 5)
    for i, result in enumerate(random_samples, 1):
        print(f"\n{i}. Image: {result['image']}")
        print(f"   BLEU-4: {result['bleu4']:.4f}")
        print(f"   Generated: {result['prediction']}")
        print(f"   Reference: {result['references'][0]}")
    
    # Save detailed results
    results_df = pd.DataFrame([{
        'image': r['image'],
        'prediction': r['prediction'],
        'reference_1': r['references'][0],
        'bleu4': r['bleu4']
    } for r in image_scores])
    results_df.to_csv('test_results_detailed.csv', index=False)
    
    print("\n" + "="*70)
    print(f"✓ Detailed results saved to: test_results_detailed.csv")
    print("="*70)
    
    return image_scores

if __name__ == "__main__":
    evaluate_test_set_detailed()
