# 🖼️ Image Caption Generator with Attention Mechanism

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning project that automatically generates natural language descriptions for images using an encoder-decoder architecture with Bahdanau attention mechanism.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Acknowledgments](#acknowledgments)

## ✨ Features

- **Attention Mechanism**: Bahdanau (additive) attention for focusing on relevant image regions
- **Pre-trained Encoder**: ResNet-50 (ImageNet) for robust feature extraction
- **Spatial Features**: 7×7 grid (49 regions) for fine-grained attention
- **Interactive GUI**: Gradio-based web interface for easy caption generation
- **Comprehensive Evaluation**: BLEU-1/2/3/4 metrics with visual reports
- **Training Visualization**: Real-time loss curves and performance tracking

## 🏗️ Architecture

### Encoder
- **Model**: ResNet-50 (pretrained on ImageNet)
- **Output**: 49 spatial regions (7×7 grid)
- **Feature Dimension**: 2048 per region

### Attention Mechanism
- **Type**: Bahdanau (Additive) Attention
- **Attention Dim**: 512
- **Purpose**: Dynamic focus on relevant image regions per word

### Decoder
- **Model**: LSTM with attention
- **Embedding Dim**: 512
- **Hidden Dim**: 512
- **Vocabulary Size**: 2,590 words
- **Parameters**: 13.4M

## 📊 Results

### Performance on Flickr8k Test Set (1,000 unseen images)

| Metric | Score | Industry Baseline |
|--------|-------|-------------------|
| BLEU-1 | 0.647 | 0.50-0.60 |
| BLEU-2 | 0.443 | 0.30-0.40 |
| BLEU-3 | 0.306 | 0.18-0.25 |
| BLEU-4 | 0.208 | 0.10-0.15 |

**Our model outperforms typical baselines on all metrics!**

### Training Performance
- Training Loss: 2.22 (final)
- Validation Loss: 3.04 (best)
- Training Time: ~25 minutes (15 epochs on RTX 3060)
- GPU Utilization: 80-95%

## 🚀 Installation

### Prerequisites
- Python 3.12+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ disk space

### Setup Steps

1. Clone the repository
git clone https://github.com/Triplejw/image-caption-generator.git
cd image-caption-generator

3. Create virtual environment
python -m venv venv
source venv/bin/activate

4. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

5. Download Flickr8k dataset
mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d adityajn105/flickr8k
unzip flickr8k.zip -d data

## 🛠️ Technical Stack

- Deep Learning: PyTorch 2.5
- Computer Vision: ResNet-50
- NLP: NLTK, BLEU metrics
- GUI: Gradio
- Hardware: NVIDIA RTX 3060

## 🤝 Acknowledgments

- Dataset: Flickr8k from Kaggle
- Architecture: "Show, Attend and Tell" (Xu et al., 2015)
- Pre-trained Model: ResNet-50

## 📄 License

MIT License

## 👤 Author

**Joshua JJ Wonder**
- GitHub: @Triplejw
- Email: wonderjj2017@gmail.com

---

Built with ❤️ using PyTorch and Attention

