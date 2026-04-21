# X-AlignFact: Cross-Lingual Fact Verification via Curriculum-Driven Consistency Training

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YourUsername/X-AlignFact/blob/main/X_AlignFact_Colab_Pipeline.ipynb)

Official implementation of the paper: **"X-AlignFact: A Cross-Lingual Alignment Framework for Bilingual Fact Verification in English and Vietnamese"** (Submitted to IEEE Access).

## 📖 Abstract
X-AlignFact is a cross-lingual fact verification framework that jointly addresses English (EN) and Vietnamese (VI) claim verification. It employs a shared XLM-RoBERTa encoder, multi-objective alignment losses, and a novel three-phase curriculum training strategy. 

## 🚀 Quick Start (Google Colab)
We have designed the codebase to run seamlessly on Google Colab with zero local installation required. 

1. Click the **"Open in Colab"** badge above.
2. In the Colab menu, go to **Runtime > Change runtime type** and select **GPU (T4 or A100)**.
3. Run the cells step-by-step. The notebook automatically handles:
   * Installing dependencies (`!pip install transformers torch...`)
   * Downloading the FEVER and ViFactCheck datasets
   * Downloading our pre-trained X-AlignFact model weights
   * Running the Inference and Evaluation pipeline

## 📂 Using Your Own Google Drive (For Training)
If you wish to train the model from scratch (which takes ~15 epochs), you will need to mount your Google Drive to save the model checkpoints to prevent data loss.

```python
from google.colab import drive
drive.mount('/content/drive')

# Clone this repository into your Drive
!git clone [https://github.com/YourUsername/X-AlignFact.git](https://github.com/YourUsername/X-AlignFact.git) /content/drive/MyDrive/X-AlignFact
%cd /content/drive/MyDrive/X-AlignFact
