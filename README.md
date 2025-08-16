# BERT Text Classifier (Arabic Sentiment Analysis)

This project implements a BERT-based text classification pipeline for Arabic text (e.g., sentiment detection).  
It is organized into two main folders:

- **`classify/`** → All Python modules (data loading, model definition, training loop, utilities).
- **`data/`** → Dataset files (`train.csv`, `val.csv`, `test.csv`).

## 📂 Project Structure
bert-text-classifier/
│
├── classify/
│ ├── init.py
│ ├── data.py
│ ├── model.py
│ ├── trainer.py
│ ├── train.py
│ └── utils.py
│
├── data/
│ ├── train.csv
│ ├── val.csv
│ └── test.csv
│
├── requirements.txt
└── README.md


## 🚀 Features
- Uses Hugging Face Transformers for BERT and tokenization.
- Training, validation, and testing pipeline with clear separation.
- Easily tunable hyperparameters (epochs, batch size, learning rate, model).
- Compatible with Colab and GPU acceleration.

## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/USERNAME/bert-text-classifier.git
cd bert-text-classifier
pip install -r requirements.txt

📊 Dataset Format

The CSV files (train.csv, val.csv, test.csv) must contain two columns:

text → input sentence

label → class label (string or numeric)

Example:

text,label
"هذا المنتج رائع",Positive
"لسانك قذر يا قمامه",Negative

▶️ Training

Run the training script:

python classify/train.py \
  --train_path data/train.csv \
  --val_path data/val.csv \
  --test_path data/test.csv \
  --output_path output/ \
  --bert_model bert-base-multilingual-cased \
  --batch_size 16 \
  --max_epochs 5 \
  --learning_rate 2e-5

🔍 To-Do

 Add Optuna hyperparameter search

 Add confusion matrix & metrics plotting

 Extend dataset support beyond sentiment

