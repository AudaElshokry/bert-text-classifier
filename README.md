# BERT Text Classifier (Arabic)

Train a BERT-based classifier on CSV files with columns: `text,label`.

## Structure
bert-text-classifier/
├─ classify/
│ ├─ data.py # Dataset + collate_fn (tokenizer)
│ ├─ model.py # build_model (HF)
│ ├─ trainer.py # training loop, eval, reports
│ ├─ utils.py # seeding, CSV helpers, label maps
│ └─ train.py # CLI entrypoint
├─ data/
│ ├─ train.csv
│ ├─ val.csv
│ └─ test.csv
├─ requirements.txt
└─ README.md

bash
Copy
Edit

## Quick Start (Colab)
```bash
!git clone https://github.com/<YOU>/bert-text-classifier.git /content/bert-text-classifier
!pip -q install -r /content/bert-text-classifier/requirements.txt

!python /content/bert-text-classifier/classify/train.py \
  --train_path /content/bert-text-classifier/data/train.csv \
  --val_path   /content/bert-text-classifier/data/val.csv \
  --test_path  /content/bert-text-classifier/data/test.csv \
  --output_path /content/output \
  --bert_model bert-base-multilingual-cased \
  --batch_size 16 --max_epochs 5 --learning_rate 2e-5
Outputs:

test_metrics.json

preds_test.csv

classification_report.txt

confusion_matrix.png

best_model.pt

pgsql
Copy
Edit

---

## What changed vs. before (high-impact fixes)
- ✅ Correct import paths (`classify.*`, not `comp9312.*`).
- ✅ Proper `attention_mask` dtype (long/bool).
- ✅ Gradient clipping applied.
- ✅ Early stopping on `val f1_macro` + best checkpoint restore.
- ✅ Clean metrics output + confusion matrix + classification report.
- ✅ Safe `CUDA_VISIBLE_DEVICES` handling (uses provided IDs).
- ✅ Optional `--fp16` mixed precision for speed.