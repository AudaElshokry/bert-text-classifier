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


Outputs:

test_metrics.json

preds_test.csv

classification_report.txt

confusion_matrix.png

best_model.pt


## What changed vs. before (high-impact fixes)
- ✅ Correct import paths (`classify.*`, not `comp9312.*`).
- ✅ Proper `attention_mask` dtype (long/bool).
- ✅ Gradient clipping applied.
- ✅ Early stopping on `val f1_macro` + best checkpoint restore.
- ✅ Clean metrics output + confusion matrix + classification report.
- ✅ Safe `CUDA_VISIBLE_DEVICES` handling (uses provided IDs).
- ✅ Optional `--fp16` mixed precision for speed.