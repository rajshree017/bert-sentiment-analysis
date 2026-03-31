# 🤖 BERT Sentiment Analysis — DistilBERT Fine-tuning

Fine-tune DistilBERT on IMDB reviews to classify sentiment. Achieves ~93% accuracy.

## 📁 Folder Structure
```
3_bert_sentiment/
├── train.py          # Fine-tuning + evaluation + inference
├── requirements.txt
└── README.md
```

## 🚀 Setup & Run
```bash
pip install -r requirements.txt

# Downloads IMDB dataset automatically via HuggingFace
python train.py
```

## 📊 Expected Results
| Metric | Value |
|--------|-------|
| Accuracy | ~93% |
| F1-score | ~0.93 |

## 🧠 What You'll Learn
- HuggingFace Transformers fine-tuning workflow
- Tokenization with padding/truncation
- Linear LR warmup scheduling
- Classification report interpretation
