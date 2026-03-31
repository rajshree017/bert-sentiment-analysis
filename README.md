# BERT Sentiment Analysis
Fine-tuned DistilBERT on IMDB reviews for sentiment classification.

## 📊 Analyses Performed
- Tokenization with padding & truncation
- Training loss per epoch
- Classification Report (Precision, Recall, F1)
- Custom text sentiment prediction

## 🤖 ML Model
- DistilBERT fine-tuned on IMDB dataset
- AdamW optimizer with linear warmup
- Accuracy: ~93%
- F1-Score: ~0.93

## 🛠️ Technologies Used
- Python
- PyTorch
- HuggingFace Transformers & Datasets
- Scikit-learn

## ▶️ How to Run
pip install torch transformers datasets scikit-learn
python train.py

## 📁 Output
Trained model saved in saved_model/ folder.

## 👩‍💻 Author
Rajshree - ML Engineer | Python Developer
