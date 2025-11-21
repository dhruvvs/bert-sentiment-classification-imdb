# BERT Sentiment Classification on IMDB

This project fine-tunes a **BERT** model (`bert-base-uncased`) using the **Hugging Face Transformers** ecosystem for **binary sentiment classification** on the [IMDB movie review dataset](https://huggingface.co/datasets/imdb).

It shows a complete, reproducible workflow:
- Loading and splitting datasets with `datasets`
- Tokenization and dynamic padding with `AutoTokenizer` + `DataCollatorWithPadding`
- Fine-tuning BERT for sequence classification with `Trainer`
- Evaluating with accuracy, precision, recall, and F1-score

---

## Project Highlights

- **Model**: `bert-base-uncased` fine-tuned for sentiment analysis  
- **Dataset**: IMDB reviews (train/validation/test)  
- **Metrics**:  
  - Accuracy  
  - Precision (weighted)  
  - Recall (weighted)  
  - F1-score (weighted)  
- **Training loop**: Managed entirely via `transformers.Trainer`  
- **Early stopping**: Stops when validation F1 no longer improves  
- **GPU support**: Automatic mixed precision (`fp16`) when CUDA is available  

---

## Tech Stack

- Python
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face Datasets](https://github.com/huggingface/datasets)
- scikit-learn
- accelerate / evaluate (for HF ecosystem compatibility)

---


