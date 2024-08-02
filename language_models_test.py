import pandas as pd
import torch
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# Prerequisits:
data = pd.read_csv('data/crypto_news/cryptonews.csv')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
text_input = data['text'].head(10).tolist()
print(text_input)

#region ElKulako/cryptobert model:
#print('\n')
print('\nElKulako/cryptobert model:')
model_name = "ElKulako/cryptobert"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 3)
model.to(device)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=64, truncation=True, padding = 'max_length', device = device)

preds = pipe(text_input)
print(preds)
#endregion

#region kk08/CryptoBERT model:
#print('\n')
print('\nkk08/CryptoBERT model:')
tokenizer = BertTokenizer.from_pretrained("kk08/CryptoBERT")
model = BertForSequenceClassification.from_pretrained("kk08/CryptoBERT")
model.to(device)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
result = classifier(text_input)
print(result)
#endregion

#region ProsusAI/FinBERT
#print('\n')
print('\nProsusAI/FinBERT model:')
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.to(device)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
result = classifier(text_input)
print(result)
#endregion

#region FacebookAI/roberta-base
#print('\n')
print('\nFacebookAI/roberta-base model:')
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModelForSequenceClassification.from_pretrained('roberta-base')
model.to(device)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
result = classifier(text_input)
print(result)
#endregion