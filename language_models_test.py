import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset, DatasetDict, load_metric

# Enable CUDA launch blocking for detailed error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Prerequisits:
print('Loading data ...')
data = pd.read_csv('data/crypto_news/cryptonews_dataset.csv')

# Map sentiment labels to numerical values
print('Mapping labels...')
label_mapping = {'Neutral': 0, 'Negative': 1, 'Positive': 2}
data['label'] = data['sentiment'].map(label_mapping)

print('Splitting data ...')
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['text'].tolist(), 
    data['label'].tolist(),
    test_size = 0.2,
    random_state=42
)

train_data = {'text': train_texts, 'label': train_labels}
test_data = {'text': test_texts, 'label': test_labels}

train_dataset = Dataset.from_dict(train_data)
test_dataset = Dataset.from_dict(test_data)
datasets = DatasetDict({"train": train_dataset, "test": test_dataset})

#region Training model 
def train_model(model_name):
    try:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 3, ignore_mismatched_sizes=True)
        model.to(device)

        tokenized_datasets = datasets.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=128), batched=True)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        training_args = TrainingArguments(
            output_dir = f'./results_{model_name}',
            num_train_epochs= 3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            warmup_ratio=0.01,
            logging_dir=f'./logs_{model_name}',
            logging_steps=10,
            eval_strategy='epoch'
        )

        trainer = Trainer(model=model,
                        args=training_args,
                        train_dataset=tokenized_datasets['train'],
                        eval_dataset=tokenized_datasets['test'],
                        data_collator=data_collator)
        
        trainer.train()
        return model, trainer
    
    except Exception as e:
        # Handle exceptions
        print(f"An error occurred while training model {model_name}: {e}")
        return None, None
#endregion

#Training each model 
# models = ['ElKulako/cryptobert', 
# 'kk08/CryptoBERT', 
# 'ProsusAI/finbert', 
# 'FacebookAI/roberta-base', 
# 'microsoft/deberta-v3-base', 
# 'google-bert/bert-base-uncased',
# 'distilbert/distilbert-base-cased']
models = ['ElKulako/cryptobert']

trained_models = {}
for model_name in models:
    print(f'Training model: {model_name}')
    model, trainer = train_model(model_name)
    if model is not None:
        trained_models[model_name] = (model, trainer)
    else:
        print(f"Skipping model {model_name} due to training error.")

#region Evaluation models:
from sklearn.metrics import accuracy_score

def evaluate_model(model, tokenizer, test_texts, test_labels):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Tokenize the test dataset
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt')

    # Create a test DataLoader
    test_dataset = torch.utils.data.TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_labels))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    model.eval()
    model.to(device)
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())

    accuracy = accuracy_score(test_labels, predictions)
    return accuracy


print(f"Evaluating models start...")
# Evaluate each trained model
for model_name, (model, trainer) in trained_models.items():
    print(f'Evaluating {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    accuracy = evaluate_model(model, tokenizer, test_texts, test_labels)
    print(f"Accuracy of {model_name}: {accuracy}")
 #endregion   