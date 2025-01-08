import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from torch.utils.data import DataLoader
import os

test_dataset_path = 'archive/test.csv'
df_test = pd.read_csv(test_dataset_path)

df_test = df_test[['Text', 'Sentiment']]
df_test = df_test.dropna()


label_mapping = {
    'Positive': 2,
    'Neutral': 1,
    'Negative': 0
}
df_test = df_test[df_test['Sentiment'].isin(label_mapping.keys())]


if len(df_test) == 0:
    raise ValueError("Test dataset is empty after filtering. Please check your data preparation steps.")


test_labels = df_test['Sentiment'].map(label_mapping).astype(int)


model_base_path = 'results-1/checkpoint-66528'

# Use the correct path for tokenizer and model
#if not os.path.exists(os.path.join(model_base_path, 'config.json')):
#    model_base_path = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
tokenizer_path = 'model-1'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_base_path)

# Tokenize the test dataset
test_encodings = tokenizer(list(df_test['Text']), truncation=True, padding=True, max_length=512)

# Convert the test dataset to Hugging Face Dataset format
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Set labels as dtype long
        return item

    def __len__(self):
        return len(self.labels)

test_dataset = SentimentDataset(test_encodings, list(test_labels))


if len(test_dataset) == 0:
    raise ValueError("Test dataset is empty after processing. Please check your data preparation steps.")


model.eval()

def evaluate_test_model(batch_size=64):
    print("Evaluating model on the test set...")
    device = 'mps' if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Use DataLoader for batch processing
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize lists for predictions and true labels
    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing", unit="batch"):
            # Move batch to the correct device
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            batch_labels = batch['labels'].to(device)

            # Get predictions for the batch
            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=-1)

            # Append the predictions and true labels
            preds.extend(batch_preds.tolist())
            labels.extend(batch_labels.tolist())

    
    report = classification_report(labels, preds, labels=[0, 1, 2], target_names=['negative', 'neutral', 'positive'])
    cm = confusion_matrix(labels, preds)

    # Write the evaluation results to a text file
    with open('test_evaluation_results_2.txt', 'w') as f:
        f.write("Test Set Evaluation Results:\n")
        f.write(report + "\n")
        f.write(f'Confusion matrix:\n{cm}\n')

    print("Evaluation results saved.")

if __name__ == "__main__":
    evaluate_test_model(batch_size=64)