import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load dataset
dataset_path = 'archive/train.csv'
df = pd.read_csv(dataset_path)

df = df[['Text', 'Sentiment']]
df = df.dropna()

label_mapping = {
    'Positive': 2,
    'Neutral': 1,
    'Negative': 0
}
df = df[df['Sentiment'].isin(label_mapping.keys())]

if len(df) == 0:
    raise ValueError("Dataset is empty after filtering. Please check your data preparation steps.")

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(df['Text'], df['Sentiment'], test_size=0.2, random_state=42)

train_labels = train_labels.map(label_mapping).astype(int)
val_labels = val_labels.map(label_mapping).astype(int)

# Load the tokenizer and model
model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=512)

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

train_dataset = SentimentDataset(train_encodings, list(train_labels))
val_dataset = SentimentDataset(val_encodings, list(val_labels))

if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise ValueError("Training or validation dataset is empty after processing. Please check your data preparation steps.")

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=20,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=1e-5,
    learning_rate=5e-5,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Fine-tune the model
def fine_tune_model():
    trainer.train()  # Continue training the existing model with the new dataset
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')
    print("Model fine-tuning complete and saved to './fine_tuned_model'")

def evaluate_model():
    print("Evaluating model on the validation set...")
    predictions = trainer.predict(val_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    report = classification_report(labels, preds, target_names=label_mapping.keys())
    print("Validation Set Evaluation Results:\n", report)

if __name__ == "__main__":
    print("Starting Model Fine-tuning...")
    fine_tune_model()
    print("Model Fine-tuning Finished.")
    evaluate_model()