import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import wandb
from models import SBERTHybrid
from load_data import TokenizeDataLoaders
from dotenv import load_dotenv
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import csv
from transformer_factory import TransformerFactory

# Loading the API key from the .env file
load_dotenv()
wandb_api_key = os.getenv('WANDB_API_KEY')
if wandb_api_key is None:
    raise ValueError("WANDB_API_KEY not found. Please set it in the .env file.")
wandb.login(key=wandb_api_key)

config = {
    'batch_size': 4,
    'epochs': 10,
    'learning_rate': 3e-5,
    'model': 'SBERTHybrid',
    'llm': 'distilbert',
    'optimizer': 'Adam',
    'loss': 'CrossEntropyLoss',
    'weight_decay': 1e-4
}

run = wandb.init(
    project="jobfit",
    config=config,
    name="sbert-training-17",
    reinit=True
)

config = wandb.config

# Load datasets
train_df = pd.read_csv('https://media.githubusercontent.com/media/tanmay-sketch/cse-404-fs24-jobfit/refs/heads/tanmay/data/processed_train_data.csv')
eval_df = pd.read_csv('https://media.githubusercontent.com/media/tanmay-sketch/cse-404-fs24-jobfit/refs/heads/tanmay/data/processed_eval_data.csv')
test_df = pd.read_csv('https://media.githubusercontent.com/media/tanmay-sketch/cse-404-fs24-jobfit/refs/heads/tanmay/data/processed_test_data.csv')

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

num_gpus = torch.cuda.device_count()
if num_gpus > 1 and config.batch_size % num_gpus != 0:
    config.batch_size = (config.batch_size // num_gpus) * num_gpus
    print(f"Adjusted batch size to {config.batch_size} for {num_gpus} GPUs")
else:
    print(f'Using device: {device}')

# Initialize tokenizers and models
transformer_handler = TransformerFactory()

tokenizer, model1 = transformer_handler.get_tokenizer_and_model('bert')
_, model2 = transformer_handler.get_tokenizer_and_model('bert')  # Initialize a second BERT model

model1 = model1.to(device)
model2 = model2.to(device)

# Initialize DataLoaders
data_loaders = TokenizeDataLoaders(
    tokenizer=tokenizer,
    batch_size=config.batch_size,
    max_length=512,
    num_workers=2
)

train_loader, eval_loader, test_loader = data_loaders.get_tokenized_data_loaders(train_df, eval_df, test_df) 

# Initialize the model
sberthybrid = SBERTHybrid(model1, model2).to(device)

if torch.cuda.device_count() > 1:
    sberthybrid = nn.DataParallel(sberthybrid)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(sberthybrid.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
num_epochs = config.epochs

# Training and Evaluation Loop
for epoch in range(num_epochs):
    sberthybrid.train()
    epoch_loss = 0
    for batch in train_loader:
        # Split input into parts for each model
        input_ids_resume1 = batch['input_ids_resume1'].to(device)
        attention_mask_resume1 = batch['attention_mask_resume1'].to(device)
        input_ids_resume2 = batch['input_ids_resume2'].to(device)
        attention_mask_resume2 = batch['attention_mask_resume2'].to(device)
        input_ids_job1 = batch['input_ids_job1'].to(device)
        attention_mask_job1 = batch['attention_mask_job1'].to(device)
        input_ids_job2 = batch['input_ids_job2'].to(device)
        attention_mask_job2 = batch['attention_mask_job2'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = sberthybrid(
            input_ids_resume1, attention_mask_resume1, 
            input_ids_resume2, attention_mask_resume2, 
            input_ids_job1, attention_mask_job1, 
            input_ids_job2, attention_mask_job2
        )
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": epoch_loss / len(train_loader)
    })

    print(f"Epoch {epoch + 1}, Training Loss: {epoch_loss / len(train_loader)}")

    sberthybrid.eval()
    eval_loss = 0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for batch in eval_loader:
            input_ids_resume1 = batch['input_ids_resume1'].to(device)
            attention_mask_resume1 = batch['attention_mask_resume1'].to(device)
            input_ids_resume2 = batch['input_ids_resume2'].to(device)
            attention_mask_resume2 = batch['attention_mask_resume2'].to(device)
            input_ids_job1 = batch['input_ids_job1'].to(device)
            attention_mask_job1 = batch['attention_mask_job1'].to(device)
            input_ids_job2 = batch['input_ids_job2'].to(device)
            attention_mask_job2 = batch['attention_mask_job2'].to(device)
            labels = batch['labels'].to(device)

            logits = sberthybrid(
                input_ids_resume1, attention_mask_resume1, 
                input_ids_resume2, attention_mask_resume2, 
                input_ids_job1, attention_mask_job1, 
                input_ids_job2, attention_mask_job2
            )
            loss = loss_fn(logits, labels)
            eval_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

    eval_accuracy = correct_preds / total_preds
    eval_loss /= len(eval_loader)

    wandb.log({
        "epoch": epoch + 1,
        "eval_loss": eval_loss,
        "eval_accuracy": eval_accuracy
    })

    print(f"Epoch {epoch + 1}, Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}")

# Test Evaluation and Metrics
sberthybrid.eval()
test_loss = 0
correct_preds = 0
total_preds = 0
misclassified_samples = []

with torch.no_grad():
    for batch in test_loader:
        input_ids_resume1 = batch['input_ids_resume1'].to(device)
        attention_mask_resume1 = batch['attention_mask_resume1'].to(device)
        input_ids_resume2 = batch['input_ids_resume2'].to(device)
        attention_mask_resume2 = batch['attention_mask_resume2'].to(device)
        input_ids_job1 = batch['input_ids_job1'].to(device)
        attention_mask_job1 = batch['attention_mask_job1'].to(device)
        input_ids_job2 = batch['input_ids_job2'].to(device)
        attention_mask_job2 = batch['attention_mask_job2'].to(device)
        labels = batch['labels'].to(device)

        logits = sberthybrid(
            input_ids_resume1, attention_mask_resume1, 
            input_ids_resume2, attention_mask_resume2, 
            input_ids_job1, attention_mask_job1, 
            input_ids_job2, attention_mask_job2
        )
        loss = loss_fn(logits, labels)
        test_loss += loss.item()

        _, predicted = torch.max(logits, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

        for i in range(labels.size(0)):
            if predicted[i] != labels[i]:
                misclassified_samples.append({
                    "true_label": labels[i].item(),
                    "predicted_label": predicted[i].item(),
                    "resume_text": tokenizer.decode(batch['input_ids_resume1'][i].cpu(), skip_special_tokens=True),
                    "job_description_text": tokenizer.decode(batch['input_ids_job1'][i].cpu(), skip_special_tokens=True)
                })

test_loss /= len(test_loader)
test_accuracy = correct_preds / total_preds
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

wandb.log({
    "test_loss": test_loss,
    "test_accuracy": test_accuracy 
})

# Save misclassified examples
with open('misclassified_samples.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["true_label", "predicted_label", "resume_text", "job_description_text"])
    writer.writeheader()
    writer.writerows(misclassified_samples)

# Precision, Recall, F1 Score
y_true = []
y_pred = []
with torch.no_grad():
    for batch in test_loader:
        input_ids_resume1 = batch['input_ids_resume1'].to(device)
        attention_mask_resume1 = batch['attention_mask_resume1'].to(device)
        input_ids_resume2 = batch['input_ids_resume2'].to(device)
        attention_mask_resume2 = batch['attention_mask_resume2'].to(device)
        input_ids_job1 = batch['input_ids_job1'].to(device)
        attention_mask_job1 = batch['attention_mask_job1'].to(device)
        input_ids_job2 = batch['input_ids_job2'].to(device)
        attention_mask_job2 = batch['attention_mask_job2'].to(device)
        labels = batch['labels'].to(device)

        logits = sberthybrid(
            input_ids_resume1, attention_mask_resume1, 
            input_ids_resume2, attention_mask_resume2, 
            input_ids_job1, attention_mask_job1, 
            input_ids_job2, attention_mask_job2
        )
        _, predicted = torch.max(logits, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
wandb.log({
    "precision": precision,
    "recall": recall,
    "f1": f1
})

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
wandb.log({"confusion_matrix": wandb.Image(plt)})
plt.close()

run.finish()