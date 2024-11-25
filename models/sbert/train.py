import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import wandb
from models import SBERTSoftmax, distilbert_model
from load_data import get_tokenized_data_loaders
from dotenv import load_dotenv
from sklearn.metrics import precision_recall_fscore_support
import os

# Loading the API key from the .env file
load_dotenv()
wandb_api_key = os.getenv('WANDB_API_KEY')
if wandb_api_key is None:
    raise ValueError("WANDB_API_KEY not found. Please set it in the .env file.")
wandb.login(key=wandb_api_key)

config = {
    'batch_size': 4,
    'epochs': 10,
    'learning_rate': 2e-5,
    'model': 'SBERTSoftmax',
    'optimizer': 'Adam',
    'scheduler': 'StepLR',
    'scheduler_step_size': 1,
    'scheduler_gamma': 0.1,
    'loss': 'CrossEntropyLoss',
}

run = wandb.init(
    project="jobfit",
    config=config,
    name="sbert-training-2",
    reinit=True
)

config = wandb.config

train_df = pd.read_csv('../data/processed_train.csv')
eval_df = pd.read_csv('../data/processed_eval.csv')
test_df = pd.read_csv('../data/processed_test.csv')

train_loader, eval_loader, test_loader = get_tokenized_data_loaders(train_df, eval_df, test_df) 

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

model, tokenizer = distilbert_model()
model = model.to(device)

sbertsoftmax = SBERTSoftmax(model).to(device)
loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(sbertsoftmax.parameters(), lr=config.learning_rate)

# learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)    

num_epochs = 10

for epoch in range(num_epochs):
    sbertsoftmax.train()
    epoch_loss = 0
    for batch in train_loader:
        input_ids_resume = batch['input_ids_resume'].to(device)
        attention_mask_resume = batch['attention_mask_resume'].to(device)
        input_ids_job_description = batch['input_ids_job_description'].to(device)
        attention_mask_job_description = batch['attention_mask_job_description'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = sbertsoftmax(input_ids_resume, attention_mask_resume, input_ids_job_description, attention_mask_job_description)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": loss / len(train_loader)
    })

    print(f"Epoch {epoch}, Training Loss: {epoch_loss}")

    sbertsoftmax.eval()
    eval_loss = 0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for batch in eval_loader:
            input_ids_resume = batch['input_ids_resume'].to(device)
            attention_mask_resume = batch['attention_mask_resume'].to(device)
            input_ids_job_description = batch['input_ids_job_description'].to(device)
            attention_mask_job_description = batch['attention_mask_job_description'].to(device)
            labels = batch['labels'].to(device)

            logits = sbertsoftmax(input_ids_resume, attention_mask_resume, input_ids_job_description, attention_mask_job_description)
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

    scheduler.step()

# ------------ Evaluating Test Loss --------------
sbertsoftmax.eval()
test_loss = 0
correct_preds = 0
total_preds = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids_resume = batch['input_ids_resume'].to(device)
        attention_mask_resume = batch['attention_mask_resume'].to(device)
        input_ids_job_description = batch['input_ids_job_description'].to(device)
        attention_mask_job_description = batch['attention_mask_job_description'].to(device)
        labels = batch['labels'].to(device)

        logits = sbertsoftmax(input_ids_resume, attention_mask_resume, input_ids_job_description, attention_mask_job_description)
        loss = loss_fn(logits, labels)
        test_loss += loss.item()

        _, predicted = torch.max(logits, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

test_loss /= len(test_loader)
test_accuracy = correct_preds / total_preds
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

wandb.log({
    "test_loss": test_loss,
    "test_accuracy": test_accuracy 
})

# ------------ Evaluating Precision, Recall, F1 Score --------------
sbertsoftmax.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for batch in test_loader:
        input_ids_resume = batch['input_ids_resume'].to(device)
        attention_mask_resume = batch['attention_mask_resume'].to(device)
        input_ids_job_description = batch['input_ids_job_description'].to(device)
        attention_mask_job_description = batch['attention_mask_job_description'].to(device)
        labels = batch['labels'].to(device)

        logits = sbertsoftmax(input_ids_resume, attention_mask_resume, input_ids_job_description, attention_mask_job_description)
        _, predicted = torch.max(logits, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
    
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
wandb.log({
    "precision": precision,
    "recall": recall,
    "f1": f1
})


run.finish()