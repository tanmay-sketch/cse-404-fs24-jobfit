import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import wandb
from models import SBERTHybrid
from load_data import TokenizeDataLoaders
from dotenv import load_dotenv
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import csv
from transformer_factory import TransformerFactory

# Load the WandB API key
load_dotenv()
wandb_api_key = os.getenv('WANDB_API_KEY')
if wandb_api_key is None:
    raise ValueError("WANDB_API_KEY not found. Please set it in the .env file.")
wandb.login(key=wandb_api_key)

# WANDB Configuration
config = {
    'batch_size': 4,
    'epochs': 10,
    'learning_rate': 3e-5,
    'model': 'SBERTHybrid',
    'llm': 'bert',
    'optimizer': 'Adam',
    'loss': 'CrossEntropyLoss',
    'weight_decay': 1e-4
}

run = wandb.init(
    project="jobfit",
    config=config,
    name="sbert-training-21",
    reinit=True
)

config = wandb.config

# Load datasets
train_df = pd.read_csv('https://media.githubusercontent.com/media/tanmay-sketch/cse-404-fs24-jobfit/refs/heads/tanmay/data/processed_train_data.csv')
eval_df = pd.read_csv('https://media.githubusercontent.com/media/tanmay-sketch/cse-404-fs24-jobfit/refs/heads/tanmay/data/processed_eval_data.csv')
test_df = pd.read_csv('https://media.githubusercontent.com/media/tanmay-sketch/cse-404-fs24-jobfit/refs/heads/tanmay/data/processed_test_data.csv')

# Add unique IDs to datasets
train_df['id'] = range(len(train_df))
eval_df['id'] = range(len(eval_df))
test_df['id'] = range(len(test_df))

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

# Initialize tokenizers and models
transformer_handler = TransformerFactory()
tokenizer, model1 = transformer_handler.get_tokenizer_and_model('bert')
_, model2 = transformer_handler.get_tokenizer_and_model('bert')  # Second BERT model

model1 = model1.to(device)
model2 = model2.to(device)

# Initialize DataLoaders
data_loaders = TokenizeDataLoaders(
    tokenizer=tokenizer,
    batch_size=config.batch_size,
    max_length=512,
    num_workers=1  
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
        # Move data to device
        input_ids_resume1 = batch['input_ids_resume1'].to(device)
        attention_mask_resume1 = batch['attention_mask_resume1'].to(device)
        input_ids_resume2 = batch['input_ids_resume2'].to(device)
        attention_mask_resume2 = batch['attention_mask_resume2'].to(device)
        input_ids_job1 = batch['input_ids_job1'].to(device)
        attention_mask_job1 = batch['attention_mask_job1'].to(device)
        input_ids_job2 = batch['input_ids_job2'].to(device)
        attention_mask_job2 = batch['attention_mask_job2'].to(device)
        labels = batch['labels'].to(device)

        # Forward, backward, optimize
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

    wandb.log({"epoch": epoch + 1, "train_loss": epoch_loss / len(train_loader)})
    print(f"Epoch {epoch + 1}, Training Loss: {epoch_loss / len(train_loader):.4f}")

# Evaluation
def evaluate(loader, dataset_df, name="Eval"):
    sberthybrid.eval()
    eval_loss, correct_preds, total_preds = 0, 0, 0
    y_true, y_pred = [], []
    misclassified_samples = []
    all_predictions = []

    with torch.no_grad():
        for batch in loader:
            input_ids_resume1 = batch['input_ids_resume1'].to(device)
            attention_mask_resume1 = batch['attention_mask_resume1'].to(device)
            input_ids_resume2 = batch['input_ids_resume2'].to(device)
            attention_mask_resume2 = batch['attention_mask_resume2'].to(device)
            input_ids_job1 = batch['input_ids_job1'].to(device)
            attention_mask_job1 = batch['attention_mask_job1'].to(device)
            input_ids_job2 = batch['input_ids_job2'].to(device)
            attention_mask_job2 = batch['attention_mask_job2'].to(device)
            labels = batch['labels'].to(device)
            ids = batch['id']
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

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            # Collect predictions with IDs
            for i in range(labels.size(0)):
                all_predictions.append({
                    "id": ids[i].item(),
                    "true_label": labels[i].item(),
                    "predicted_label": predicted[i].item()
                })

                if predicted[i] != labels[i]:
                    misclassified_samples.append({
                        "true_label": labels[i].item(),
                        "predicted_label": predicted[i].item(),
                        "resume_text": tokenizer.decode(batch['input_ids_resume1'][i].cpu(), skip_special_tokens=True),
                        "job_description_text": tokenizer.decode(batch['input_ids_job1'][i].cpu(), skip_special_tokens=True)
                    })

    eval_accuracy = correct_preds / total_preds
    eval_loss /= len(loader)
    print(f"{name} Loss: {eval_loss:.4f}, {name} Accuracy: {eval_accuracy:.4f}")
    wandb.log({f"{name.lower()}_loss": eval_loss, f"{name.lower()}_accuracy": eval_accuracy})
    return y_true, y_pred, misclassified_samples, all_predictions

try:
    # Evaluate on test data
    y_true, y_pred, misclassified_samples, all_predictions = evaluate(test_loader, test_df, name="Test")

    # Save predictions
    all_predictions_df = pd.DataFrame(all_predictions)
    all_predictions_df.to_csv("all_predictions.csv", index=False)

    # Precision, Recall, F1 Score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    wandb.log({
        "precision": precision, 
        "recall": recall, 
        "f1": f1})

    # Confusion Matrix and Visualization
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(y_true), yticklabels=set(y_true))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})
    plt.close()

except Exception as e:
    print(f"An error occurred: {e}")
    wandb.log({"error": str(e)})

finally:
     # Save the model
    model_save_path = "sberthybrid_model.pt"
    torch.save(sberthybrid.state_dict(), model_save_path)
    wandb.save(model_save_path)

    print("Cleaning up and finishing the run...")
    run.finish()