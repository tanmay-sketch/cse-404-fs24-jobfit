import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TokenizeData(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        """
        df: pandas DataFrame with columns 'resume_text, 'job_description_text','label'
        """

        self.resume_text = df['resume_text'].tolist()
        self.job_description_text = df['job_description_text'].tolist()
        assert len(self.resume_text) == len(self.job_description_text) == len(self.labels)
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.resume_text)
    
    def __getitem__(self, idx):
        encoding_resume = self.tokenizer(
            self.resume_text[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        encoding_job_description = self.tokenizer(
            self.job_description_text[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {
            'input_ids_resume': encoding_resume['input_ids'].squeeze(0),
            'attention_mask_resume': encoding_resume['attention_mask'].squeeze(0),
            'input_ids_job_description': encoding_job_description['input_ids'].squeeze(0),
            'attention_mask_job_description': encoding_job_description['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

        return item


def get_tokinized_data_loaders(train_data, eval_data, test_data, batch_size=4):
    train_dataset = TokenizeData(train_data)
    eval_dataset = TokenizeData(eval_data)
    test_dataset = TokenizeData(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, eval_loader, test_loader