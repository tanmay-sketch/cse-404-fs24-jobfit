import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer

class TokenizeData(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        """
        df: pandas DataFrame with columns 'resume_text', 'job_description_text', 'label'
        """
        self.resume_text = df['resume_text'].tolist()
        self.job_description_text = df['job_description_text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

        assert len(self.resume_text) == len(self.job_description_text) == len(self.labels)

    def __len__(self):
        return len(self.resume_text)
    
    def __getitem__(self, idx):
        resume_text = self.resume_text[idx]
        job_description_text = self.job_description_text[idx]

        # Split texts into two parts
        resume_part1 = resume_text[:len(resume_text) // 2]
        resume_part2 = resume_text[len(resume_text) // 2:]
        job_part1 = job_description_text[:len(job_description_text) // 2]
        job_part2 = job_description_text[len(job_description_text) // 2:]

        encoding_resume1 = self.tokenizer(
            resume_part1,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        encoding_resume2 = self.tokenizer(
            resume_part2,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        encoding_job1 = self.tokenizer(
            job_part1,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        encoding_job2 = self.tokenizer(
            job_part2,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {
            'input_ids_resume1': encoding_resume1['input_ids'].squeeze(0),
            'attention_mask_resume1': encoding_resume1['attention_mask'].squeeze(0),
            'input_ids_resume2': encoding_resume2['input_ids'].squeeze(0),
            'attention_mask_resume2': encoding_resume2['attention_mask'].squeeze(0),
            'input_ids_job1': encoding_job1['input_ids'].squeeze(0),
            'attention_mask_job1': encoding_job1['attention_mask'].squeeze(0),
            'input_ids_job2': encoding_job2['input_ids'].squeeze(0),
            'attention_mask_job2': encoding_job2['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

        return item

class TokenizeDataLoaders:
    def __init__(self, tokenizer, batch_size=4, max_length=512, num_workers=2):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers

    def get_tokenized_data_loaders(self, train_data, eval_data, test_data):
        train_dataset = TokenizeData(train_data, self.tokenizer, self.max_length)
        eval_dataset = TokenizeData(eval_data, self.tokenizer, self.max_length)
        test_dataset = TokenizeData(test_data, self.tokenizer, self.max_length)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,drop_last=True)
        eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,drop_last=False)

        return train_loader, eval_loader, test_loader