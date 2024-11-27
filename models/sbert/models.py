import torch
import torch.nn as nn

class SBERTSoftmax(nn.Module):
    def __init__(self, model):
        super(SBERTSoftmax, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.model.config.hidden_size * 3, 3)

    def forward(self, input_ids_resume, attention_mask_resume, input_ids_job_description, attention_mask_job_description):
        outputs_resume = self.model(input_ids_resume, attention_mask=attention_mask_resume)
        outputs_job_description = self.model(input_ids_job_description, attention_mask=attention_mask_job_description)

        pooled_outputs_resume = outputs_resume.last_hidden_state[:, 0, :]
        pooled_outputs_job_description = outputs_job_description.last_hidden_state[:, 0, :]

        abs_diff = torch.abs(pooled_outputs_resume - pooled_outputs_job_description)

        combined = self.dropout(torch.cat([pooled_outputs_resume, pooled_outputs_job_description, abs_diff], dim=1))
        logits = self.fc(combined)

        return logits
    
class SBERTCosineSimilarity(nn.Module):
    def __init__(self, model):
        super(SBERTCosineSimilarity, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.model.config.hidden_size * 2 + 1, 3) # + 1 for cosine similarity
    
    def forward(self, input_ids_resume, attention_mask_resume, input_ids_job_description, attention_mask_job_description):
        outputs_resume = self.model(input_ids_resume, attention_mask=attention_mask_resume)
        outputs_job_description = self.model(input_ids_job_description, attention_mask=attention_mask_job_description)

        pooled_outputs_resume = outputs_resume.last_hidden_state[:, 0, :]
        pooled_outputs_job_description = outputs_job_description.last_hidden_state[:, 0, :]

        cos = nn.CosineSimilarity(dim=1)
        output = cos(pooled_outputs_resume,pooled_outputs_job_description) # Shape: (batch_size, )

        # unsqueeze [similarity1, similarity2, similarity3, ...] -> [[similarity1],[similarity2],..]
        combined = torch.cat([pooled_outputs_resume,pooled_outputs_job_description,output.unsqueeze(1)],dim=1)
        combined = self.dropout(combined)
        logits = self.fc(combined)

        return logits

class SBERTHybrid(nn.Module):
    def __init__(self, model):
        super(SBERTCosineSimilarity, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.model.config.hidden_size * 3 + 1, 3) # + 1 for cosine similarityf
    
    def forward(self, input_ids_resume, attention_mask_resume, input_ids_job_description, attention_mask_job_description):
        outputs_resume = self.model(input_ids_resume, attention_mask=attention_mask_resume)
        outputs_job_description = self.model(input_ids_job_description, attention_mask=attention_mask_job_description)

        pooled_outputs_resume = outputs_resume.last_hidden_state[:, 0, :]
        pooled_outputs_job_description = outputs_job_description.last_hidden_state[:, 0, :]

        abs_diff = torch.abs(pooled_outputs_resume - pooled_outputs_job_description)

        cos = nn.CosineSimilarity(dim=1)
        output = cos(pooled_outputs_resume,pooled_outputs_job_description) # Shape: (batch_size, )

        # combined both absolute difference and cosine similarity
        combined = torch.cat([pooled_outputs_resume,pooled_outputs_job_description,abs_diff,output.unsqueeze(1)],dim=1)
        combined = self.dropout(combined)
        logits = self.fc(combined)

        return logits

    
