import torch
import torch.nn as nn

class SBERTSoftmax(nn.Module):
    def __init__(self, model):
        super(SBERTSoftmax, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(p=0.2)
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
        self.dropout = nn.Dropout(p=0.2)
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
    def __init__(self, model1, model2):
        super(SBERTHybrid, self).__init__()
        self.model_resume_1 = model1
        self.model_resume_2 = model2
        self.model_job_1 = model1
        self.model_job_2 = model2

        self.dropout = nn.Dropout(p=0.2)
        self.bn = nn.BatchNorm1d(model1.config.hidden_size * 6 + 1)  # Adjust for concatenated embeddings
        self.fc = nn.Linear(model1.config.hidden_size * 6 + 1, 3)  # +1 for cosine similarity
    
    def forward(self, input_ids_resume1, attention_mask_resume1, input_ids_resume2, attention_mask_resume2,
                input_ids_job1, attention_mask_job1, input_ids_job2, attention_mask_job2):
        # Forward pass for resume
        outputs_resume_1 = self.model_resume_1(input_ids_resume1, attention_mask=attention_mask_resume1)
        outputs_resume_2 = self.model_resume_2(input_ids_resume2, attention_mask=attention_mask_resume2)

        # Forward pass for job description
        outputs_job_1 = self.model_job_1(input_ids_job1, attention_mask=attention_mask_job1)
        outputs_job_2 = self.model_job_2(input_ids_job2, attention_mask=attention_mask_job2)

        # Pooling the CLS tokens
        pooled_outputs_resume_1 = outputs_resume_1.last_hidden_state[:, 0, :]
        pooled_outputs_resume_2 = outputs_resume_2.last_hidden_state[:, 0, :]
        pooled_outputs_job_1 = outputs_job_1.last_hidden_state[:, 0, :]
        pooled_outputs_job_2 = outputs_job_2.last_hidden_state[:, 0, :]

        # Combine outputs for resume and job
        pooled_outputs_resume = torch.cat([pooled_outputs_resume_1, pooled_outputs_resume_2], dim=1)
        pooled_outputs_job = torch.cat([pooled_outputs_job_1, pooled_outputs_job_2], dim=1)

        # Compute absolute difference and cosine similarity
        abs_diff = torch.abs(pooled_outputs_resume - pooled_outputs_job)
        cos = nn.CosineSimilarity(dim=1)
        output_cos = cos(pooled_outputs_resume, pooled_outputs_job).unsqueeze(1)

        # Concatenate all features
        combined = torch.cat([pooled_outputs_resume, pooled_outputs_job, abs_diff, output_cos], dim=1)
        if combined.size(0) > 1:  # Apply BatchNorm only for batch size > 1
            combined = self.bn(combined)
        combined = self.dropout(combined)
        logits = self.fc(combined)

        return logits

    
