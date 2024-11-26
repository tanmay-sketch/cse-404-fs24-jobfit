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