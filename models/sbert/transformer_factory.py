from transformers import DistilBertTokenizer, DistilBertModel
from transformers import BertTokenizer, BertModel

class TransformerFactory:
    def __init__(self):
        self.config = {
            'distilbert': {
                'name': 'distilbert-base-uncased',
                'tokenizer': DistilBertTokenizer,
                'model': DistilBertModel
            },
            'bert': {
                'name': 'bert-base-uncased',
                'tokenizer': BertTokenizer,
                'model': BertModel
            }
        }
    
    def get_tokenizer_and_model(self, model_name):
        tokenizer = self.config[model_name]['tokenizer'].from_pretrained(self.config[model_name]['name'])
        model = self.config[model_name]['model'].from_pretrained(self.config[model_name]['name'])

        return tokenizer, model