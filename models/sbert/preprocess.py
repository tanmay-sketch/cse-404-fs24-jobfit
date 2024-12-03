import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize and lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

df_train = pd.read_csv('../../data/train.csv')
df_test = pd.read_csv('../../data/test.csv')

df_train['resume_text'] = df_train['resume_text'].apply(preprocess_text)
df_train['job_description_text'] = df_train['job_description_text'].apply(preprocess_text)
unique_labels = {label: idx for idx, label in enumerate(df_train['label'].unique())}
df_train['label'] = df_train['label'].apply(lambda x: unique_labels[x])

df_test['resume_text'] = df_test['resume_text'].apply(preprocess_text)
df_test['job_description_text'] = df_test['job_description_text'].apply(preprocess_text)
df_test['label'] = df_test['label'].apply(lambda x: unique_labels[x])

train_data, eval_data = train_test_split(df_train, test_size=0.2, random_state=42,stratify=df_train['label'])

train_data.to_csv('processed_train_data.csv', index=False)
eval_data.to_csv('processed_eval_data.csv', index=False)
df_test.to_csv('processed_test_data.csv', index=False)

