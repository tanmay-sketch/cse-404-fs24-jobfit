from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

# Load datasets
processed_df = pd.read_csv('https://media.githubusercontent.com/media/tanmay-sketch/cse-404-fs24-jobfit/refs/heads/tanmay/data/processed_train_data.csv')
processed_eval = pd.read_csv('https://media.githubusercontent.com/media/tanmay-sketch/cse-404-fs24-jobfit/refs/heads/tanmay/data/processed_eval_data.csv')
processed_test = pd.read_csv('https://media.githubusercontent.com/media/tanmay-sketch/cse-404-fs24-jobfit/refs/heads/tanmay/data/processed_test_data.csv')

X_train = processed_df[['resume_text', 'job_description_text']].copy()
y_train = processed_df['label']
X_eval = processed_eval[['resume_text', 'job_description_text']].copy()
y_eval = processed_eval['label']
X_test = processed_test[['resume_text', 'job_description_text']].copy()
y_test = processed_test['label']

# Transforming to cosine similarity
class CosineSimilarityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def fit(self, X, y=None):
        # Handle both pandas DataFrame and NumPy array input
        if isinstance(X, pd.DataFrame):
            texts = pd.concat([X.iloc[:, 0], X.iloc[:, 1]], axis=0)  # Combine columns for fitting
        else:
            texts = np.concatenate([X[:, 0], X[:, 1]], axis=0)  # For NumPy array input
        self.vectorizer.fit(texts)
        return self

    def transform(self, X):
        # Handle both pandas DataFrame and NumPy array input
        if isinstance(X, pd.DataFrame):
            resume_texts = X.iloc[:, 0]
            job_desc_texts = X.iloc[:, 1]
        else:
            resume_texts = X[:, 0]
            job_desc_texts = X[:, 1]

        resume_tfidf = self.vectorizer.transform(resume_texts)
        job_desc_tfidf = self.vectorizer.transform(job_desc_texts)
        similarities = cosine_similarity(resume_tfidf, job_desc_tfidf).diagonal()
        return np.array(similarities).reshape(-1, 1)

# Define TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(binary=True)

# Create a transformer for the features
feature_transformer = ColumnTransformer([
    ('resume_tfidf', TfidfVectorizer(binary=True), 'resume_text'),
    ('job_desc_tfidf', TfidfVectorizer(binary=True), 'job_description_text'),
    ('cosine_similarity', CosineSimilarityTransformer(tfidf_vectorizer), ['resume_text', 'job_description_text'])
])

# Define the pipeline
pipeline = Pipeline([
    ('features', feature_transformer),
    ('clf', KNeighborsClassifier())
])

# Define hyperparameters for grid search
param_grid = {
    'features__resume_tfidf__max_features': [1000, 3000, 5000, 10000],
    'features__job_desc_tfidf__max_features': [1000, 3000, 5000, 10000],
    'clf__n_neighbors': [3, 5, 7],  # Number of neighbors
    'clf__weights': ['uniform', 'distance'],  # Weighting for neighbors
    'clf__metric': ['euclidean', 'manhattan', 'cosine']  # Distance metrics
}

# Perform grid search with cross-validation on the training data
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Output best parameters
print("Best parameters found: ", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Evaluate on the eval set
eval_predictions = best_model.predict(X_eval)
eval_accuracy = accuracy_score(y_eval, eval_predictions)
print("Evaluation accuracy: ", eval_accuracy)

# Finally, test on the test set
test_predictions = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Test accuracy: ", test_accuracy)