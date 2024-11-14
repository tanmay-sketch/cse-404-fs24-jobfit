# cse-404-fs24-jobfit
Repository for CSE 404 Project

Setup Instructions:

1. Clone the repository
```
git clone https://github.com/tanmay-sketch/cse-404-fs24-jobfit.git
```

2. Create conda environment
```
conda create -n jobfit_env python=3.10
``` 

3. Activate the environment
```
conda activate jobfit_env
```

4. Install the required packages
```
pip install -r requirements.txt
```

## Step 5 Performance's

This project uses four ML appoaches, including logistic regression, K-Nearest Neighbors (KNN), Multinomial Naive Bayes, and Bernoulli Naive Bayes. These are used to classify and analyze job descriptions and resumes based on the similarities and relevances in the text. Each model has its own strengths in manipulating classification tasks and text data, which together give us a comprehensive comparison across different algorithms. Below, each model is described.

### Models Used

#### 1. Logistic Regression (LR)

LR is a common linear model for both bunary and multiclass classification. It shines when used with sparse data, making it a strong choice for text classification, in which the TF-IDF vectorizer creates sparce feature matrices.

LR is often a strong baseline model for text classification tasks. Because it is linear in nature, it is efficient and interpretable, especially with high-dimensional data like text.

LR shows consistent accuracy with minimal overfitting. It gives us a solid baseline for comparing, although this simplicity did limit its ability to find complex relationships in the data.

From our model:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=500))
])

param_grid = {
    'tfidf__max_features': [1000, 3000, 5000, 10000],
    'clf__C': [0.1, 1, 10]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X, y)
```

#### 2. K-Nearest Neighbor (KNN)

KNN is an instance-based algorith that classifies samples based on the highest majority label of its nearest neighbors inside of the feature space. In our model, we used TF-IDF to turn the text data into feature vectors, allowing KNN to find the distances between the sample.

We decided to use KNN in order to observe the impact of a non-linear and distance-based model on our text data. KNN's performance depends on selecting the appropriate number of neighbors as well as a distance metric, making it a strong candidate for parameter tuning.

Although KNN is effective in finding local patterns in the data, it was overall slower and prone to overfitting on our multi-dimensional TF-IDF features as opposed to other models. However, it was powerful when it came to capruring sample-based relationships that some linear models might midd.

From our model:
```python
from sklearn.neighbors import KNeighborsClassifier

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', KNeighborsClassifier(metric='euclidean'))
])

param_grid = {
    'tfidf__max_features': [1000, 3000, 5000, 10000],
    'clf__n_neighbors': [3, 5, 7]
}

grid_search_knn = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search_knn.fit(X, y)
```

#### 3. Multinomial Naive Bayes (MNB)

MNB is a probabilistic classifier that is based on the Bayes' Theorem. It assumes that features or words are independent given the class. It is often utilized in text classification because it directly models the distributon of word counts.

We included MNB because of its known strong performance in text classification, particularly with TF-IDF or word count vectors. It is able to handle multi-dimensional data well and often serves as a strong baseline for text-classification.

MNV performed well, with quick training and high accuracy on both sets. But, its independence assumption limits its ability to find dependencies inbetween words, so it may underperform on complicated and context-heavy data.

From our model:
```python
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', MultinomialNB())
])

param_grid = {
    'tfidf__max_features': [1000, 3000, 5000, 10000],
    'clf__alpha': [0.1, 1, 10]
}

grid_search_nb = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search_nb.fit(X, y)
```

#### 4. Bernoulli Naive Bayes (BNB)

BNB is another variation of Naive Bayes that works particularly well with binary features, where each word being present or absent influences the classification. Similar to MNB, it assumes independence inbetween features.

We included BNB in order to compare the effects of treating words as either present or absent instead of their frequencies. Its useful for short text or datasets in which binary word presence can be meaningful.

NBM showed a comparable performance to MNB but was slightly worse in accuracy because of the high dimensionality of TF-IDF data and lack of boolean encoding. However, it give insight into how the binary feature assumption can affect performance.

From our model:
```python
from sklearn.naive_bayes import BernoulliNB

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', BernoulliNB())
])

param_grid = {
    'tfidf__max_features': [1000, 3000, 5000, 10000],
    'clf__alpha': [0.1, 1, 10]
}

grid_search_bernoulli_nb = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search_bernoulli_nb.fit(X, y)
```


#### 5. Siamese BERT Neural Network

We also implemented a Siamese BERT Neural Network model to compare the performance of traditional machine learning models with deep learning models. The Siamese BERT model is a neural network that uses the BERT model to encode the input text data and then uses a similarity function to compare the embeddings of the two input texts. The model is trained to predict whether the two input texts are similar or dissimilar. The model was only trained for 3 epochs due to computational constraints.

From our model:
```python
class SBERT(nn.Module):
    def __init__(self, bert_model):
        super(SBERT, self).__init__()
        self.bert = bert_model
        self.fc = nn.Linear(self.bert.config.hidden_size * 3, 3)  # 3 classes for classification

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        outputs_a = self.bert(input_ids_a, attention_mask=attention_mask_a)
        pooled_output_a = outputs_a.last_hidden_state[:, 0, :]  # CLS token for sentence A

        outputs_b = self.bert(input_ids_b, attention_mask=attention_mask_b)
        pooled_output_b = outputs_b.last_hidden_state[:, 0, :]  # CLS token for sentence B

        abs_diff = torch.abs(pooled_output_a - pooled_output_b)
        combined = torch.cat((pooled_output_a, pooled_output_b, abs_diff), dim=1)
        logits = self.fc(combined)

        return logits  # Return raw logits; softmax will be applied in loss calculation
```

Performance of the model:
* Evaluation accuracy: 0.704
* Testing accuracy: 0.5571


### Team Members
- Sasi Vattikuti
- Ryan Bolin
- Shantanu Ingalagi
- Tanmay Grandhisiri
- Sanya Nigam

