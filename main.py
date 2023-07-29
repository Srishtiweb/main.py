import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

data = pd.read_csv('fake news/train.tsv', sep='\t', header=None)
data.columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state', 'party',
                'barely_true', 'false', 'half_true', 'mostly_true', 'pants_on_fire', 'context']


tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X = tfidf_vectorizer.fit_transform(data['statement'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = svm.SVC()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


new_articles = [
    "Scientists discover a new species of dinosaur in Africa.",
    "Aliens have landed in New York City."
]
X_new = tfidf_vectorizer.transform(new_articles)
predictions = model.predict(X_new)

for article, prediction in zip(new_articles, predictions):
    print(f"Article: {article}")
    print(f"Prediction: {prediction}")
    print()
