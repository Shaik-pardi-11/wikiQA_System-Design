from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_model(df):
    vectorizer = TfidfVectorizer()

    # Combine question + answer
    X = vectorizer.fit_transform(df['Question'] + " " + df['Sentence'])
    y = df['Label']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, vectorizer

def predict(model, vectorizer, question, sentences):
    inputs = [question + " " + s for s in sentences]

    X = vectorizer.transform(inputs)
    scores = model.predict_proba(X)[:, 1]

    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)

    return ranked