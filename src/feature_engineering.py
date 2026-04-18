import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

def build_tfidf_vectorizer(max_features=5000):
    """
    Create TF-IDF vectorizer.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english"
    )

    return vectorizer


def fit_tfidf(vectorizer, train_texts):
    """
    Fit TF-IDF vectorizer on training data.
    """
    X_train = vectorizer.fit_transform(train_texts)

    return X_train


def transform_tfidf(vectorizer, texts):
    """
    Transform texts using trained TF-IDF vectorizer.
    """
    return vectorizer.transform(texts)

def tokenize_texts(texts):
    """
    Convert text into token lists.
    """
    return [text.split() for text in texts]


def train_word2vec(tokenized_texts,
                   vector_size=100,
                   window=5,
                   min_count=2):
    """
    Train Word2Vec model.
    """

    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4
    )

    return model


def sentence_vector(sentence, w2v_model, vector_size=100):
    """
    Convert a sentence into an averaged Word2Vec vector.
    """

    words = sentence.split()

    vectors = []

    for word in words:

        if word in w2v_model.wv:

            vectors.append(w2v_model.wv[word])

    if len(vectors) == 0:

        return np.zeros(vector_size)

    return np.mean(vectors, axis=0)


def build_sentence_vectors(texts, w2v_model, vector_size=100):
    """
    Convert a list of sentences into vectors.
    """

    return np.vstack([
        sentence_vector(text, w2v_model, vector_size)
        for text in texts
    ])
