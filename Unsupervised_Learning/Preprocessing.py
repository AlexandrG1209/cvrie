from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def vectorize(texts):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(texts)

    return X, vectorizer


def reduce_dimensions(X, components=100):
    svd = TruncatedSVD(
        n_components=components,
        random_state=42
    )

    X_reduced = svd.fit_transform(X)

    return X_reduced, svd