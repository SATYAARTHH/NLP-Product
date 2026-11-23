"""Sparse CPU classifiers and probability calibration."""

from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def text_classifier(character_features: bool = True) -> Pipeline:
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=35_000, sublinear_tf=True)
    if character_features:
        features = FeatureUnion([
            ("word", word),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=45_000, sublinear_tf=True)),
        ])
    else:
        features = word
    return Pipeline([
        ("tfidf", features),
        ("classifier", LogisticRegression(max_iter=1_500, class_weight="balanced", C=4.0, random_state=13)),
    ])


class TemperatureScaledClassifier:
    """Wrap a fitted probabilistic classifier with validation-set temperature scaling."""

    def __init__(self, estimator):
        self.estimator = estimator
        self.temperature = 1.0

    @property
    def classes_(self):
        return self.estimator.classes_

    def fit_temperature(self, texts, labels):
        probabilities = np.clip(self.estimator.predict_proba(texts), 1e-9, 1.0)
        target = np.array([np.where(self.classes_ == label)[0][0] for label in labels])
        logits = np.log(probabilities)
        best_loss = float("inf")
        for temperature in np.linspace(0.5, 5.0, 181):
            calibrated = softmax(logits / temperature, axis=1)
            loss = -np.log(calibrated[np.arange(len(target)), target]).mean()
            if loss < best_loss:
                best_loss = loss
                self.temperature = float(temperature)
        return self

    def predict_proba(self, texts):
        probabilities = np.clip(self.estimator.predict_proba(texts), 1e-9, 1.0)
        return softmax(np.log(probabilities) / self.temperature, axis=1)

    def predict(self, texts):
        probabilities = self.predict_proba(texts)
        return self.classes_[probabilities.argmax(axis=1)]
