from functools import lru_cache
import re
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
from pymorphy2 import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN


class Extractor:
    def __init__(self):
        self.word2vec = None
        self.morph = MorphAnalyzer()

    @lru_cache(20000)
    def _morph_parse(self, word):
        return self.morph.parse(word)

    def _tokenize(self, text):
        tokens = word_tokenize(text.lower())
        result = []
        for token in tokens:
            morph = self._morph_parse(token)
            if len(morph) > 0:
                if morph[0].tag.POS is not None:
                    result.append(morph[0])
        return ["{0}_{1}".format(morph.word, morph.tag.POS) for morph in result]

    def fit(self, texts, word2vec_params):
        converted_texts = [self._tokenize(text) for text in texts]
        self.word2vec = Word2Vec(converted_texts, **word2vec_params)

    def _tfidf_order_features(self, tfidf, matrix):
        mean_features = np.asarray(matrix.mean(axis=0))[0]
        ordered_features = mean_features.argsort()[::-1]
        feature_names = tfidf.get_feature_names()
        result = []
        for feature in ordered_features:
            result.append(feature_names[feature])
        return np.array(result)

    def _tfidf_feature_filter(self, features):
        ignorance_filter = lambda text: bool(re.match(".*prep", text)) or \
                                        bool(re.match(".*infn", text)) or \
                                        bool(re.match(".*verb", text))
        feature_filter = lambda text: bool(re.match(".*adjf .*noun", text)) and not ignorance_filter(text)
        return [feature for feature in features if feature_filter(feature)]

    def _document_distance(self, doc1, doc2):
        doc1vec = np.array([np.zeros([self.word2vec.vector_size])] +
                           [self.word2vec[token] for token in doc1.split(" ")
                            if token in self.word2vec]).sum(axis=0)
        doc2vec = np.array([np.zeros([self.word2vec.vector_size])] +
                           [self.word2vec[token] for token in doc2.split(" ")
                            if token in self.word2vec]).sum(axis=0)
        return cosine(doc1vec, doc2vec)

    def _top_features(self, converted_texts, ngram_min, ngram_max, top_tfidf_features):
        features = []
        for size in range(ngram_min, ngram_max + 1):
            tfidf = TfidfVectorizer(ngram_range=(ngram_min, ngram_max))
            tfidf_transformed_texts = tfidf.fit_transform(converted_texts)
            tfidf_features = self._tfidf_order_features(tfidf, tfidf_transformed_texts)
            top_features = self._tfidf_feature_filter(tfidf_features)[:top_tfidf_features]
            features += top_features
        features = list(set(features))
        features.sort()
        return features

    def _feature_distances(self, features):
        distances = np.zeros([len(features), len(features)])
        for i, feature1 in enumerate(features):
            for j, feature2 in enumerate(features):
                features_distance = self._document_distance(feature1, feature2)
                distances[i, j] = features_distance
                distances[j, i] = features_distance
        return distances

    def _cluster_features(self, features, distances):
        dbscan = DBSCAN(0.2, min_samples=1, metric="precomputed")
        clusters = dbscan.fit_predict(distances)
        items = {}
        for cluster, feature in zip(clusters, features):
            items[cluster] = items.get(cluster, []) + [feature]
        return items

    def _choose_features(self, features, distances):
        choosen_ngrams = []
        for key, values in self._cluster_features(features, distances).items():
            values_indices = np.array([features.index(val) for val in values])
            if len(values) < 2:
                continue
            values_distances = distances[values_indices, :][:, values_indices]
            index_mean_distances = np.zeros([len(values_indices)])
            for i in range(0, len(values_indices)):
                index_mean_distances[i] = np.delete(values_distances[i], i, axis=0).mean()
            choosen_ngram = values[index_mean_distances.argmin()]
            choosen_ngrams.append(choosen_ngram)
        return choosen_ngrams

    def _apply_rules(self, rules, text):
        if isinstance(text, list):
            return [self._apply_rules(rules, item) for item in text]
        for rule in rules:
            text = rule(text)
        return text

    def transform(self, texts, ngram_min, ngram_max, top_tfidf_features):
        converted_texts = [" ".join(self._tokenize(text)) for text in texts]
        features = self._top_features(converted_texts, ngram_min, ngram_max, top_tfidf_features)
        distances = self._feature_distances(features)
        choosen_features = self._choose_features(features, distances)

        rules = [lambda text: re.sub("_adjf+ (\w+)_intj", ", \g<1>", text),
                 lambda text: re.sub("_noun+ (\w+)_adjf", ", \g<1>", text),
                 lambda text: re.sub("^\w+_conj", "", text),
                 lambda text: re.sub("\w+_conj$", "", text),
                 lambda text: re.sub("^\w+_pred", "", text),
                 lambda text: re.sub("\w+_pred$", "", text),
                 lambda text: re.sub("^\w+_precl", "", text),
                 lambda text: re.sub("\w+_precl$", "", text),
                 lambda text: re.sub("_[a-z]+", "", text),
                 lambda text: text.strip()]
        return self._apply_rules(rules, choosen_features)