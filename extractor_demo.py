from gensim.models import KeyedVectors
from extractor import Extractor
import pandas as pd
import sys


if __name__ == '__main__':
    assert len(sys.argv) == 5, "Need trained word2vec path / dataset path / product id  / max ngrams per tfidf"
    w2v_path = sys.argv[1]
    w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    df_path = sys.argv[2]
    df = pd.read_csv(df_path)
    product_id = int(sys.argv[3])
    max_ngram_per_tfidf = int(sys.argv[4])
    index = df["PRODUCT"] == product_id
    texts = list(df.loc[index, "TEXT"]) + \
            list(df.loc[index, "BENEFITS"]) + \
            list(df.loc[index, "DRAWBACKS"])
    texts = list(map(str, filter(bool, texts)))
    extractor = Extractor()
    extractor.word2vec = w2v
    extracted = extractor.transform(texts, 1, 4, max_ngram_per_tfidf)
    print(extracted)