from gensim.models import KeyedVectors
from extractor import Extractor
import pandas as pd
import sys


if __name__ == '__main__':
    assert len(sys.argv) == 5, "Need trained word2vec path / dataset path / product id (-1 to work with all)  / max ngrams per tfidf"
    w2v_path = sys.argv[1]
    w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    extractor = Extractor()
    extractor.word2vec = w2v
    df_path = sys.argv[2]
    df = pd.read_csv(df_path)
    products = list(set(df["PRODUCT"]))
    products.sort()
    product_id = int(sys.argv[3])


    def shor_product(product_id):
        max_ngram_per_tfidf = int(sys.argv[4])
        index = df["PRODUCT"] == product_id
        texts = list(df.loc[index, "TEXT"]) + \
                list(df.loc[index, "BENEFITS"]) + \
                list(df.loc[index, "DRAWBACKS"])
        texts = list(map(str, filter(bool, texts)))
        extracted = extractor.transform(texts, 1, 4, max_ngram_per_tfidf)
        print(product_id)
        print(extracted)

    if product_id == -1:
        for id in products:
            shor_product(id)
    else:
        shor_product(product_id)