import numpy as np
import pandas as pd
import argparse
from collections import Counter


class BernoulliLogLikelihood:
    def train(self, train_data):
        """
        For a training set, the method computes the word likelihood as
        the document count of a word / total number of documents
        + Laplace smoothing
        """
        self.word_likelihood = (np.sum(train_data, axis=0) + 1) / sum(train_data.shape)

    def loglikelihood(self, test_data):
        """
        Given a test dataset, the method computes the log likelihood of each
        document.
        """
        return np.log(self.word_likelihood * test_data + (1 - test_data)
                      * (1 - self.word_likelihood)).sum(axis=1)


class BayesClassifier:
    def __init__(self, maximum_likelihood_models, priors):
        self.maximum_likelihood_models = maximum_likelihood_models
        self.priors = priors
        if len(self.maximum_likelihood_models) != len(self.priors):
            print('Number of ML models must be equal to number of priors!')
        self.n_classes = len(self.maximum_likelihood_models)

    def loglikelihood(self, test_data):
        """
        Returns a matrix of size number of test ex. x number of classes
        containing the log probabilities of each test example
        under each model, trained by BernoulliLogLikelihood.
        """

        log_pred = np.zeros((test_data.shape[0], self.n_classes))

        for i in range(self.n_classes):
            log_pred[:, i] = (self.maximum_likelihood_models[i].loglikelihood(test_data)
                              + np.log(self.priors[i]))
        return log_pred


class Classifier():
    def __init__(self, min_len=1, min_freq=4, quantile=0.2):
        self.min_len = min_len
        self.min_freq = min_freq
        self.quantile = quantile

    def bernoulli_vectorizer(self, corpus, vocabulary):
        """
        The method does a binary vectorization of the corpus,
        given a vocabulary.
        """
        corpus_split = [set(doc.split()) for doc in corpus]
        return np.array([[1 if w in doc else 0 for w in vocabulary]
                         for doc in corpus_split])

    def preselect_vocabulary(self, corpus):
        """
        The method extracts a vocabulary (a set) out of the given corpus of
        documents.
        """
        tokens = [word.strip(".,()'") for doc in corpus for word in doc.split()]
        tokens = [word for word in tokens if word.isalpha() and len(word) > self.min_len]
        occurrences = Counter(tokens)
        tokens = [w for w in tokens if occurrences[w] > self.min_freq]
        return set(tokens)

    def clean_corpus(self, corpus):
        """
        The method splits the documents of the corpus in tokens and performs
        a basic cleaning.
        """
        return np.array([" ".join([w.strip(".,()'") for w in doc.split()])
                        for doc in corpus])

    def tf_idf_filter(self, documents, terms):
        """
        The method reduces the size of the vocabulary based on
        calculated tf-idf weights.
        """
        N = len(documents)
        tf = {w: np.zeros(len(documents))*np.nan for w in terms}
        for i, doc in enumerate(documents):
            n_doc = len(doc)
            for w, n in Counter(doc.split()).items():
                if w in tf:
                    tf[w][i] = np.log(1+n/n_doc)
        tf = pd.DataFrame(tf)
        idf = np.log(N/tf.notnull().sum(0).values)
        tf_idf = tf.mul(idf, axis=1)
        return tf_idf.loc[:, (tf_idf.mean() > tf_idf.mean().quantile(self.quantile))].columns.values

    def fit(self, X, y):
        """
        The method combines other methods and classes to
        clean the corpus and select the vocabulary,
        vectorize the training data per class,
        fit a model per class and calculate priors.
        """
        # Clean corpus and select vocabulary
        vocab = self.preselect_vocabulary(X)
        corpus = self.clean_corpus(X)
        self.vocab = self.tf_idf_filter(corpus, vocab)

        # Vectorize training data per class
        self.classes = np.unique(y)
        train_v = [self.bernoulli_vectorizer(corpus[y == self.classes[i]],
                                             self.vocab)
                   for i in range(len(self.classes))]

        # Fit a model per class
        self.models = []
        for split in train_v:
            model = BernoulliLogLikelihood()
            model.train(split)
            self.models.append(model)
        # Calculate priors
        self.priors = [len(split)/len(X) for split in train_v]

    def predict(self, X):
        """
        The method vectorizes the input data, obtains class predictions
        and returns the most likely labels.
        """
        classifier = BayesClassifier(self.models, self.priors)
        log_prob = classifier.loglikelihood(self.bernoulli_vectorizer(X, self.vocab))
        classes_pred = log_prob.argmax(1)
        return np.array(self.classes[classes_pred])

    def score(self, X, y):
        """
        The method calculates predictions and compares them to the true labels.
        It returns an accuracy score.
        """
        return np.mean(self.predict(X) == y)


if __name__ == '__main__':
    import time
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--traindataset', action="store", dest='train_data',
                        required=True, default='ift3395-6390-arxiv/train.csv',
                        help='Training dataset')
    parser.add_argument('-d', '--testdata', action="store", dest='test_data',
                        required=True, default='ift3395-6390-arxiv/test.csv',
                        help='Test data')
    parser.add_argument('-l', '--testlabels', action="store", dest='test_labels',
                        required=False, default='predictions.csv',
                        help='Test labels')
    parser.add_argument('-o', '--out', action="store_true",
                        dest="output")
    args = parser.parse_args()

    # Load data
    train_df = pd.read_csv(args.train_data, sep=',')
    train_data = np.array([a[0] for a in train_df.values[:, 1:-1]])
    train_labels = train_df.values[:, -1]
    test_data = pd.read_csv(args.test_data, sep=',').values[:, -1]

    # Initialize and fit a classifier
    cls = Classifier()
    cls.fit(train_data, train_labels)
    if args.output:
        pd.DataFrame(cls.predict(test_data),
                     columns=["Category"]).to_csv("predictions.csv",
                                                  index_label="Id")
    else:
        test_labels = pd.read_csv(args.test_labels, sep=',').values[:, -1]
        print(f"Accuracy: {cls.score(test_data, test_labels):.4f}")
        print(f"Time: {(time.time() - start_time):.2f} sec")
