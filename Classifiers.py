import os
from subprocess import call
from nltk.util import ngrams
from Analysis import Evaluation
import numpy as np
from sklearn import svm
# tokenization help from: https://stackoverflow.com/questions/46965524/create-sparse-word-matrix-in-python-bag-of-words
from sklearn.feature_extraction import DictVectorizer
from collections import Counter, OrderedDict

class NaiveBayesText(Evaluation):
    def __init__(self,smoothing,bigrams,trigrams,discard_closed_class):
        """
        initialisation of NaiveBayesText classifier.

        @param smoothing: use smoothing?
        @type smoothing: booleanp

        @param bigrams: add bigrams?
        @type bigrams: boolean

        @param trigrams: add trigrams?
        @type trigrams: boolean

        @param discard_closed_class: restrict unigrams to nouns, adjectives, adverbs and verbs?
        @type discard_closed_class: boolean
        """
        # set of features for classifier
        self.vocabulary=set()
        # prior probability
        self.prior={}
        # conditional probablility
        self.condProb={}
        # use smoothing?
        self.smoothing=smoothing
        # add bigrams?
        self.bigrams=bigrams
        # add trigrams?
        self.trigrams=trigrams
        # restrict unigrams to nouns, adjectives, adverbs and verbs?
        self.discard_closed_class=discard_closed_class
        # stored predictions from test instances
        self.predictions=[]

    def extractVocabulary(self,reviews):
        """
        extract features from training data and store in self.vocabulary.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        for sentiment,review in reviews:
            for token in self.extractReviewTokens(review):
                self.vocabulary.add(token)

    def extractReviewTokens(self,review):
        """
        extract tokens from reviews.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @return: list of strings
        """
        text=[]
        for token in review:
            # check if pos tags are included in review e.g. ("bad","JJ")
            if len(token)==2 and self.discard_closed_class:
                if token[1][0:2] in ["NN","JJ","RB","VB"]: text.append(token)
            else:
                text.append(token)
        if self.bigrams:
            for bigram in ngrams(review,2): text.append(bigram)
        if self.trigrams:
            for trigram in ngrams(review,3): text.append(trigram)
        return text

    def create_vocab_dict(self):
        vocab_to_id = {}
        for word in self.vocabulary:
            vocab_to_id[word] = len(vocab_to_id) # TODO: ask if this is a typo
        return vocab_to_id

    def get_cond_prob(self, token, sentiment, token_freqs, n_words_sentiment, smoother=1):
        # NOTE: written for ease of abstraction + modification
        # get the conditional probability for a given token and sentiment class
        if token in token_freqs[sentiment]:
            f_w = token_freqs[sentiment][token] # counts
            if self.smoothing:
                f_w += smoother
            # print("fw: ", f_w, " smoothing? ", self.smoothing)
            self.condProb[token][sentiment] = f_w/n_words_sentiment[sentiment]
        else:
            # token never occured (note, redundant w/ dict definition)
            # avoids strange log zero issues
            # print("NEVER OCCURED!!")
            if self.smoothing:
                self.condProb[token][sentiment] = smoother/n_words_sentiment[sentiment]
            else: self.condProb[token][sentiment] = 0
            #np.finfo(float).eps # NOTE: this is bad b/c means that we can NEVER place any probability on this token

    def train(self,reviews):
        """
        train NaiveBayesText classifier.

        1. reset self.vocabulary, self.prior and self.condProb
        2. extract vocabulary (i.e. get features for training)
        3. get prior and conditional probability for each label ("POS","NEG") and store in self.prior and self.condProb
           note: to get conditional concatenate all text from reviews in each class and calculate token frequencies
                 to speed this up simply do one run of the movie reviews and count token frequencies if the token is in the vocabulary,
                 then iterate the vocabulary and calculate conditional probability (i.e. don't read the movie reviews in their entirety
                 each time you need to calculate a probability for each token in the vocabulary)

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        # TODO Q1

        # reset
        self.vocabulary = set()
        self.prior = {}
        self.condProb = {}

        # extract vocab
        self.extractVocabulary(reviews)

        # get prior by counting number of docs of a particular class
        tot_num_docs = len(reviews)
        n_pos = np.sum([1 for sentiment, _ in reviews if sentiment == "POS"])
        n_neg = tot_num_docs - n_pos # b/c two poss classes
        self.prior = {"POS": n_pos/tot_num_docs, "NEG": n_neg/tot_num_docs}

        # get conditional probabilities
        # first count token frequencies per class by looping over reviews once
        token_freqs = {"POS": {}, "NEG": {}}
        for sentiment, review in reviews:
            tokens = self.extractReviewTokens(review)
            for token in tokens:
                if token in token_freqs[sentiment]:
                    token_freqs[sentiment][token] += 1
                else:
                    if token in self.vocabulary: # only add if token in vocab
                        token_freqs[sentiment][token] = 1

        # get total number of words in each class (for cond prob denominator)
        n_words_sentiment = {}
        sentiment_classes = ["POS","NEG"]
        for sent in sentiment_classes:
            n_words_sentiment[sent] = np.sum([token_freqs[sent][word] for word in token_freqs[sent]])

        # now get conditional probs by iterating over vocab
        # set empty for all tokens to start
        self.condProb = {token: {"POS": 0, "NEG": 0} for token in self.vocabulary}

        laplace_smoother = 1 # laplace smoothing of a constant value
        if self.smoothing:
            # add the num tot words in the vocab to each n_words
            n_tot_words = len(self.vocabulary)#np.sum([n_words_sentiment[sent] for sent in sentiment_classes])
            for sent in sentiment_classes:
                n_words_sentiment[sent] += (n_tot_words * laplace_smoother)

        # for token in self.vocabulary:
        #     # get cond prob for both class
        #     for sent in sentiment_classes:
        #         self.get_cond_prob(token, sentiment, token_freqs, n_words_sentiment, laplace_smoother)

        for token in self.vocabulary:
            # get cond prob for both class
            for sent in sentiment_classes:
                if token in token_freqs[sent]:
                    if self.smoothing:
                        self.condProb[token][sent] = (token_freqs[sent][token] + laplace_smoother)/n_words_sentiment[sent]
                    else:
                        self.condProb[token][sent] = token_freqs[sent][token]/n_words_sentiment[sent]
                else:
                    # token never occured (note, redundant w/ dict definition)
                    if self.smoothing:
                        self.condProb[token][sent] = laplace_smoother/n_words_sentiment[sent]
                    else:
                        self.condProb[token][sent] = 0 # NOTE: this is bad b/c means that we can NEVER place any probability on this token

        # TODO Q2 (use switch for smoothing from self.smoothing)

    def test(self,reviews, overwrite=True):
        """
        test NaiveBayesText classifier and store predictions in self.predictions.
        self.predictions should contain a "+" if prediction was correct and "-" otherwise.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        # TODO Q1
        preds = []
        sentiment_classes = ["POS", "NEG"]
        for true_sentiment, review in reviews:
            # get predicted sentiment
            # first, extract tokens
            tokens = self.extractReviewTokens(review)
            # next, get each class' prob
            sentiment_probs = {}
            for sent in sentiment_classes:
                # log of the prior prob
                prior_prob = np.log(self.prior[sent])
                # summed log likelihood per word
                cond_prob = 0
                # todo: possibly set = to 100% to avoid skewing towards negative
                for token in tokens:
                    if token in self.vocabulary: # note: might not be needed
                        if self.condProb[token][sent] == 0: cond_prob += 0
                        else: cond_prob += np.log(self.condProb[token][sent])
                sentiment_probs[sent] = prior_prob + cond_prob
            # finally, take argmax over classes as pred
            # help from: https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
            # note: if both zero, could randomly return to avoid always choosing one class
            pred_sent = max(sentiment_probs, key=lambda k: sentiment_probs[k])
            if pred_sent == true_sentiment:
                preds.append("+")
            else: preds.append("-")
        if overwrite: self.predictions = preds
        else: return preds # for later updating of predictions array

class SVMText(Evaluation):
    def __init__(self,bigrams,trigrams,discard_closed_class):
        """
        initialisation of SVMText object
        @param bigrams: add bigrams?
        @type bigrams: boolean
        @param trigrams: add trigrams?
        @type trigrams: boolean
        @param svmlight_dir: location of smvlight binaries
        @type svmlight_dir: string
        @param svmlight_dir: location of smvlight binaries
        @type svmlight_dir: string
        @param discard_closed_class: restrict unigrams to nouns, adjectives, adverbs and verbs?
        @type discard_closed_class: boolean
        """
        self.svm_classifier = svm.SVC()
        self.predictions=[]
        self.vocabulary=set()
        # add in bigrams?
        self.bigrams=bigrams
        # add in trigrams?
        self.trigrams=trigrams
        # restrict to nouns, adjectives, adverbs and verbs?
        self.discard_closed_class=discard_closed_class
        
        # maintain a tokenizer
        self.v = DictVectorizer()

    def extractVocabulary(self,reviews):
        self.vocabulary = set()
        for sentiment, review in reviews:
            for token in self.extractReviewTokens(review):
                 self.vocabulary.add(token)

    def extractReviewTokens(self,review):
        """
        extract tokens from reviews.
        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        @return: list of strings
        """
        text=[]
        for term in review:
            # check if pos tags are included in review e.g. ("bad","JJ")
            if len(term)==2 and self.discard_closed_class:
                if term[1][0:2] in ["NN","JJ","RB","VB"]: text.append(term)
            else:
                text.append(term)
        if self.bigrams:
            for bigram in ngrams(review,2): text.append(term)
        if self.trigrams:
            for trigram in ngrams(review,3): text.append(term)
        return text

    def getFeatureVec(self, review, num_features):
        """
        Custom function to get feature vector for a given review
        """
        feats = np.zeros(num_features)
        tokens = self.extractReviewTokens(review)
        for idx, feature in enumerate(self.vocabulary):
            feats[idx] = 1#tokens.count(feature)
        return feats

    def getFeatures(self,reviews):
        """
        determine features and labels from training reviews.
        1. extract vocabulary (i.e. get features for training)
        2. extract features for each review as well as saving the sentiment
        3. append each feature to self.input_features and each label to self.labels
        (self.input_features will then be a list of list, where the inner list is
        the features)
        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """

        self.input_features = []
        self.labels = []
        
        # label is just first elmt per reviews
        self.labels = [label for label, _ in reviews]
        # extract tokens per review
#         review_tokens = [np.array(self.extractReviewTokens(review)) for _,review in reviews]
        review_tokens = np.empty(len(reviews), dtype=object) # ensures that we can save POS tag info as tuples
        for idx,(_, review) in enumerate(reviews): 
            review_tokens[idx] = self.extractReviewTokens(review)
        
        # get vocab and get sparse vectors
        # help from: https://stackoverflow.com/questions/46965524/create-sparse-word-matrix-in-python-bag-of-words
        sparse_features = self.v.fit_transform(Counter(f) for f in np.array(review_tokens))
        self.input_features = sparse_features # size = (num reviews, vocab size)
        
        return sparse_features # counts of occurances  


    def train(self,reviews):
        """
        train svm. This uses the sklearn SVM module, and further details can be found using
        the sci-kit docs. You can try changing the SVM parameters.
        @param reviews: training data
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        # function to determine features in training set.
        self.getFeatures(reviews)

        # reset SVM classifier and train SVM model
        self.svm_classifier = svm.SVC()
        self.svm_classifier.fit(self.input_features, self.labels)

    def test(self,reviews, overwrite=True):
        """
        test svm
        @param reviews: test data
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        # TODO Q6.1
        # get test features using pre-loaded featurizer (e.g., vocab already extracted by fitting) 
        true_labels = [label for label, _ in reviews]
        review_tokens = np.empty(len(reviews), dtype=object) # ensures that we can save POS tag info as tuples
        for idx,(_, review) in enumerate(reviews): 
            review_tokens[idx] = self.extractReviewTokens(review)
        test_features = self.v.transform(Counter(f) for f in np.array(review_tokens))
        
        pred_y = list(self.svm_classifier.predict(test_features))
        
        preds = []
        for pred, true in zip(pred_y, true_labels):  
            if pred == true: preds.append("+")
            else: preds.append("-")

        if overwrite: self.predictions = preds
        else: return preds
