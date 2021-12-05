import numpy, os
import numpy as np
from subprocess import call
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn import svm
from Classifiers import SVMText
# from nltk.tokenize import word_tokenize
# import nltk
# nltk.download('punkt')

class SVMDoc2Vec(SVMText):
    """
    class for baseline extension using SVM with Doc2Vec pre-trained vectors
    """
    def __init__(self,model,bigrams,trigrams,discard_closed_class, normalize_vecs=False):
        """
        initialisation of SVMDoc2Vec classifier.
        @param model: pre-trained doc2vec model to use
        @type model: string (e.g. random_model.model)
        """
        SVMText.__init__(self, bigrams,trigrams,discard_closed_class)
        self.svm_classifier = svm.SVC()
        self.predictions = []
        self.model = model
        self.normalize_vecs = normalize_vecs # added parameter to play with :) 
        self.preds_per_fold = []
        self.score_per_fold= []
        
        self.pred_y = []
        self.true_y = []

    def normalize(self,vector):
        """
        normalise vector between -1 and 1 inclusive.
        @param vector: vector inferred from doc2vec
        @type vector: numpy array
        @return: normalised vector
        """
        # TODO Q8
        # help from: https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
        min_val = -1 
        max_val = 1 
        norm_vec = ((vector - min_val)/(max_val - min_val)) * (max_val - min_val) + min_val
        return norm_vec


    # since using pre-trained vectors don't need to determine features
    def getFeatures(self,reviews):
        """
        infer document vector for each review and add it to the list of features.
        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """

        # TODO Q8
        self.labels = [label for label, _ in reviews]
        all_review_tokens = np.empty(len(reviews), dtype=object) # ensures that we can save POS tag info as tuples
        for idx,(_, review) in enumerate(reviews): 
            all_review_tokens[idx] = self.extractReviewTokens(review)
        # convert between review tokens to doc features
        doc_vecs = [self.model.infer_vector(review_tokens) for review_tokens in all_review_tokens]
        # todo: play with whether we normalize or not!! 
        if self.normalize_vecs: 
            doc_vecs = [self.normalize(vector) for vec in doc_vecs]
        self.input_features = doc_vecs
        return doc_vecs
        
     # override test to use extracted embeddings as features
    def test(self,reviews, overwrite=True):
        """
        test svm
        @param reviews: test data
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        # TODO Q6.1
        # get test features using pre-loaded featurizer (e.g., vocab already extracted by fitting) 
        true_labels = [label for label, _ in reviews]
        test_features = self.getFeatures(reviews)
        
        pred_y = list(self.svm_classifier.predict(test_features))
        
        self.pred_y = pred_y
        self.true_y = true_labels
        
        preds = []
        for pred, true in zip(pred_y, true_labels):  
            if pred == true: preds.append("+")
            else: preds.append("-")

        if overwrite: self.predictions = preds
        else: return preds
        
class DocFeaturizer(): 
    """
    class for housing operations on the Doc2Vec featurizer (custom written for ease of i/o)
    """
    def __init__(self, dim_features=50, window=2,dm=1, dbow_words=0, dm_concat=0):
        # initialize model
        self.model = Doc2Vec(vector_size=dim_features, window=window,dm=dm,dbow_words=dbow_words, dm_concat=dm_concat) 
        
        
    def train_model(self, docs, epochs=10): 
        # help from: https://www.tutorialspoint.com/gensim/gensim_doc2vec_model.html
        # and: https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html
        
        # wrap in tagged object 
        # help from: https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5
        docs = [doc for sentiment_label, doc in docs]
        tagged_data = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]
        
        # first, extract the vocab
        self.model.build_vocab(tagged_data)
        self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=epochs)
   
    def infer_vector(self, doc_tokens): 
        # note: same name as main class to allow this object to be used w/ other classes w/o modification 
        return self.model.infer_vector(doc_tokens)
    
    def get_embeddings(self, docs, normalize=False):
        # extract embeddings for set of documents
        docs = [doc for sentiment_label, doc in docs]
        embeddings = [self.infer_vector(tokens) for tokens in docs]
        return embeddings
    
        