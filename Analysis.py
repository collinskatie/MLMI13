import math,sys
import numpy as np

class Evaluation():
    """
    general evaluation class implemented by classifiers
    """
    def crossValidate(self,corpus):
        """
        function to perform 10-fold cross-validation for a classifier.
        each classifier will be inheriting from the evaluation class so you will have access
        to the classifier's train and test functions.

        1. read reviews from corpus.folds and store 9 folds in train_files and 1 in test_files
        2. pass data to self.train and self.test e.g., self.train(train_files)
        3. repeat for another 9 runs making sure to test on a different fold each time

        @param corpus: corpus of movie reviews
        @type corpus: MovieReviewCorpus object
        """
        # reset predictions
        self.predictions=[]
        self.preds_per_fold= []
        # TODO Q3
        num_folds = len(set(corpus.folds))
        # todo ask question: are the predictions storing the avg??
        for fold_i in range(num_folds):
            # hold out the i-th fold data
            test_files = corpus.folds[fold_i]
            train_files = np.array([np.array(corpus.folds[fold_j]) for fold_j in range(num_folds) if fold_j != fold_i])
            train_files = np.reshape(train_files, [train_files.shape[0]*train_files.shape[1], train_files.shape[-1]])
            self.train(train_files)
            preds = self.test(test_files, overwrite=False)
            self.preds_per_fold.append(preds)
            self.predictions.extend(preds)

    def getStdDeviation(self):
        """
        get standard deviation across folds in cross-validation.
        """
        # get the avg accuracy and initialize square deviations
        avgAccuracy,square_deviations=self.getAccuracy(),0
        # find the number of instances in each fold
        fold_size=len(self.predictions)//10
        # calculate the sum of the square deviations from mean
        for fold in range(0,len(self.predictions),fold_size):
            square_deviations+=(self.predictions[fold:fold+fold_size].count("+")/float(fold_size) - avgAccuracy)**2
        # std deviation is the square root of the variance (mean of square deviations)
        return math.sqrt(square_deviations/10.0)

    def getAccuracy(self, cross_val_preds=False):
        """
        get accuracy of classifier.

        @return: float containing percentage correct
        """
        
        if cross_val_preds: 
            # score per fold
            self.score_per_fold = []
           
            for preds in self.preds_per_fold: 
                self.score_per_fold.append(preds.count("+")/float(len(preds)))
        
        # note: data set is balanced so just taking number of correctly classified over total
        # "+" = correctly classified and "-" = error
        return self.predictions.count("+")/float(len(self.predictions))
