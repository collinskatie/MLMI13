from Analysis import Evaluation
from Analysis import Evaluation

class SentimentLexicon(Evaluation):
    def __init__(self):
        """
        read in lexicon database and store in self.lexicon
        """
        # if multiple entries take last entry by default
        self.lexicon = self.get_lexicon_dict()

    def get_lexicon_dict(self):
        lexicon_dict = {}
        with open('data/sent_lexicon', 'r') as f:
            for line in f:
                word = line.split()[2].split("=")[1]
                polarity = line.split()[5].split("=")[1]
                magnitude = line.split()[0].split("=")[1]
                lexicon_dict[word] = [magnitude, polarity]
        return lexicon_dict

    def classify(self,reviews,threshold,magnitude):
        """
        classify movie reviews using self.lexicon.
        self.lexicon is a dictionary of word: [polarity_info, magnitude_info], e.g. "bad": ["negative","strongsubj"].
        explore data/sent_lexicon to get a better understanding of the sentiment lexicon.
        store the predictions in self.predictions as a list of strings where "+" and "-" are correct/incorrect classifications respectively e.g. ["+","-","+",...]

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @param threshold: threshold to center decisions on. instead of using 0, there may be a bias in the reviews themselves which could be accounted for.
                          experiment for good threshold values.
        @type threshold: integer

        @type magnitude: use magnitude information from self.lexicon?
        @param magnitude: boolean
        """
        # reset predictions
        self.predictions=[]
        # TODO Q0

        threshold = 0.2

        # loop over all reviews in coprus
        for review in reviews:
            true_class, review_data = review
            # score words (ignore POS for now)
            n_positive = 0 # count num positive
            n_total = 0 # only count the number that are in lexicon
            for word, _ in review_data:
                # convert word to lower case as keys are case sensitive
                word = word.lower()
                # skip if not in lexicon (todo: is there a better way to handle missing words??)
                if word not in self.lexicon: continue
                else:
                    word_info = self.lexicon[word]
                    # get whether pos or neg
                    n_positive += 1 if word_info[1] == "positive" else 0
                    n_total += 1
            # get a normalized "positivity" score
            if n_total > 0:
                positivity = n_positive / n_total # divide by num words
            else: positivity = 0 # e.g., no data - avoid divide by zero
            if positivity > threshold: pred_class = "POS"
            else: pred_class = "NEG"

            # check if matches true or not
            if pred_class == true_class: self.predictions.append("+")
            else: self.predictions.append("-")

        return self.predictions
