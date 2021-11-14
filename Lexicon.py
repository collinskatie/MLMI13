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

        # loop over all reviews in coprus
        for review in reviews:
            true_class, review_data = review
            # score words (ignore Part of Speech for now)
            positivity_score = 0 # b/c higher = more positive
            for obj in review_data:
                if type(obj) == list: word = obj[0]
                else: word = obj
                # convert word to lower case as keys are case sensitive
                # TODO: add ideas about lower!! e.g., names ([death] "will", Will), acronymns, use cases
                word = word.lower() # b/c want to avoid separating the data (more data-efficient for lower)
                # skip if not in lexicon (todo: is there a better way to handle missing words??)
                if word not in self.lexicon: continue
                else:
                    word_info = self.lexicon[word]
                    # geta sentiment score
                    if word_info[1] == "positive":
                        sentiment_val = 1
                    elif word_info[1] == "negative":
                        sentiment_val = -1
                    else: # neutral
                        sentiment_val = 0
                    # possibly scale depending on parameters
                    if not magnitude: # binary
                        # get whether pos or neg
                        positivity_score += sentiment_val
                    else: # weighted
                        # idea: if strong ==> *2
                        # note: multiplier doesn't impact "neutral"
                        #   do we want to change that?? (^)
                        # NOTE: multiplier scale is also a hyperparam!
                        multiplier = 2 if "strong" in word_info[0] else 1
                        positivity_score += multiplier * sentiment_val

            if positivity_score > threshold: pred_class = "POS"
            else: pred_class = "NEG"

            # check if matches true or not
            if pred_class == true_class: self.predictions.append("+")
            else: self.predictions.append("-")

        return self.predictions
