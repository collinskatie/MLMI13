import os, codecs, sys
from nltk.stem.porter import PorterStemmer
import numpy as np

class MovieReviewCorpus():
    def __init__(self,stemming,pos,use_imdb=False):
        """
        initialisation of movie review corpus.

        @param stemming: use porter's stemming?
        @type stemming: boolean

        @param pos: use pos tagging?
        @type pos: boolean
        """
        # raw movie reviews
        self.reviews=[]
        # held-out train/test set
        self.train=[]
        self.test=[]
        # folds for cross-validation
        self.folds={} # round robin splitting
        self.folds_conseq = {} # consecutive splitting
        # porter stemmer
        self.stemming = stemming
        self.stemmer=PorterStemmer() if stemming else None
        # part-of-speech tags
        self.pos=pos
        
        # which dataset to pull from (added flag)
        self.use_imdb = use_imdb
        if use_imdb: 
            self.data_dir = f"data/aclImdb/"
        else: 
            self.data_dir = f"data/reviews/"
        
        
        # import movie reviews
        self.get_reviews(self.data_dir)

    def get_reviews(self, data_dir = f"data/reviews/"):
        """
        processing of movie reviews.

        1. parse reviews in data/reviews and store in self.reviews.

           the format expected for reviews is: [(string,list), ...] e.g. [("POS",["a","good","movie"]), ("NEG",["a","bad","movie"])].
           in data/reviews there are .tag and .txt files. The .txt files contain the raw reviews and .tag files contain tokenized and pos-tagged reviews.

           to save effort, we recommend you use the .tag files. you can disregard the pos tags to begin with and include them later.
           when storing the pos tags, please use the format for each review: ("POS/NEG", [(token, pos-tag), ...]) e.g. [("POS",[("a","DT"), ("good","JJ"), ...])]

           to use the stemmer the command is: self.stemmer.stem(token)

        2. store training and held-out reviews in self.train/test. files beginning with cv9 go in self.test and others in self.train

        3. store reviews in self.folds. self.folds is a dictionary with the format: self.folds[fold_number] where fold_number is an int 0-9.
           you can get the fold number from the review file name.
        """
        
        # maintain lists that we want info from
        train_info = []
        test_info = []
        cv_info = {} # round robin
        cv_info_conseq = {}
        
        # convert a single review into (token, pos-tag format)
        def get_single_review_wPOS(fpth):
            # read in data from a single file (each file = a single review)
            # help reading in data from: https://www.pythontutorial.net/python-basics/python-read-text-file/
            with open(fpth) as f:
                full_review_data = f.readlines()
                full_review_data = [l.strip() for l in full_review_data] # remove trailing new line

            # token separatred by "\t" from POS
            # todo (for future): could optionally ignore punctuation!!
            parsed_review_data = []
            for token_data in full_review_data:
                if "\t" not in token_data: continue
                token, pos_tag = token_data.split("\t")
#                 token = token.lower() # OPTIONAL!!!! discuss!!
                if self.stemming: 
                    token = self.stemmer.stem(token)
                if self.pos:
                    data_obj = (token, pos_tag)
                else:
                    data_obj = token#(token, pos_tag)
                parsed_review_data.append(data_obj)#(token, pos_tag))

            return parsed_review_data
       
        # convert a single review into just token list
        def get_single_review(fpth):
            # read in data from a single file (each file = a single review)
            # help reading in data from: https://www.pythontutorial.net/python-basics/python-read-text-file/
            with open(fpth) as f:
                # NOTE: assumes we want to split by space -- discuss!!
                # e.g., "kick the bucket" diff meaning than individ words, but sep by spaces
                full_review_data = f.readlines()[0].split(" ") 

            # todo (for future): could optionally ignore punctuation!!
            parsed_review_data = []
            for token in full_review_data:
                token = token.lower() # OPTIONAL!!!! discuss!!
                if self.stemming: 
                    token = self.stemmer.stem(token)
                data_obj = token#(token, pos_tag)
                parsed_review_data.append(data_obj)#(token, pos_tag))

            return parsed_review_data
        
        if not self.use_imdb:

            # define sentiment classes (note: could be changed in the future)
            sentiment_classes = ["POS", "NEG"]

            for sent_class in sentiment_classes:
                sent_dir = f"{data_dir}{sent_class}/" # "sent" = "sentiment"
                all_reviews = [rev for rev in os.listdir(sent_dir) if rev[-4:] == ".tag"]

                # process each review and put in associated train/test based on file number
                # (also determines fold)
                for review_idx, review_file_name in enumerate(all_reviews):

                    fold_num = int(review_file_name[3]) # all start w/ cv
                    parsed_review_data = get_single_review_wPOS(f"{sent_dir}{review_file_name}")
                    review_metadata = [sent_class, parsed_review_data]

                    if fold_num == 9:
                        test_info.append(review_metadata)
                    else:
                        train_info.append(review_metadata)

                    # TODO: update w/ round-robin splitting
                    fold_num = review_idx % 10 # b/c mod-10
                    if fold_num not in cv_info:
                        cv_info[fold_num] = [review_metadata]
                    else:
                        cv_info[fold_num].append(review_metadata)


                    # CONSECUTIVE SPLITTING
                    if fold_num not in cv_info_conseq: cv_info_conseq[fold_num] = [review_metadata]
                    else: cv_info_conseq[fold_num].append(review_metadata)
        else: 
            # parse reviews from imbd data format
            supra_folders = ["train", "test"]
            sub_folders = ["pos", "neg", "unsup"]
            for split in supra_folders: # corresponds to train/test main split
                for sent_class in sub_folders: # corresponds to labeled sentiment
                    sent_dir = f"{data_dir}{split}/{sent_class}/" 
                    all_reviews = [rev for rev in os.listdir(sent_dir) if rev[-4:] == ".txt"]   
                    # process each review and put in associated train/test based on file number
                    # (also determines fold)
                    for review_idx, review_file_name in enumerate(all_reviews):

                        orig_fold_num = int(review_file_name.split("_")[1].split(".")[0]) # include split at end
                        parsed_review_data = get_single_review(f"{sent_dir}{review_file_name}")
                        sent_class = sent_class.upper() # for consistency with other dataset -- use all caps w/ labels
                        review_metadata = [sent_class, parsed_review_data]

                        if split == "test":
                            test_info.append(review_metadata)
                        else:
                            train_info.append(review_metadata)

                        # round robin
                        fold_num = review_idx % 10 # b/c mod-10
                        if fold_num not in cv_info:
                            cv_info[fold_num] = [review_metadata]
                        else:
                            cv_info[fold_num].append(review_metadata)


                        # CONSECUTIVE SPLITTING
                        if orig_fold_num not in cv_info_conseq: cv_info_conseq[orig_fold_num] = [review_metadata]
                        else: cv_info_conseq[orig_fold_num].append(review_metadata)

        # set class attributes using curated metadata
        self.train = train_info
        self.test = test_info
        self.folds = cv_info
        self.folds_conseq = cv_info_conseq

        print(f"num train: {len(self.train)}, num test: {len(self.test)}")

        # all review data is the combined train and test
        self.reviews = train_info + test_info #list(np.concat(self.train, self.test))

        print(f"tot num reviews: {len(self.reviews)}")
