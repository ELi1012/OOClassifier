# Copyright 2020-2022 Paul Lu
import sys
import copy     # for deepcopy()
from string import punctuation      # list of punctuation

Debug = False   # Sometimes, print for debugging.  Overridable on command line.
InputFilename = "file.input.txt"
TargetWords = [
        'outside', 'today', 'weather', 'raining', 'nice', 'rain', 'snow',
        'day', 'winter', 'cold', 'warm', 'snowing', 'out', 'hope', 'boots',
        'sunny', 'windy', 'coming', 'perfect', 'need', 'sun', 'on', 'was',
        '-40', 'jackets', 'wish', 'fog', 'pretty', 'summer'
        ]


def open_file(filename=InputFilename):
    try:
        f = open(filename, "r")
        return(f)
    except FileNotFoundError:
        # FileNotFoundError is subclass of OSError
        if Debug:
            print("File Not Found")
        return(sys.stdin)
    except OSError:
        if Debug:
            print("Other OS Error")
        return(sys.stdin)


def safe_input(f=None, prompt=""):
    try:
        # Case:  Stdin
        if f is sys.stdin or f is None:
            line = input(prompt)
        # Case:  From file
        else:
            assert not (f is None)
            assert (f is not None)
            line = f.readline()
            if Debug:
                print("readline: ", line, end='')
            if line == "":  # Check EOF before strip()
                if Debug:
                    print("EOF")
                return("", False)
        return(line.strip(), True)
    except EOFError:
        return("", False)


class C274:
    def __init__(self):
        self.type = str(self.__class__)
        return

    def __str__(self):
        return(self.type)

    def __repr__(self):
        s = "<%d> %s" % (id(self), self.type)
        return(s)


class ClassifyByTarget(C274):
    def __init__(self, lw=[]):
        super().__init__()      # Call superclass
        # self.type = str(self.__class__)
        self.allWords = 0
        self.theCount = 0
        self.nonTarget = []
        self.set_target_words(lw)
        self.initTF()
        return

    def initTF(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        return

    # FIXME:  Incomplete.  Finish get_TF() and other getters/setters.
    def get_TF(self):
        return(self.TP, self.FP, self.TN, self.FN)

    # TODO: Could use Use Python properties
    #     https://www.python-course.eu/python3_properties.php
    def set_target_words(self, lw):
        # Could also do self.targetWords = lw.copy().  Thanks, TA Jason Cannon
        self.targetWords = copy.deepcopy(lw)
        return

    def get_target_words(self):
        return(self.targetWords)

    def get_allWords(self):
        return(self.allWords)

    def incr_allWords(self):
        self.allWords += 1
        return

    def get_theCount(self):
        return(self.theCount)

    def incr_theCount(self):
        self.theCount += 1
        return

    def get_nonTarget(self):
        return(self.nonTarget)

    def add_nonTarget(self, w):
        self.nonTarget.append(w)
        return

    def print_config(self, printSorted=True):
        print("-------- Print Config --------")
        ln = len(self.get_target_words())
        print("TargetWords (%d): " % ln, end='')
        if printSorted:
            print(sorted(self.get_target_words()))
        else:
            print(self.get_target_words())
        return

    def print_run_info(self, printSorted=True):
        print("-------- Print Run Info --------")
        print("All words:%3s. " % self.get_allWords(), end='')
        print(" Target words:%3s" % self.get_theCount())
        print("Non-Target words (%d): " % len(self.get_nonTarget()), end='')
        if printSorted:
            print(sorted(self.get_nonTarget()))
        else:
            print(self.get_nonTarget())
        return

    def print_confusion_matrix(self, targetLabel, doKey=False, tag=""):
        assert (self.TP + self.TP + self.FP + self.TN) > 0
        print(tag+"-------- Confusion Matrix --------")
        print(tag+"%10s | %13s" % ('Predict', 'Label'))
        print(tag+"-----------+----------------------")
        print(tag+"%10s | %10s %10s" % (' ', targetLabel, 'not'))
        if doKey:
            print(tag+"%10s | %10s %10s" % ('', 'TP   ', 'FP   '))
        print(tag+"%10s | %10d %10d" % (targetLabel, self.TP, self.FP))
        if doKey:
            print(tag+"%10s | %10s %10s" % ('', 'FN   ', 'TN   '))
        print(tag+"%10s | %10d %10d" % ('not', self.FN, self.TN))
        return

    def eval_training_set(self, tset, targetLabel, lines=True):
        print("-------- Evaluate Training Set --------")
        self.initTF()
        # zip is good for parallel arrays and iteration
        z = zip(tset.get_instances(), tset.get_lines())
        for ti, w in z:
            lb = ti.get_label()
            cl = ti.get_class()
            if lb == targetLabel:
                if cl:
                    self.TP += 1
                    outcome = "TP"
                else:
                    self.FN += 1
                    outcome = "FN"
            else:
                if cl:
                    self.FP += 1
                    outcome = "FP"
                else:
                    self.TN += 1
                    outcome = "TN"
            explain = ti.get_explain()
            # Format nice output
            if lines:
                w = ' '.join(w.split())
            else:
                w = ' '.join(ti.get_words())
                w = lb + " " + w

            # TW = testing bag of words words (kinda arbitrary)
            print("TW %s: ( %10s) %s" % (outcome, explain, w))
            if Debug:
                print("-->", ti.get_words())
        self.print_confusion_matrix(targetLabel)
        return

    def classify_by_words(self, ti, update=False, tlabel="last"):
        inClass = False
        evidence = ''
        lw = ti.get_words()
        for w in lw:
            if update:
                self.incr_allWords()
            if w in self.get_target_words():    # FIXME Write predicate
                inClass = True
                if update:
                    self.incr_theCount()
                if evidence == '':
                    evidence = w            # FIXME Use first word, but change
            elif w != '':
                if update and (w not in self.get_nonTarget()):
                    self.add_nonTarget(w)
        if evidence == '':
            evidence = '#negative'
        if update:
            ti.set_class(inClass, tlabel, evidence)
        return(inClass, evidence)

    # Could use a decorator, but not now
    def classify(self, ti, update=False, tlabel="last"):
        cl, e = self.classify_by_words(ti, update, tlabel)
        return(cl, e)

    def classify_all(self, ts, update=True, tlabel="classify_all"):
        for ti in ts.get_instances():
            cl, e = self.classify(ti, update=update, tlabel=tlabel)
        return


class ClassifyByTopN(ClassifyByTarget):
    def __init__(self, lw=[]):
        super().__init__(lw)      # Call superclass

    def target_top_n(self, tset, num=5, label=''):
        # counts all words in all training instances
        # whose label matches arg label
        # of object tset, which is of class TrainingSet
        inst_list = tset.get_instances()        # returns inst.inobjhash (list of training instances)
        word_dict = {}
        new_targetwords = []

        for obj_inst in inst_list:
            #print(str(obj_inst.get_words()) + (obj_inst.get_label()))
            if obj_inst.inst["label"] == label:
                wordlist = obj_inst.inst["words"]
                for w in wordlist:
                    if w in word_dict:
                        # w already exists in the dict
                        # increment its value by 1
                        word_dict[w] += 1
                    else:
                        # w does not exist in dict
                        # add key w to dict
                        word_dict.update({w: 1})
        # get top num words
        # returns a list of tuples from dict
        # sorted by values from greatest to least
        sorted_words = sorted(word_dict.items(),
                              key=lambda item: item[1],
                              reverse=True)
        

        # use counter to account for ties
        # get top (num) most frequent words
        counter = 0
        enough_words = False
        word_num = len(sorted_words)
        if word_num > num:     # number of words is greater than num
            numth_freq = sorted_words[num - 1][1]
        elif word_num > 0:     # all words are included
            numth_freq = sorted_words[-1][1]
            enough_words = True

        debugging = False
        if debugging:
            print(sorted_words)

        # numth_freq gets frequency of numth word on list 
        while not enough_words and counter < word_num:
            if debugging:
                print('counter: '+str(counter))
                
            # check if number of target words is equal or
            # greater than num
            if counter >= num:
                # check if word frequency of current word is equal to
                # frequency of numth word
                if debugging:
                    print('counter greater than num at '+str(counter))
                    print('frequency of current word: '+str(sorted_words[counter][1]))
                if sorted_words[counter][1] != numth_freq:
                    enough_words = True
                    break
            
            

            #print('appended word '+ sorted_words[counter][0])
            new_targetwords.append(sorted_words[counter][0])
            counter += 1
            
            

        self.set_target_words(new_targetwords)


class TrainingInstance(C274):
    def __init__(self):
        super().__init__()              # Call superclass
        # self.type = str(self.__class__)
        self.inst = dict()
        # FIXME:  Get rid of dict, and use attributes
        self.inst["label"] = "N/A"      # Class, given by oracle
        self.inst["words"] = []         # Bag of words
        self.inst["class"] = ""         # Class, by classifier
        self.inst["explain"] = ""       # Explanation for classification
        self.inst["experiments"] = dict()   # Previous classifier runs
        return

    def preprocess_words(self, mode=''):
        # returns a list of preprocessed words
        # apply preprocessing to all words in current training instance object
        input_list = self.inst["words"]

        Stop_Words = [
            "i", "me", "my", "myself", "we", "our",
            "ours", "ourselves", "you", "your",
            "yours", "yourself", "yourselves", "he",
            "him", "his", "himself", "she", "her",
            "hers", "herself", "it", "its", "itself",
            "they", "them", "their", "theirs",
            "themselves", "what", "which", "who",
            "whom", "this", "that", "these", "those",
            "am", "is", "are", "was", "were", "be",
            "been", "being", "have", "has", "had",
            "having", "do", "does", "did", "doing",
            "a", "an", "the", "and", "but", "if",
            "or", "because", "as", "until", "while",
            "of", "at", "by", "for", "with",
            "about", "against", "between", "into",
            "through", "during", "before", "after",
            "above", "below", "to", "from", "up",
            "down", "in", "out", "on", "off", "over",
            "under", "again", "further", "then",
            "once", "here", "there", "when", "where",
            "why", "how", "all", "any", "both",
            "each", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than",
            "too", "very", "s", "t", "can", "will",
            "just", "don", "should", "now"
            ]

        # input comes from argument
        # lowercase each word
        for i in range(len(input_list)):
            input_list[i] = input_list[i].lower()
        output_words = []

        # remove punctuation and symbols
        if mode != 'keep-symbols':
            input_list = ' '.join(input_list)
            for i in punctuation:
                input_list = input_list.replace(i, '')
            input_list = input_list.split()

        nums = '0123456789'
        # remove numbers unless string consists entirely of numbers
        # only add words to output list if word is not a stopword
        for current_word in input_list:
            # word will not be added if word is a stopword
            # and user does not want to keep stopwords
            if current_word not in Stop_Words or mode == 'keep-stops':

                if mode == 'keep-digits':
                    # append the word without checking for/removing numbers
                    output_words.append(current_word)

                else:
                    # check for numbers to remove if word has any letters
                    new_word = current_word
                    found_letter = False

                    # loop through each character
                    for char in current_word:
                        if char in nums:
                            # current character is a number
                            # replace number with empty string
                            new_word = new_word.replace(char, '')
                        else:
                            # character is not a number
                            # (could be letter or punctuation)
                            found_letter = True

                    # check if string has any non-number characters
                    if found_letter is False:
                        # keep the string as is and add to output_words
                        output_words.append(current_word)
                    else:
                        # append string without numbers
                        output_words.append(new_word)

        # return as list of words
        return output_words

    def get_label(self):
        return(self.inst["label"])

    def get_words(self):
        return(self.inst["words"])

    def set_class(self, theClass, tlabel="last", explain=""):
        # tlabel = tag label
        self.inst["class"] = theClass
        self.inst["experiments"][tlabel] = theClass
        self.inst["explain"] = explain
        return

    def get_class_by_tag(self, tlabel):             # tlabel = tag label
        cl = self.inst["experiments"].get(tlabel)
        if cl is None:
            return("N/A")
        else:
            return(cl)

    def get_explain(self):
        cl = self.inst.get("explain")
        if cl is None:
            return("N/A")
        else:
            return(cl)

    def get_class(self):
        return self.inst["class"]

    def process_input_line(
                self, line, run=None,
                tlabel="read", inclLabel=False
            ):
        for w in line.split():
            if w[0] == "#":
                self.inst["label"] = w
                if inclLabel:
                    self.inst["words"].append(w)
            else:
                self.inst["words"].append(w)

        if not (run is None):
            cl, e = run.classify(self, update=True, tlabel=tlabel)
        return(self)


class TrainingSet(C274):
    def __init__(self):
        super().__init__()      # Call superclass
        # self.type = str(self.__class__)
        self.inObjList = []     # Unparsed lines, from training set     STORES LINES AS LIST
        self.inObjHash = []     # Parsed lines, in dictionary/hash      STORES TRAINING INSTANCE OBJECTS AS LIST
        self.variable = dict()  # NEW: Configuration/environment variables
        return

    def return_nfolds(self, num=3):
        # code from TA : Jordan Van Den Bruel
        # returns a list of TrainingSets storing TrainingInstances
        # inside sets
        nfolds = []
        obj_list = self.get_instances()     # list of training instances
        lines = self.get_lines()            # list of lines from text file

        for i in range(num):
            
            tset = TrainingSet()
            tset.variable = copy.deepcopy(self.variable)
            nfolds.append(tset)
            
            for j in range(len(obj_list)):
                if j % num == i:
                    tset.inObjHash.append(copy.deepcopy(obj_list[j]))
                    tset.inObjList.append(copy.deepcopy(lines[j]))
                    #print(str(i)+' '+str(obj_list[j].get_words()))
            
            
        return nfolds

    def copy(self):
        newset = TrainingSet()
        newset.inObjList = copy.deepcopy(self.inObjList)
        newset.inObjHash = copy.deepcopy(self.inObjHash)
        newset.variable = copy.deepcopy(self.variable)
        return newset

    def add_training_set(self, tset):
        # adds all training instances of tset (of class TrainingSet)
        # to an object of class TrainingSet
        # add training instances to self?
        obj_hash = tset.get_instances()
        obj_lines = tset.get_lines()


        # new_tset.inObjList.append(copy.deepcopy(tset.inObjList))
        # new_tset.variable = copy.deepcopy(tset.variable)

        #self.inObjHash = copy.deepcopy(tset.get_instances())
        #self.inObjList = copy.deepcopy(tset.get_lines())
        self.inObjList = copy.deepcopy(tset.inObjList)
        self.inObjHash = copy.deepcopy(tset.inObjHash)
        self.variable = copy.deepcopy(tset.variable)
        

        #for i in range(len(obj_list)):
        #    self.inObjHash.append(copy.deepcopy(obj_list[i]))
            # FIXME might have to copy other attributes as well
        return

    def set_env_variable(self, k, v):
        self.variable[k] = v
        return

    def get_env_variable(self, k):
        if k in self.variable:
            return(self.variable[k])
        else:
            return ""

    def inspect_comment(self, line):
        if len(line) > 1 and line[1] != ' ':      # Might be variable
            v = line.split(maxsplit=1)
            self.set_env_variable(v[0][1:], v[1])
        return

    def get_instances(self):
        return(self.inObjHash)      # FIXME Should protect this more

    def get_lines(self):
        return(self.inObjList)      # FIXME Should protect this more

    def print_training_set(self):
        print("-------- Print Training Set --------")
        z = zip(self.inObjHash, self.inObjList)
        for ti, w in z:
            lb = ti.get_label()
            cl = ti.get_class_by_tag("last")     # Not used
            explain = ti.get_explain()
            print("( %s) (%s) %s" % (lb, explain, w))
            if Debug:
                print("-->", ti.get_words())
        return

    def process_input_stream(self, inFile, run=None):
        assert not (inFile is None), "Assume valid file object"
        cFlag = True
        while cFlag:
            # returns each line.strip() and true if not EOF
            line, cFlag = safe_input(inFile)
            if not cFlag:
                break
            assert cFlag, "Assume valid input hereafter"

            if len(line) == 0:   # Blank line.  Skip it.
                continue

            # Check for comments *and* environment variables
            if line[0] == '%':  # Comments must start with % and variables
                self.inspect_comment(line)
                continue

            # Save the training data input, by line
            self.inObjList.append(line)
            # Save the training data input, after parsing
            ti = TrainingInstance()
            ti.process_input_line(line, run=run)
            self.inObjHash.append(ti)
        return

    def preprocess(self, mode=''):
        # perform preprocessing for all training instances
        # in current training set
        # TrainingInstance.preprocess_words(mode, wordlist)

        for ti in self.get_instances():
            ti.inst["words"] = ti.preprocess_words(mode)


# Very basic test of functionality
def basemain():
    global Debug
    tset = TrainingSet()
    run1 = ClassifyByTarget(TargetWords)
    if Debug:
        print(run1)     # Just to show __str__
        lr = [run1]
        print(lr)       # Just to show __repr__

    argc = len(sys.argv)
    if argc == 1:   # Use stdin, or default filename
        inFile = open_file()
        assert not (inFile is None), "Assume valid file object"
        tset.process_input_stream(inFile, run1)
        inFile.close()
    else:
        for f in sys.argv[1:]:
            # Allow override of Debug from command line
            if f == "Debug":
                Debug = True
                continue
            if f == "NoDebug":
                Debug = False
                continue

            inFile = open_file(f)
            assert not (inFile is None), "Assume valid file object"
            tset.process_input_stream(inFile, run1)
            inFile.close()

    print("--------------------------------------------")
    plabel = tset.get_env_variable("pos-label")
    print("pos-label: ", plabel)
    print("NOTE: Not using any target words from the file itself")
    print("--------------------------------------------")

    if Debug:
        tset.print_training_set()
    run1.print_config()
    run1.print_run_info()
    run1.eval_training_set(tset, plabel)

    return


if __name__ == "__main__":
    basemain()
