# Copyright 2020-2022 Paul Lu
# As much as possible, only test new functionality of Task III (e.g., unit test)
#       N-fold cross validation:  additional methods for class TrainingSet
#
import sys
import copy     # for deepcopy()

# What we might do to test your code
from ooclassifier import *

Debug = False   # Sometimes, print for debugging

def main():
    tset = TrainingSet()        # Using **NEW** functionality
    run1 = ClassifyByTarget()   # Existing class from base

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
    print("Task III")
    print("--------------------------------------------")
    pfeatures = tset.get_env_variable("pos-features")
    print("pos-features: ", pfeatures)
    plabel = tset.get_env_variable("pos-label")
    print("pos-label: ", plabel)
    print("--------------------------------------------")

    # Do the classification per environment variables of input
    run1.set_target_words(pfeatures.strip().split())  # FIXME Could be cleaner
    #     FIXME We could make set_target_words() polymorphic
    run1.classify_all(tset)
    run1.print_config()
    run1.eval_training_set(tset, plabel)

    # *********************************************
    # *** Look here *** for Task III**
    # *********************************************
    ts_3 = tset.return_nfolds(2)

    print("********************************************")
    run1.classify_all(ts_3[0])
    run1.print_config()
    run1.eval_training_set(ts_3[0], plabel)

    # Hmm, the following code is a candidate for being put into a method
    tp, fp, tn, fn = run1.get_TF()
    precision = float(tp) / float(tp + fp)
    recall = float(tp) / float(tp + fn)
    accuracy = float(tp + tn) / float(tp + tn + fp + fn)
    print("Accuracy:  %3.2g = " % accuracy, end='')
    print("(%d + %d) / (%d + %d + %d + %d)" % (tp, tn, tp, tn, fp, fn))
    print("Precision: %3.2g = " % precision, end='')
    print("%d / (%d + %d)" % (tp, tp, fp))
    print("Recall:    %3.2g = " % recall, end='')
    print("%d / (%d + %d)" % (tp, tp, fn))

    print("********************************************")
    run1.classify_all(ts_3[1])
    run1.print_config()
    run1.eval_training_set(ts_3[1], plabel)

    # Hmm, the following code is a candidate for being put into a method
    tp, fp, tn, fn = run1.get_TF()
    precision = float(tp) / float(tp + fp)
    recall = float(tp) / float(tp + fn)
    accuracy = float(tp + tn) / float(tp + tn + fp + fn)
    print("Accuracy:  %3.2g = " % accuracy, end='')
    print("(%d + %d) / (%d + %d + %d + %d)" % (tp, tn, tp, tn, fp, fn))
    print("Precision: %3.2g = " % precision, end='')
    print("%d / (%d + %d)" % (tp, tp, fp))
    print("Recall:    %3.2g = " % recall, end='')
    print("%d / (%d + %d)" % (tp, tp, fn))
    return


if __name__ == "__main__":
    main()
