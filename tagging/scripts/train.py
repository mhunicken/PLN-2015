"""Train a sequence tagger.

Usage:
  train.py [-m <model>] [-n <n>] [-c <classifier>] -o <file>
  train.py -h | --help

Options:
  -m <model>        Model to use [default: base]:
                        base: Baseline
                        ml: Maximum Likelihood Hidden Markov Model
                        me: Maximum Entropy Markov Model
  -n <n>            Size of the ngram
  -c <classifier>   Classifier for maximum entropy markov model [default: lr]:
                        lr: Logistic Regression
                        nb: Multinomial Naive Bayes
                        svm: Support Vector Machine with linear kernel
  -o <file>         Output model file.
  -h --help         Show this screen.
"""
from docopt import docopt


if __name__ == '__main__':
    opts = docopt(__doc__)

    import pickle

    from corpus.ancora import SimpleAncoraCorpusReader
    from tagging.baseline import BaselineTagger
    from tagging.hmm import MLHMM
    from tagging.memm import MEMM

    models = {
        'base': BaselineTagger,
        'ml': MLHMM,
        'me': MEMM,
    }

    # load the data
    files = 'CESS-CAST-(A|AA|P)/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/', files)
    sents = corpus.tagged_sents()

    # train the model
    if opts['-m'] == 'base':
        model = models[opts['-m']](sents)
    else:
        model = models[opts['-m']](int(opts['-n']), sents, opts['-c'])

    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
