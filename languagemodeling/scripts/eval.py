"""Evaulate a language model using the test set.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Language model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from nltk.corpus import PlaintextCorpusReader

import os.path
import sys
# Add ../../ to PYTHONPATH
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        os.pardir, os.pardir))

if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the model
    filename = opts['-i']
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    # load the test set
    corpus = PlaintextCorpusReader(
        'corpus_wikipedia',
        'spanishText_480000_485000'
    )
    sents = corpus.sents()

    avg_logp = model.avg_log_probability(sents)
    perplexity = model.perplexity(sents, avg_logp)
    cross_entropy = model.cross_entropy(sents)

    print('Average log probability: %s' % avg_logp)
    print('Perplexity: %s' % perplexity)
    print('Cross entropy: %s' % cross_entropy)
