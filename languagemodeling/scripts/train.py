"""Train an n-gram model.

Usage:
  train.py -n <n> [-m <model>] -o <file>
  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -m <model>    Model to use [default: ngram]:
                  ngram: Unsmoothed n-grams.
                  addone: N-grams with add-one smoothing.
                  interpolated: N-grams with linear interpolation
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from nltk.corpus import PlaintextCorpusReader, gutenberg

import os.path
import sys
# Add ../../ to PYTHONPATH
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        os.pardir, os.pardir))

from languagemodeling.ngram import NGram, AddOneNGram, InterpolatedNGram


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    #corpus = PlaintextCorpusReader(
    #    'gutenberg',
    #    '.*'
    #)
    sents = gutenberg.sents()

    # train the model
    n = int(opts['-n'])
    model_type = opts['-m'] or 'ngram'
    if model_type == 'ngram':
        model = NGram(n, sents)
    elif model_type == 'addone':
        model = AddOneNGram(n, sents)
    elif model_type == 'interpolated':
        model = InterpolatedNGram(n, sents)
    else:
        print('Invalid model type')
        exit(1)
    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f, protocol=2)
    f.close()
