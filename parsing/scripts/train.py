"""Train a parser.

Usage:
  train.py [-m <model>] [--hm=<n>] -o <file>
  train.py -h | --help

Options:
  -m <model>    Model to use [default: flat]:
                  flat: Flat trees
                  rbranch: Right branching trees
                  lbranch: Left branching trees
                  upcfg: Unlexicalized probabilistic CFG
  --hm=<n>      Horizontal Markov of order n (for upcfg)
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from corpus.ancora import SimpleAncoraCorpusReader

from parsing.baselines import Flat, RBranch, LBranch
from parsing.upcfg import UPCFG

models = {
    'flat': Flat,
    'rbranch': RBranch,
    'lbranch': LBranch,
    'upcfg': UPCFG
}


if __name__ == '__main__':
    opts = docopt(__doc__)

    print('Loading corpus...')
    files = 'CESS-CAST-(A|AA|P)/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/', files)

    print('Training model...')
    model_name = opts['-m']
    if model_name == 'upcfg':
        hm = None
        if opts['--hm']:
            hm = int(opts['--hm'])
        model = UPCFG(corpus.parsed_sents(), horzMarkov=hm)
    else:
        model = models[model_name](corpus.parsed_sents())
    print('Saving...')
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
