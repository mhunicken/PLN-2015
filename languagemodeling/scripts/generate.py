"""Generate natural language sentences using a language model.

Usage:
  generate.py -i <file> -n <n>
  generate.py -h | --help

Options:
  -i <file>     Language model file.
  -n <n>        Number of sentences to generate.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

import os.path
import sys
# Add ../../ to PYTHONPATH
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        os.pardir, os.pardir))

from languagemodeling.ngram import NGramGenerator


if __name__ == '__main__':
    opts = docopt(__doc__)
    sys.stderr.write('Generate\n')
    # load the model
    filename = opts['-i']
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    sys.stderr.write('Loaded model\n')
    # generate
    n = int(opts['-n'])
    generator = NGramGenerator(model)
    sys.stderr.write('Initialized generator\n')
    for i in range(n):
        print('Sentence %s:' % i)
        print(' '.join(generator.generate_sent()))
