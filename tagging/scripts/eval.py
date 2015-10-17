"""Evaulate a tagger.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Tagging model file.
  -h --help     Show this screen.
"""
from docopt import docopt
from collections import defaultdict
import pickle
import sys

from corpus.ancora import SimpleAncoraCorpusReader


def progress(msg, width=None):
    """Ouput the progress of something on the same line."""
    if not width:
        width = len(msg)
    sys.stderr.write('\b' * width + msg)
    sys.stderr.flush()


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    # load the data
    files = '3LB-CAST/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/', files)
    sents = list(corpus.tagged_sents())

    # tag
    hits, total = 0, 0
    hits_known, total_known = 0, 0
    conf_mat = defaultdict(int)
    conf_mat_rows = set()
    conf_mat_cols = set()
    n = len(sents)
    for i, sent in enumerate(sents):
        word_sent, gold_tag_sent = zip(*sent)

        model_tag_sent = model.tag(word_sent)
        assert len(model_tag_sent) == len(gold_tag_sent), i

        conf_sent = [
            (g, m)
            for m, g in zip(model_tag_sent, gold_tag_sent)
            if m != g
        ]
        for g, m in conf_sent:
            conf_mat_rows.add(g)
            conf_mat_cols.add(m)
            conf_mat[(g, m)] += 1

        # global score
        hits += len(sent) - len(conf_sent)
        total += len(sent)
        acc = float(hits) / total

        pred_known = [
            (m, g)
            for m, g, w in zip(model_tag_sent, gold_tag_sent, word_sent)
            if not model.unknown(w)
        ]
        hits_known += sum(m == g for m, g in pred_known)
        total_known += len(pred_known)

        progress('{:3.1f}% ({:2.2f}%)'.format(float(i) * 100 / n, acc * 100))

    acc = float(hits) / total

    print('')
    print('Accuracy: {:2.2f}%'.format(acc * 100))
    acc_known = hits_known*100. / total_known
    print('Accuracy (known words): {:2.2f}%'.format(acc_known))
    acc_unknown = (hits-hits_known)*100. / (total-total_known)
    print('Accuracy (unknown words): {:2.2f}%'.format(acc_unknown))

    conf_mat_rows = list(conf_mat_rows)
    conf_mat_cols = list(conf_mat_cols)
    print('Confussion matrix:')
    print('\t' + '\t'.join(conf_mat_cols))
    for t1 in conf_mat_rows:
        print(t1 + '\t' + '\t'.join(
            '%.2f' % (conf_mat[(t1, t2)]*100 / (total-hits))
            if conf_mat[(t1, t2)] else '-'
            for t2 in conf_mat_cols
        ))
