"""Print corpus statistics.

Usage:
  stats.py
  stats.py -h | --help

Options:
  -h --help     Show this screen.
"""
from docopt import docopt

from corpus.ancora import SimpleAncoraCorpusReader


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/')
    sents = list(corpus.tagged_sents())

    # compute the statistics
    print('Number of sents: %s' % len(sents))
    nwords = sum(map(len, sents))
    print('Number of occurrences of words: %s' % nwords)

    tags = {}
    words = {}

    for sent in sents:
        for word, tag in sent:
            if word not in words:
                words[word] = {}
            if tag not in words[word]:
                words[word][tag] = 0
            if tag not in tags:
                tags[tag] = {}
            if word not in tags[tag]:
                tags[tag][word] = 0
            words[word][tag] += 1
            tags[tag][word] += 1

    print('Number of words (vocabulary): %s' % len(words))
    print('Number of tags (tags vocabulary): %s' % len(tags))

    most_frequent_tags = sorted(
        tags.keys(),
        key=lambda k: -sum(f for _, f in tags[k].items())
    )[:10]

    print('\nMost frequent tags:')
    for tag in most_frequent_tags:
        print('Tag: %s' % tag)
        freq = sum(f for _, f in tags[tag].items())
        print('\tFrequency: %s (%.2f%%)' % (freq, freq*100. / nwords))
        most_frequent_words = sorted(
            tags[tag].keys(),
            key=lambda k: -tags[tag][k]
        )[:5]
        print('\tMost frequent words for tag:  %s'
              % ' '.join(most_frequent_words))

    ambig = {i: [] for i in range(1, 10)}
    for word, tags in words.items():
        if len(tags) in ambig:
            ambig[len(tags)].append(
                (word, sum(f for _, f in words[word].items())))

    print('\nLevels of ambiguity:')
    for lev, words_lev in ambig.items():
        print('Level %s:' % lev)
        print('\tNumber of words: %s (%.2f%%)'
              % (len(words_lev), len(words_lev)*100. / len(words)))
        if len(words_lev):
            most_frequent_words = sorted(words_lev, key=lambda k: -k[1])[:5]
            most_frequent_words = next(zip(*most_frequent_words))
            print('\tMost frequent words:  %s' % ' '.join(most_frequent_words))
