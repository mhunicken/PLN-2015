import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from featureforge.vectorizer import Vectorizer

from .features import History, word_lower, word_istitle, word_isupper, \
    word_isdigit, prev_tags, PrevWord


class MEMM(object):

    def __init__(self, n, tagged_sents):
        """
        n -- order of the model.
        tagged_sents -- list of sentences, each one being a list of pairs.
        """
        self._n = n
        vect = Vectorizer([
            word_lower, word_istitle, word_isupper, word_isdigit,
            prev_tags, PrevWord(word_lower)
        ])
        vect.fit(self.sents_histories(tagged_sents))
        self.classifier = Pipeline([
            ('vect', vect),
            ('clf', LogisticRegression())
        ])
        self.classifier.fit(
            list(self.sents_histories(tagged_sents)),
            list(self.sents_tags(tagged_sents)))

        self._vocab = {w for sent in tagged_sents for _, w in sent}

    def sents_histories(self, tagged_sents):
        """
        Iterator over the histories of a corpus.

        tagged_sents -- the corpus (a list of sentences)
        """
        return itertools.chain(
            *(self.sent_histories(sent) for sent in tagged_sents))

    def sent_histories(self, tagged_sent):
        """
        Iterator over the histories of a tagged sentence.

        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        sent, tags = zip(*tagged_sent)
        sent = list(sent)
        tags = tuple(tags)
        return (
            History(
                sent,
                ('<s>',)*max(self._n-1-i, 0) + tags[max(i+1-self._n, 0):i],
                i
            ) for i in range(len(sent)))

    def sents_tags(self, tagged_sents):
        """
        Iterator over the tags of a corpus.

        tagged_sents -- the corpus (a list of sentences)
        """
        return itertools.chain(
            *(self.sent_tags(sent) for sent in tagged_sents))

    def sent_tags(self, tagged_sent):
        """
        Iterator over the tags of a tagged sentence.

        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        return (tag for _, tag in tagged_sent)

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        result = []
        prev_tags = ('<s>',) * (self._n - 1)
        for i, word in enumerate(sent):
            tag = self.tag_history(History(sent, prev_tags, i))
            prev_tags = (prev_tags + (tag,))[1:]
            result.append(tag)
        return result

    def tag_history(self, h):
        """Tag a history.

        h -- the history.
        """
        return self.classifier.predict([h])[0]

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return w not in self._vocab
