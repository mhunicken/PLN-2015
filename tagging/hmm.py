from math import log

SENT_START = '<s>'
SENT_END = '</s>'


class HMM(object):

    def __init__(self, n, tagset, trans, out):
        """
        n -- n-gram size.
        tagset -- set of tags.
        trans -- transition probabilities dictionary.
        out -- output probabilities dictionary.
        """
        self.n = n
        self._tagset = tagset
        self._trans = trans
        self._out = out

    def tagset(self):
        """Returns the set of tags.
        """
        return self._tagset

    def trans_prob(self, tag, prev_tags):
        """Probability of a tag.

        tag -- the tag.
        prev_tags -- tuple with the previous n-1 tags (optional only if n = 1).
        """
        return self._trans[prev_tags][tag]

    def out_prob(self, word, tag):
        """Probability of a word given a tag.

        word -- the word.
        tag -- the tag.
        """
        return self._out[tag][word]

    def tag_prob(self, y):
        """
        Probability of a tagging.
        Warning: subject to underflow problems.

        y -- tagging.
        """
        result = 1.
        prev_tags = (SENT_START,) * (self.n-1)
        for tag in y:
            result *= self.trans_prob(tag, prev_tags)
            prev_tags = (prev_tags + (tag,))[1:]
        result *= self.trans_prob('</s>', prev_tags)
        return result

    def prob(self, x, y):
        """
        Joint probability of a sentence and its tagging.
        Warning: subject to underflow problems.

        x -- sentence.
        y -- tagging.
        """
        result = self.tag_prob(y)
        for word, tag in zip(x, y):
            result *= self.out_prob(word, tag)
        return result

    def tag_log_prob(self, y):
        """
        Log-probability of a tagging.

        y -- tagging.
        """
        result = 0.
        prev_tags = (SENT_START,) * (self.n-1)
        for tag in y:
            result += log(self.trans_prob(tag, prev_tags), 2)
            prev_tags = (prev_tags + (tag,))[1:]
        result += log(self.trans_prob(SENT_END, prev_tags), 2)
        return result

    def log_prob(self, x, y):
        """
        Joint log-probability of a sentence and its tagging.

        x -- sentence.
        y -- tagging.
        """
        result = self.tag_log_prob(y)
        for word, tag in zip(x, y):
            result += log(self.out_prob(word, tag), 2)
        return result

    def tag(self, sent):
        """Returns the most probable tagging for a sentence.

        sent -- the sentence.
        """
        return ViterbiTagger(self).tag(sent)


class ViterbiTagger:

    def __init__(self, hmm):
        """
        hmm -- the HMM.
        """
        self.hmm = hmm
        self._pi = {}

    def tag(self, sent):
        """Returns the most probable tagging for a sentence.

        sent -- the sentence.
        """
        trans = self.hmm._trans
        self._pi = {}
        self._pi[0] = {(SENT_START,)*(self.hmm.n-1): (log(1., 2), [])}
        sent = tuple(sent)

        for i, word in enumerate(sent):
            self._pi[i+1] = next_pi = {}
            for prev_tags, (prev_prob, all_tags) in self._pi[i].items():
                for tag in trans[prev_tags]:
                    tags = (prev_tags + (tag,))[1:]
                    prob = prev_prob + \
                        log(trans[prev_tags][tag], 2) + \
                        log(self.hmm.out_prob(word, tag), 2)
                    if tags not in next_pi or prob > next_pi[tags][0]:
                        next_pi[tags] = (prob, all_tags + [tag])

        best_prob = -float('inf')
        best_tagging = None
        for _, (prob, tagging) in self._pi[len(sent)].items():
            if best_tagging is None or prob > best_prob:
                best_tagging = tagging

        return best_tagging
