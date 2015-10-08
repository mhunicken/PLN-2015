from collections import defaultdict
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
        if prev_tags in self._trans and tag in self._trans[prev_tags]:
            return self._trans[prev_tags][tag]
        return 0

    def out_prob(self, word, tag):
        """Probability of a word given a tag.

        word -- the word.
        tag -- the tag.
        """
        if word in self._out[tag]:
            return self._out[tag][word]
        return 0

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


class ViterbiTagger(object):

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
        tagset = self.hmm.tagset()
        self._pi = {}
        self._pi[0] = {(SENT_START,)*(self.hmm.n-1): (log(1., 2), [])}
        sent = tuple(sent)

        for i, word in enumerate(sent):
            self._pi[i+1] = next_pi = {}
            for prev_tags, (prev_prob, all_tags) in self._pi[i].items():
                for tag in tagset:
                    tags = (prev_tags + (tag,))[1:]
                    trans_prob = self.hmm.trans_prob(tag, prev_tags)
                    out_prob = self.hmm.out_prob(word, tag)
                    if min(trans_prob, out_prob) < 1e-12:
                        continue
                    prob = prev_prob + \
                        log(trans_prob, 2) + \
                        log(out_prob, 2)
                    if tags not in next_pi or prob > next_pi[tags][0]:
                        next_pi[tags] = (prob, all_tags + [tag])

        best_prob = -float('inf')
        best_tagging = None
        for _, (prob, tagging) in self._pi[len(sent)].items():
            if best_tagging is None or prob > best_prob:
                best_tagging = tagging

        return best_tagging


class MLHMM(HMM):

    def __init__(self, n, tagged_sents, addone=True):
        """
        n -- order of the model.
        tagged_sents -- training sentences, each one being a list of pairs.
        addone -- whether to use addone smoothing (default: True).
        """
        self.addone = addone
        self._vocab = set()
        self._tag_counts = defaultdict(int)
        self._out_counts = defaultdict(dict)
        self._single_tag_count = defaultdict(int)

        for sent in tagged_sents:
            prev_tags = (SENT_START,) * (n-1)
            sent = sent + [(SENT_END, SENT_END)]
            for word, tag in sent:
                self._tag_counts[prev_tags] += 1
                prev_tags += (tag,)
                self._tag_counts[prev_tags] += 1
                prev_tags = prev_tags[1:]
                self._single_tag_count[tag] += 1
                self._vocab.add(word)
                if word not in self._out_counts[tag]:
                    self._out_counts[tag][word] = 0
                self._out_counts[tag][word] += 1

        tagset = self._single_tag_count.keys()
        super(MLHMM, self).__init__(n, tagset, None, None)

    def tcount(self, tokens):
        """Count for an k-gram for k <= n.

        tokens -- the k-gram tuple.
        """
        tokens = tuple(tokens)
        if tokens not in self._tag_counts:
            return 0
        return self._tag_counts[tokens]

    def out_count(self, tag, word):
        """Count for word tagged with tag
        """
        if word not in self._out_counts[tag]:
            return 0
        return self._out_counts[tag][word]

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return w not in self._vocab

    def trans_prob(self, tag, prev_tags):
        """Probability of a tag.

        tag -- the tag.
        prev_tags -- tuple with the previous n-1 tags (optional only if n = 1).
        """
        prev_tags = tuple(prev_tags)
        tags = prev_tags + (tag,)
        if self.addone:
            return float(self.tcount(tags) + 1) / \
                (self.tcount(prev_tags) + len(self._vocab))
        else:
            return float(self.tcount(tags)) / self.tcount(prev_tags)

    def out_prob(self, word, tag):
        """Probability of a word given a tag.

        word -- the word.
        tag -- the tag.
        """
        if self.addone:
            return float(self.out_count(tag, word) + 1) / \
                (self._single_tag_count[tag] + len(self._out_counts[tag]))
        else:
            return float(self.out_count(tag, word)) / \
                self._single_tag_count[tag]
