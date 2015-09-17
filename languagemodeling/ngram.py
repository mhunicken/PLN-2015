from collections import defaultdict
from .constants import SENT_START, SENT_END
from math import log

import random


def vocab_size_from_sents(sents):
    vocab = set()
    for sent in sents:
        for token in sent:
            vocab.add(token)
    return len(vocab) + 1  # Count </s>


class NGram(object):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self.n = n
        self.counts = defaultdict(int)
        self.candidates_next = defaultdict(set)

        for sent in sents:
            sent = [SENT_START] * (n-1) + sent + [SENT_END]
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                self.counts[ngram] += 1
                self.counts[ngram[:-1]] += 1
                self.candidates_next[ngram[:-1]].add(ngram[-1])

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.

        tokens -- the n-gram or (n-1)-gram tuple.
        """
        return self.counts[tokens]

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        prev_tokens = prev_tokens or ()
        prev_tokens = tuple(prev_tokens)
        assert len(prev_tokens) == self.n - 1
        tokens = prev_tokens + (token,)
        return float(self.count(tokens)) / self.count(prev_tokens)

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        sent += [SENT_END]
        result = 1.
        prev_tokens = (SENT_START,) * (self.n-1)

        for token in sent:
            result *= self.cond_prob(token, prev_tokens)
            if result < 1e-12:
                return 0.
            prev_tokens = (prev_tokens + (token,))[1:]

        return result

    def sent_log_prob(self, sent, base=2.):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        sent += [SENT_END]
        result = 0.
        prev_tokens = (SENT_START,) * (self.n - 1)

        try:
            for token in sent:
                result += log(self.cond_prob(token, prev_tokens), base)
                prev_tokens = (prev_tokens + (token,))[1:]
        except ValueError:
            return float("-inf")

        return result

    def avg_log_probability(self, sents, base=2.):
        result = 0
        for sent in sents:
            result += self.sent_log_prob(sent, base)
        return result / len(sents)

    def perplexity(self, sents, avg_logp=None):
        if avg_logp is None:
            avg_logp = self.avg_log_probability(sents, 2.)
        return 2**(-avg_logp)

    def cross_entropy(self, sents):
        return 0.  # TODO


#    def generate_token(self, prev_tokens=None):
#        """Randomly generate a token, given prev_tokens.

#        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
#        """
#        prev_tokens = prev_tokens or ()
#        prev_tokens = tuple(prev_tokens)
#        assert len(prev_tokens) == self.n - 1

#        p = random.random()
#        sp = 0.

#        for token in self.candidates_next[prev_tokens]:
#            sp += self.cond_prob(token, prev_tokens)
#            if sp > p:
#                return token

#        assert(0)

class NGramGenerator(object):

    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self.model = model
        self.probs = {}
        self.sorted_probs = {}
        for prev_tokens, nexts in self.model.candidates_next.items():
            self.probs[prev_tokens] = {}
            self.sorted_probs[prev_tokens] = []
            for token in nexts:
                prob = self.model.cond_prob(token, prev_tokens)
                self.probs[prev_tokens][token] = prob
                self.sorted_probs[prev_tokens].append((token, prob))
            self.sorted_probs[prev_tokens].sort(key=lambda t: (t[1], t[0]))

    def generate_sent(self):
        """Randomly generate a sentence."""
        result = []
        prev_tokens = (SENT_START,) * (self.model.n - 1)
        new_token = self.generate_token(prev_tokens)

        while new_token != SENT_END:
            result.append(new_token)
            prev_tokens = (prev_tokens + (new_token,))[1:]
            new_token = self.generate_token(prev_tokens)

        return result

    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.

        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        # TODO reconsider this:
        # return self.model.generate_token(prev_tokens)
        prev_tokens = prev_tokens or ()
        prev_tokens = tuple(prev_tokens)
        assert len(prev_tokens) == self.model.n - 1

        target = random.random()
        acum = 0.

        for token, prob in self.sorted_probs[prev_tokens]:
            acum += prob
            if acum > target:
                return token

        assert 0


class AddOneNGram(NGram):
    def __init__(self, n, sents):
        super(AddOneNGram, self).__init__(n, sents)
        self.vocab_size = vocab_size_from_sents(sents)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        prev_tokens = prev_tokens or ()
        prev_tokens = tuple(prev_tokens)
        assert len(prev_tokens) == self.n - 1
        tokens = prev_tokens + (token,)
        return float(self.count(tokens)+1) / (self.count(prev_tokens)+self.V())

    def V(self):
        """Size of the vocabulary.
        """
        return self.vocab_size


class InterpolatedNGram(AddOneNGram):
    def __init__(self, n, sents, gamma=None, addone=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        assert n > 0
        self.n = n
        self.counts = defaultdict(int)

        heldout_set = []

        for i, sent in enumerate(sents):
            if gamma is None and i%10 == 1:
                # held out
                heldout_set.append(sent)
            else:
                # train
                sent = [SENT_START] * (n-1) + sent + [SENT_END]
                for j in range(n+1):
                    for i in range(n-j, len(sent) - j + 1):
                        ngram = tuple(sent[i: i + j])
                        self.counts[ngram] += 1
                for j in range(1, n):
                    self.counts[(SENT_START,)*j] += 1

        self.vocab_size = vocab_size_from_sents(sents)

        self.gamma = gamma
        if self.gamma is None:
            # TODO train gamma
            self.gamma = 10.

        self.addone = addone

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        prev_tokens = prev_tokens or ()
        prev_tokens = tuple(prev_tokens)
        assert len(prev_tokens) == self.n - 1

        lambdas = self._lambdas_from_prev_tokens(prev_tokens)

        probs = [
            float(self.count(prev_tokens[i:]+(token,)))/self.count(prev_tokens[i:])
            if self.count(prev_tokens[i:]) else 0
            for i in range(self.n)
        ]
        if self.addone:
            probs[-1] = (self.count((token,))+1)/(self.count(()) + self.V())
        return sum(l*p for l, p in zip(lambdas, probs))

    def _lambdas_from_prev_tokens(self, prev_tokens):
        lambdas = []
        lambda_sum = 0.
        for i in range(0, self.n-1):
            cnt = self.count(prev_tokens[i:])
            lambdas.append((1-lambda_sum) * cnt / (cnt+self.gamma))
            lambda_sum += lambdas[-1]
        lambdas.append(1-lambda_sum)
        return lambdas


class BackOffNGram(object):
    # TODO
    def __init__(self):
        pass
