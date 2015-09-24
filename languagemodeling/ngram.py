from collections import defaultdict
from math import log

import random

SENT_START = '<s>'
SENT_END = '</s>'


class NGram(object):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self.n = n
        self.counts = defaultdict(int)

        for sent in sents:
            sent = [SENT_START] * (n-1) + sent + [SENT_END]
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                self.counts[ngram] += 1
                self.counts[ngram[:-1]] += 1

        self.set_vocab_size(sents)

    def set_vocab_size(self, sents):
        vocab = set()
        for sent in sents:
            for token in sent:
                vocab.add(token)
        self.vocab_size = len(vocab) + 1  # Count </s>

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.

        tokens -- the n-gram or (n-1)-gram tuple.
        """
        result = self.counts[tokens]
        if not result:
            del self.counts[tokens]
        return result

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
        sent = sent + [SENT_END]
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
        sent = sent + [SENT_END]
        result = 0.
        prev_tokens = (SENT_START,) * (self.n - 1)

        try:
            for token in sent:
                result += log(self.cond_prob(token, prev_tokens), base)
                prev_tokens = (prev_tokens + (token,))[1:]
        except ValueError:  # log(0)
            return float("-inf")

        return result

    def log_probability(self, sents, base=2.):
        """Total log probability of a test set
        """
        return sum(self.sent_log_prob(sent, base) for sent in sents)

    def cross_entropy(self, sents, base=2.):
        """Cross entropy of a test set
        """
        return -self.log_probability(sents, base) \
            / sum(len(sent) + 1 for sent in sents)

    def perplexity(self, sents, base=2.):
        """Perplexity of a test set
        """
        return base ** self.cross_entropy(sents, base)

    def logp_entropy_perplexity(self, sents, base=2.):
        logp = self.log_probability(sents, base)
        crosse = -logp / sum(len(sent) + 1 for sent in sents)
        perp = base ** crosse
        return logp, crosse, perp

    def V(self):
        """Size of the vocabulary.
        """
        return self.vocab_size


class NGramGenerator(object):

    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self.model = model
        self.probs = defaultdict(dict)
        self.sorted_probs = defaultdict(list)
        for tokens in self.model.counts:
            if len(tokens) < self.model.n:
                continue
            prev_tokens = tokens[:-1]
            token = tokens[-1]
            prob = self.model.cond_prob(token, prev_tokens)
            self.probs[prev_tokens][token] = prob
            self.sorted_probs[prev_tokens].append((token, prob))

        for prev_tokens in self.sorted_probs:
            self.sorted_probs[prev_tokens].sort(key=lambda t: (-t[1], t[0]))

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


class InterpolatedNGram(NGram):

    GAMMA_CANDIDATES = [1.5 ** x for x in range(-5, 30)]

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
        self.set_vocab_size(sents)
        heldout_set = []

        for i, sent in enumerate(sents):
            if gamma is None and i % 10 == 1:
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

        self.addone = addone

        self.gamma = gamma
        if self.gamma is None:
            best_gamma = None
            best_perp = float("inf")
            for gamma in InterpolatedNGram.GAMMA_CANDIDATES:
                self.gamma = gamma
                perp = self.perplexity(heldout_set)
                if best_gamma is None or perp < best_perp:
                    best_perp = perp
                    best_gamma = gamma
            self.gamma = best_gamma
        print(self.gamma)

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
            float(self.count(prev_tokens[i:]+(token,)))
            / self.count(prev_tokens[i:])
            if self.count(prev_tokens[i:]) else 0
            for i in range(self.n)
        ]
        if self.addone:
            probs[-1] = (self.count((token,))+1)/(self.count(()) + self.V())
        return sum(l*p for l, p in zip(lambdas, probs))

    def _lambdas_from_prev_tokens(self, prev_tokens):
        """Lambdas to be used as interpolation weights
        """
        lambdas = []
        lambda_sum = 0.
        for i in range(0, self.n-1):
            cnt = self.count(prev_tokens[i:])
            lambdas.append((1-lambda_sum) * cnt / (cnt+self.gamma))
            lambda_sum += lambdas[-1]
        lambdas.append(1-lambda_sum)
        return lambdas


class BackOffNGram(NGram):

    BETA_CANDIDATES = [0.05 * x for x in range(21)]

    def __init__(self, n, sents, beta=None, addone=True):
        """
            Back-off NGram model with discounting
            as described by Michael Collins.

            n -- order of the model.
            sents -- list of sentences, each one being a list of tokens.
            beta -- discounting hyper-parameter (if not given, estimate using
            held-out data).
            addone -- whether to use addone smoothing (default: True).
        """
        assert n > 0
        self.n = n
        self.counts = defaultdict(int)
        self.set_vocab_size(sents)
        heldout_set = []

        for i, sent in enumerate(sents):
            if beta is None and i % 10 == 1:
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

        # |A(x1..xi)|
        self.card_a = defaultdict(int)
        # sum(c(x2..xix) for x in A(x1..xi))
        self.sum_c = defaultdict(int)

        for tokens in self.counts:
            if len(tokens):
                if tokens[-1] == SENT_START:
                    continue
                prev_tokens = tokens[:-1]
                self.card_a[prev_tokens] += 1
                self.sum_c[prev_tokens] += self.count(tokens[1:])

        self.addone = addone

        self.beta = beta
        if self.beta is None:
            best_beta = None
            best_perp = float("inf")
            for beta in BackOffNGram.BETA_CANDIDATES:
                self.beta = beta
                perp = self.perplexity(heldout_set)
                if best_beta is None or perp < best_perp:
                    best_perp = perp
                    best_beta = beta
            self.beta = best_beta

    def A(self, tokens):
        """Set of words with counts > 0 for a k-gram with 0 < k < n.

        tokens -- the k-gram tuple.
        """
        # Inefficient, but used only in tests
        result = set()
        for tokens_suc in self.counts:
            if tokens_suc[:-1] == tokens and len(tokens_suc):
                result.add(tokens_suc[-1])
        return result

    def alpha(self, tokens):
        """Missing probability mass for a k-gram with 0 < k < n.

        tokens -- the k-gram tuple.
        """
        if not self.count(tokens):
            return 1.
        return self.beta * self.card_a[tokens] / self.count(tokens)

    def denom(self, tokens):
        """Normalization factor for a k-gram with 0 < k < n.

        tokens -- the k-gram tuple.
        """
        if not self.count(tokens):
            return 1.
        assert len(tokens)
        if len(tokens) == 1:
            # In unigram level, do not discount
            if self.addone:
                return 1 - float(self.sum_c[tokens]+self.card_a[tokens]) / \
                    (self.count(tokens[1:]) + self.V())
            else:
                return 1 - float(self.sum_c[tokens])/self.count(tokens[1:])
        else:
            return 1 - (self.sum_c[tokens]-self.beta*self.card_a[tokens]) / \
                self.count(tokens[1:])

    def cond_prob(self, token, prev_tokens=None):
        prev_tokens = prev_tokens or ()
        prev_tokens = tuple(prev_tokens)
        tokens = prev_tokens + (token,)

        if not len(prev_tokens):
            # unigram
            if not self.addone:
                return float(self.count(tokens)) / self.count(prev_tokens)
            return float(self.count(tokens)+1) / \
                (self.count(prev_tokens) + self.V())

        if self.count(tokens):
            return (self.count(tokens)-self.beta)/self.count(prev_tokens)
        elif self.beta > 0.:
            return self.alpha(prev_tokens) * \
                self.cond_prob(token, prev_tokens[1:]) / \
                self.denom(prev_tokens)
        else:
            # If beta is 0, there is no residual probability
            return 0.
