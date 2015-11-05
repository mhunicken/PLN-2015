from collections import defaultdict
from nltk.tree import Tree


class CKYParser:

    def __init__(self, grammar):
        """
        grammar -- a binarised NLTK PCFG.
        """
        assert grammar.is_chomsky_normal_form()
        self.grammar = grammar
        self._pi = None
        self._bp = None
        self.prhs = defaultdict(list)
        prods = self.grammar.productions()
        for prod in prods:
            self.prhs[tuple(str(t) for t in prod.rhs())].append(prod)

    def parse(self, sent):
        """Parse a sequence of terminals.

        sent -- the sequence of terminals.
        """
        self._pi = pi = {}
        self._bp = bp = {}
        prhs = self.prhs

        # initialize pi, bp
        for i in range(1, len(sent) + 1):
            for j in range(i, len(sent) + 1):
                pi[(i, j)] = {}
                bp[(i, j)] = {}

        # base case
        for i, w in enumerate(sent):
            for prod in prhs[(w,)]:
                nt = prod.lhs().symbol()
                pi[(i+1, i+1)][nt] = prod.logprob()
                bp[(i+1, i+1)][nt] = Tree(nt, [w])

        for l in range(2, len(sent) + 1):  # length of sub-sentence
            for i in range(1, len(sent) - l + 2):  # start position
                j = i + l - 1  # end position
                for k in range(i, j):  # split position
                    for rh0 in pi[(i, k)]:
                        for rh1 in pi[(k+1, j)]:
                            log_prob_rec = pi[(i, k)][rh0] + pi[(k+1, j)][rh1]
                            for prod in prhs[(rh0, rh1)]:
                                nt = prod.lhs().symbol()
                                log_prob = log_prob_rec + prod.logprob()
                                if nt not in pi[(i, j)] \
                                        or pi[(i, j)][nt] < log_prob:
                                    pi[(i, j)][nt] = log_prob
                                    bp[(i, j)][nt] = \
                                        Tree(nt,
                                             [bp[(i, k)][rh0],
                                              bp[(k+1, j)][rh1]])

        start = self.grammar.start().symbol()
        if start in pi[(1, len(sent))]:
            return pi[(1, len(sent))][start], bp[(1, len(sent))][start]
        else:
            return float('-inf'), None
