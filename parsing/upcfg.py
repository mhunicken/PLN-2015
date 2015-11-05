import copy
from collections import defaultdict
from nltk.grammar import PCFG, ProbabilisticProduction, Nonterminal
from .baselines import Flat
from .cky_parser import CKYParser
from .util import lexicalize, unlexicalize


class UPCFG:
    """Unlexicalized PCFG.
    """

    def __init__(self, parsed_sents, start='sentence'):
        """
        parsed_sents -- list of training trees.
        """
        prod_count = defaultdict(int)
        nt_count = defaultdict(int)
        for sent in parsed_sents:
            sent = copy.deepcopy(sent)
            unlexicalize(sent)
            sent.chomsky_normal_form()
            sent.collapse_unary(collapsePOS=True, collapseRoot=True)
            for prod in sent.productions():
                prod_count[prod] += 1
                nt_count[prod.lhs()] += 1

        prob_prod = [
            ProbabilisticProduction(
                p.lhs(),
                p.rhs(),
                prob=float(c) / nt_count[p.lhs()]
            )
            for p, c in prod_count.items()
        ]

        self.parser = CKYParser(PCFG(Nonterminal(start), prob_prod))
        self.start = start

    def productions(self):
        """Returns the list of UPCFG probabilistic productions.
        """
        return self.parser.grammar.productions()

    def parse(self, tagged_sent):
        """Parse a tagged sentence.

        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        sent, tags = zip(*tagged_sent)
        _, parsing = self.parser.parse(tags)
        if parsing is None:
            return Flat(None, self.start).parse(tagged_sent)
        return lexicalize(parsing, sent)
