# https://docs.python.org/3/library/unittest.html
from unittest import TestCase
from math import log2

from nltk.tree import Tree
from nltk.grammar import PCFG

from parsing.cky_parser import CKYParser


class TestCKYParser(TestCase):

    def test_parse(self):
        grammar = PCFG.fromstring(
            """
                S -> NP VP              [1.0]
                NP -> Det Noun          [0.6]
                NP -> Noun Adj          [0.4]
                VP -> Verb NP           [1.0]
                Det -> 'el'             [1.0]
                Noun -> 'gato'          [0.9]
                Noun -> 'pescado'       [0.1]
                Verb -> 'come'          [1.0]
                Adj -> 'crudo'          [1.0]
            """)

        parser = CKYParser(grammar)

        lp, t = parser.parse('el gato come pescado crudo'.split())

        # check chart
        pi = {
            (1, 1): {'Det': log2(1.0)},
            (2, 2): {'Noun': log2(0.9)},
            (3, 3): {'Verb': log2(1.0)},
            (4, 4): {'Noun': log2(0.1)},
            (5, 5): {'Adj': log2(1.0)},

            (1, 2): {'NP': log2(0.6 * 1.0 * 0.9)},
            (2, 3): {},
            (3, 4): {},
            (4, 5): {'NP': log2(0.4 * 0.1 * 1.0)},

            (1, 3): {},
            (2, 4): {},
            (3, 5): {'VP': log2(1.0) + log2(1.0) + log2(0.4 * 0.1 * 1.0)},

            (1, 4): {},
            (2, 5): {},

            (1, 5): {'S':
                     log2(1.0) +  # rule S -> NP VP
                     log2(0.6 * 1.0 * 0.9) +  # left part
                     log2(1.0) + log2(1.0) +
                     log2(0.4 * 0.1 * 1.0)},  # right part
        }
        self.assertEqualPi(parser._pi, pi)

        # check partial results
        bp = {
            (1, 1): {'Det': Tree.fromstring("(Det el)")},
            (2, 2): {'Noun': Tree.fromstring("(Noun gato)")},
            (3, 3): {'Verb': Tree.fromstring("(Verb come)")},
            (4, 4): {'Noun': Tree.fromstring("(Noun pescado)")},
            (5, 5): {'Adj': Tree.fromstring("(Adj crudo)")},

            (1, 2): {'NP': Tree.fromstring("(NP (Det el) (Noun gato))")},
            (2, 3): {},
            (3, 4): {},
            (4, 5): {'NP': Tree.fromstring("(NP (Noun pescado) (Adj crudo))")},

            (1, 3): {},
            (2, 4): {},
            (3, 5): {'VP': Tree.fromstring(
                "(VP (Verb come) (NP (Noun pescado) (Adj crudo)))")},

            (1, 4): {},
            (2, 5): {},

            (1, 5): {'S': Tree.fromstring(
                """(S
                    (NP (Det el) (Noun gato))
                    (VP (Verb come) (NP (Noun pescado) (Adj crudo)))
                   )
                """)},
        }
        self.assertEqual(parser._bp, bp)

        # check tree
        t2 = Tree.fromstring(
            """
                (S
                    (NP (Det el) (Noun gato))
                    (VP (Verb come) (NP (Noun pescado) (Adj crudo)))
                )
            """)
        self.assertEqual(t, t2)

        # check log probability
        lp2 = log2(1.0 * 0.6 * 1.0 * 0.9 * 1.0 * 1.0 * 0.4 * 0.1 * 1.0)
        self.assertAlmostEqual(lp, lp2)

    def test_parse_ambiguous(self):
        grammar = PCFG.fromstring(
            """
                S -> S ConjS            [0.2]
                S -> Verb Noun          [0.4]
                S -> Verb CN            [0.3]
                S -> 'arroz'            [0.05]
                S -> 'pescado'          [0.05]
                ConjS -> Conj S         [1.0]
                CN -> Noun ConjNoun     [1.0]
                ConjNoun -> Conj Noun   [1.0]
                Verb -> 'come'       [1.0]
                Noun -> 'pescado'       [0.5]
                Noun -> 'arroz'         [0.5]
                Conj -> 'y'             [1.0]
            """)

        parser = CKYParser(grammar)

        lp, t = parser.parse('come pescado y arroz'.split())

        # check chart
        pi = {
            (1, 1): {'Verb': log2(1.0)},
            (2, 2): {'Noun': log2(0.5), 'S': log2(0.05)},
            (3, 3): {'Conj': log2(1.0)},
            (4, 4): {'Noun': log2(0.5), 'S': log2(0.05)},

            (1, 2): {'S': log2(0.4 * 1.0 * 0.5)},
            (2, 3): {},
            (3, 4): {'ConjNoun': log2(1.0 * 1.0 * 0.5),
                     'ConjS': log2(1.0 * 1.0 * 0.05)},

            (1, 3): {},
            (2, 4): {'CN': log2(1.0) + log2(0.5) + log2(1.0 * 0.5 * 1.0),
                     'S': log2(0.2) + log2(0.05) + log2(1.0 * 1.0 * 0.05)},

            (1, 4): {'S':
                     max(
                         # S -> S ConjS
                         log2(0.2) + log2(0.4 * 1.0 * 0.5) +
                         log2(1.0 * 1.0 * 0.05),
                         # S -> Verb CN (greater)
                         log2(0.3) + log2(1.0) +
                         log2(1.0 * 0.5 * 1.0 * 0.5 * 1.0)
                     )},
        }
        self.assertEqualPi(parser._pi, pi)

        # best parsing:
        t2 = Tree.fromstring("""
            (S
                (Verb come)
                (CN
                    (Noun pescado)
                    (ConjNoun (Conj y) (Noun arroz))
                )
            )
        """)

        # check partial results
        bp = {
            (1, 1): {'Verb': Tree.fromstring("(Verb come)")},
            (2, 2): {'Noun': Tree.fromstring("(Noun pescado)"),
                     'S': Tree.fromstring("(S pescado)")},
            (3, 3): {'Conj': Tree.fromstring("(Conj y)")},
            (4, 4): {'Noun': Tree.fromstring("(Noun arroz)"),
                     'S': Tree.fromstring("(S arroz)")},

            (1, 2): {'S': Tree.fromstring("(S (Verb come) (Noun pescado))")},
            (2, 3): {},
            (3, 4): {'ConjNoun':
                     Tree.fromstring("(ConjNoun (Conj y) (Noun arroz))"),
                     'ConjS': Tree.fromstring("(ConjS (Conj y) (S arroz))")},

            (1, 3): {},
            (2, 4): {'CN':
                     Tree.fromstring("""
                         (CN (Noun pescado) (ConjNoun (Conj y) (Noun arroz)))
                     """),
                     'S':
                     Tree.fromstring("""
                        (S (S pescado) (ConjS (Conj y) (S arroz)))
                     """)},

            (1, 4): {'S': t2}
        }
        self.assertEqual(parser._bp, bp)

        # check tree
        self.assertEqual(t, t2)

        # check log probability
        lp2 = log2(0.3 * 1.0 * 1.0 * 0.5 * 1.0 * 0.5 * 1.0)
        self.assertAlmostEqual(lp, lp2)

    def assertEqualPi(self, pi1, pi2):
        self.assertEqual(set(pi1.keys()), set(pi2.keys()))

        for k in pi1.keys():
            d1, d2 = pi1[k], pi2[k]
            self.assertEqual(d1.keys(), d2.keys(), k)
            for k2 in d1.keys():
                prob1 = d1[k2]
                prob2 = d2[k2]
                self.assertAlmostEqual(prob1, prob2)
