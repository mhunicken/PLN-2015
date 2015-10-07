from collections import defaultdict


class BaselineTagger:

    def __init__(self, tagged_sents):
        """
        tagged_sents -- training sentences, each one being a list of pairs.
        """
        token_tag_count = {}
        tag_count = defaultdict(int)
        for sent in tagged_sents:
            for token, tag in sent:
                if token not in token_tag_count:
                    token_tag_count[token] = defaultdict(int)
                token_tag_count[token][tag] += 1
                tag_count[tag] += 1

        self.most_frequent_token_tag = {}
        for token, tags in token_tag_count.items():
            self.most_frequent_token_tag[token] = \
                sorted(tags.items(), key=lambda x: -x[1])[0][0]
        self.most_frequent_tag = \
            sorted(tag_count.items(), key=lambda x: -x[1])[0][0]

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        return [self.tag_word(w) for w in sent]

    def tag_word(self, w):
        """Tag a word.

        w -- the word.
        """
        if self.unknown(w):
            return self.most_frequent_tag
        return self.most_frequent_token_tag[w]

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return w not in self.most_frequent_token_tag
