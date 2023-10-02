"""
Word Swap by OpenHowNet
-------------------------------
"""


import pickle

from textattack.shared import utils

from .word_swap import WordSwap


class WordSwapPreposition(WordSwap):
    """Transforms an input by replacing its words with prepositions."""

    def __init__(self, max_candidates=-1, **kwargs):
        super().__init__(**kwargs)
        self.max_candidates = max_candidates
        self.pos_dict = {"ADJ": "adj", "NOUN": "noun", "ADV": "adv", "VERB": "verb"}


    def _get_replacement_words(self, word, word_pos):
        word_pos = self.pos_dict.get(word_pos, None)
        if word_pos is None:
            return []
        else:
            return ['at','on','in','before','after','by','until','till','for','during','through','from','since','within','to','with','of','over','up','out','down','as','between','among','without','about','under','below','above','across','past','into','onto','outside','off','except','besides']
    #['of','for','to','from','on','as','at','by','up','down','under','among','into','out','through','across','besides','within','without','before','until']
    def _get_transformations(self, current_text, indices_to_modify):
        # words = current_text.words
        transformed_texts = []
        for i in indices_to_modify:
            word_to_replace = current_text.words[i]
            word_to_replace_pos = current_text.pos_of_word_index(i)
            replacement_words = self._get_replacement_words(
                word_to_replace, word_to_replace_pos
            )
            for r in replacement_words:
                transformed_texts.append(current_text.replace_word_at_index(i, r))
        return transformed_texts


    def extra_repr_keys(self):
        return ["max_candidates"]
    

def recover_word_case(word, reference_word):
    """Makes the case of `word` like the case of `reference_word`.

    Supports lowercase, UPPERCASE, and Capitalized.
    """
    if reference_word.islower():
        return word.lower()
    elif reference_word.isupper() and len(reference_word) > 1:
        return word.upper()
    elif reference_word[0].isupper() and reference_word[1:].islower():
        return word.capitalize()
    else:
        # if other, just do not alter the word's case
        return word
