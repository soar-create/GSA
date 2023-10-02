from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import InputReduction
from textattack.search_methods import GSA
from textattack.transformations import WordDeletion

from .attack_recipe import AttackRecipe
#import time

#start = time.clock()
class GSA2023(AttackRecipe):
    """
    Gradient-based Simulated Annealing for Efficient Rubbish Text Attack
    """

    @staticmethod
    def build(model_wrapper):
        # At each step, we remove the word with the lowest importance value until
        # the model changes its prediction.
        transformation = WordDeletion()

        constraints = [RepeatModification(), StopwordModification()]
        #
        # Goal is untargeted classification
        #
        goal_function = InputReduction(model_wrapper, maximizable=True)
        #
        # "For each word in an input sentence, we measure its importance by 
        # the gradient of the loss function."
        #
        # "Instead of looking at the words with high importance values—what
        # interpretation methods commonly do—we take a complementary approach
        # and study how the model behaves when the supposedly unimportant words are
        # removed."
        #
        search_method = GSA(wir_method="gradient")

        return Attack(goal_function, constraints, transformation, search_method)
#end = time.clock()
#print('Running time: %s Seconds' % (end - start))
