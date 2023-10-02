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
        transformation = WordDeletion()
        constraints = [RepeatModification(), StopwordModification()]
        #
        # Goal is untargeted classification
        #"The objective function is designed to delete the most words 
        #while keeping the prediction of the victim model constant."
        #
        goal_function = InputReduction(model_wrapper, maximizable=True)
        #
        # "For each word in an input sentence, we measure its importance by 
        # the gradient of the loss function."
        #
        search_method = GSA(wir_method="gradient")

        return Attack(goal_function, constraints, transformation, search_method)
#end = time.clock()
#print('Running time: %s Seconds' % (end - start))
