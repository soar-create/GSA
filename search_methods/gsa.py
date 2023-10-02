import numpy as np
import torch
import random
from torch.nn.functional import softmax
from itertools import permutations

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared.validators import (
    transformation_consists_of_word_swaps_and_deletions,
)

import math
import random


class GSA(SearchMethod):
    """An attack that greedily chooses from a list of possible perturbations in
    order of index, after ranking indices by importance.

    Args:
        wir_method: method for ranking most important words
        model_wrapper: model wrapper used for gradient-based ranking
    """

    def __init__(self, wir_method="unk"):
        self.wir_method = wir_method

    def _get_index_order(self, initial_text):
        """Returns word indices of ``initial_text`` in descending order of
        importance."""
        len_text = len(initial_text.words)

        if self.wir_method == "unk":
            leave_one_texts = [
                initial_text.replace_word_at_index(i, "[UNK]") for i in range(len_text)
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])

        elif self.wir_method == "weighted-saliency":
            # first, compute word saliency
            leave_one_texts = [
                initial_text.replace_word_at_index(i, "[UNK]") for i in range(len_text)
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            saliency_scores = np.array([result.score for result in leave_one_results])

            softmax_saliency_scores = softmax(
                torch.Tensor(saliency_scores), dim=0
            ).numpy()

            # compute the largest change in score we can find by swapping each word
            delta_ps = []
            for idx in range(len_text):
                transformed_text_candidates = self.get_transformations(
                    initial_text,
                    original_text=initial_text,
                    indices_to_modify=[idx],
                )
                if not transformed_text_candidates:
                    # no valid synonym substitutions for this word
                    delta_ps.append(0.0)
                    continue
                swap_results, _ = self.get_goal_results(transformed_text_candidates)
                score_change = [result.score for result in swap_results]
                if not score_change:
                    delta_ps.append(0.0)
                    continue
                max_score_change = np.max(score_change)
                delta_ps.append(max_score_change)

            index_scores = softmax_saliency_scores * np.array(delta_ps)

        elif self.wir_method == "delete":
            leave_one_texts = [
                initial_text.delete_word_at_index(i) for i in range(len_text)
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])
            

        elif self.wir_method == "gradient":
            victim_model = self.get_victim_model()
            index_scores = np.zeros(initial_text.num_words)
            grad_output = victim_model.get_grad(initial_text.tokenizer_input)
            gradient = grad_output["gradient"]
            word2token_mapping = initial_text.align_with_model_tokens(victim_model)
            for i, word in enumerate(initial_text.words):
                matched_tokens = word2token_mapping[i]
                #print(matched_tokens)
                if not matched_tokens:
                    index_scores[i] = 0.0
                else:
                    if matched_tokens!=None:
                    	matched_tokens=None
                    else:
                    	agg_grad = np.mean(gradient[matched_tokens], axis=0)
                    	index_scores[i] = np.linalg.norm(agg_grad, ord=1)

            search_over = False

        elif self.wir_method == "random":
            index_order = np.arange(len_text)
            np.random.shuffle(index_order)
            search_over = False
        else:
            raise ValueError(f"Unsupported WIR method {self.wir_method}")

        if self.wir_method != "random":
            index_order = (index_scores).argsort()[::-1]
    
        return index_order, search_over
    
    def perform_search(self, initial_result):
        attacked_text = initial_result.attacked_text
        
        # Sort words by order of importance
        index_order, search_over = self._get_index_order(attacked_text)
        # Starts Simulated Annealing
        inputlen=len(index_order)
        T = 900 # initial temperature
        Tmin = (math.pow(0.9,inputlen+3))*T  # lowest temperature
        #adjust k for the length of the input text
        if inputlen < 21:
            k=14
        else:
            k=15
        x = initial_result  # initialize x as original input text
        xin = initial_result  # initialize xin as original input text in cycle
        t = 0  # time,external iterations
        while T >= Tmin:
            index_order, search_over = self._get_index_order(x.attacked_text)
            weights=[]
            iw=0       
            if t==0:
                weights=torch.Tensor([1,0])
                weightsam=torch.multinomial(weights,1)
                listk=weightsam.numpy()
            
            elif t==inputlen-2:
                listk=[0]
            else:
                iw=0
                while iw  < len(index_order):
                    
                    if 1-(1/len(index_order))*iw<0 or iw>min(3*t,len(index_order)):
                        w=0
                    else:
                        w=1-(1/len(index_order))*iw
                    weights.append(w)
                    iw+=1
                
                weights=torch.Tensor(weights)
                weightsam=torch.multinomial(weights,k,replacement=True)
                listk=weightsam.numpy()
            for indice in listk:
                y=xin.score
                if indice >= len(index_order):
                    #print("continue due to indice out of range")
                    continue
                moves = self.get_transformations(
                            x.attacked_text,
                            original_text=initial_result.attacked_text,
                            indices_to_modify=[index_order[indice]]
                            )
                if len(moves) == 0:
                    continue
                xNew, _ = self.get_goal_results(moves)
                xNew_sorted = sorted(xNew, key=lambda x: -x.score)
                yNew=xNew_sorted[0].score
                if xNew_sorted[0].score!=0:
                    if yNew-y > 0:
                        xin = xNew_sorted[0]
                    else:
                    # metropolis principle
                        p = math.exp(-((y-yNew)*1000000)/T)
                        r = np.random.uniform(low=0, high=1)
                        #r=torch.rand(1).item()
                        if r < p:
                            xin = xNew_sorted[0]
                else:
                    continue
            t += 1
            T = 0.9*T
            x=xin
        
        return x
        
    def check_transformation_compatibility(self, transformation):
        """Since it ranks words by their importance, GreedyWordSwapWIR is
        limited to word swap and deletion transformations."""
        return transformation_consists_of_word_swaps_and_deletions(transformation)

    @property
    def is_black_box(self):
        if self.wir_method == "gradient":
            return False
        else:
            return True

    def extra_repr_keys(self):
        return ["wir_method"]
