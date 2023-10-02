"""
Beam Search
===============

"""
import numpy as np

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
import math
import torch
import random
from torch.nn.functional import softmax
from itertools import permutations

class BeamSearch(SearchMethod):
    """An attack that maintinas a beam of the `beam_width` highest scoring
    AttackedTexts, greedily updating the beam with the highest scoring
    transformations from the current beam.

    Args:
        goal_function: A function for determining how well a perturbation is doing at achieving the attack's goal.
        transformation (Transformation): The type of transformation.
        beam_width (int): the number of candidates to retain at each step
    """

    def __init__(self, beam_width=8):
        self.beam_width = beam_width
    """
    def perform_search(self, initial_result):
        beam = [initial_result.attacked_text]
        best_result = initial_result
        while not best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            potential_next_beam = []
            for text in beam:
                transformations = self.get_transformations(
                    text, original_text=initial_result.attacked_text
                )
                potential_next_beam += transformations

            if len(potential_next_beam) == 0:
                # If we did not find any possible perturbations, give up.
                return best_result
            results, search_over = self.get_goal_results(potential_next_beam)
            scores = np.array([r.score for r in results])
            best_result = results[scores.argmax()]
            if search_over:
                return best_result

            # Refill the beam. This works by sorting the scores
            # in descending order and filling the beam from there.
            best_indices = (-scores).argsort()[: self.beam_width]
            beam = [potential_next_beam[i] for i in best_indices]

        return best_result
    
    def perform_search(self, initial_result):
        T = 900 # initial temperature
        Tmin = (math.pow(0.9,inputlen+3))*T  # lowest temperature
        T_r=500
        Tmin_r = (math.pow(0.9,6))*T_r
        #k =round(inputlen*0.6)  # internal iterations
        #k=min(16,9+int(inputlen/5))
        k=15
        best_range=5
        adversarial_examples=[]
        x = initial_result  # initialize x as original input text
        xin = initial_result  # initialize xin as original input text in cycle
        t = 0  # time,external iterations
        
        while T >= Tmin:
            select_result=[]
            neighbors_list = [[] for _ in range(len(x.words))]
            best_neighbors = [[] for _ in range(len(x.words))]
            transformed_texts = self.get_transformations(
                x, original_text=initial_result.attacked_text
            )
            for transformed_text in transformed_texts:
                diff_idx = next(
                    iter(transformed_text.attack_attrs["newly_modified_indices"])
                )
                neighbors_list[diff_idx].append(transformed_text)
            for i in range(len(neighbors_list)):
                neighbor_results, self._search_over = self.get_goal_results(
                    neighbors_list[i]
                )
            neighbor_scores = np.array([r.score for r in neighbor_results])
            score_diff = neighbor_scores - x.score
            best_idxs = (neighbor_scores).argsort()[:best_range]
            best_neighbors[i]=[neighbor_results[m] for m in best_idxs]
            select_size=0
            while select_size<k:
                for j in range(best_range):
                    for i in range(len(best_neighbors)):
                        if j < len(best_neighbors[i]): 
                            select_result.append(best_neighbors[i][j])
                            select_size+=1
    """    
    def get_index_order(self, initial_text):
        """Returns word indices of ``initial_text`` in descending order of
        importance."""
        len_text = len(initial_text.words)
        """
        if self.wir_method == "unk":
            leave_one_texts = [
                initial_text.replace_word_at_index(i, "[UNK]") for i in range(len_text)
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])
        
        #elif self.wir_method == "weighted-saliency":
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
        """
        leave_one_texts = [
            initial_text.delete_word_at_index(i) for i in range(len_text)
        ]
            #print(leave_one_texts)
        leave_one_results, search_over = self.get_goal_results(leave_one_texts)
        index_scores = np.array([result.score for result in leave_one_results])
            #print(index_scores)
        """
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
        """
        #if self.wir_method != "random":
            #index_order = (index_scores).argsort()[::-1]
        index_order = (index_scores).argsort()
        return index_order,index_scores,search_over

    def perform_search(self, initial_result):
        # Sort words by order of importance
        #index_order, search_over = self._get_index_order(attacked_text)
        
        # Starts Simulated Annealing
        T = 1000  # initial temperature
        Ta=0.85 #温度下降率
        Tmin = (math.pow(Ta,len(initial_result.attacked_text.words)-2))*T  # lowest temperature
        k = 15  # internal iterations
        x = initial_result  # initialize x as original input text
        xin=initial_result
        #print(x.attacked_text)
        #print(x.score)
        while T >= Tmin:
            indices=[]
            for w in range(len(x.attacked_text.words)):
                word_pos = x.attacked_text.pos_of_word_index(w)
                word_rep = x.attacked_text.words[w]
                pos_dict = {"ADJ": "adj", "NOUN": "noun", "ADV": "adv", "VERB": "verb"}
                nonpos_word={"A":1,"An":1,"a":1,"an":1,"is":1,"are":1,"been":1,"am":1,"be":1,"was":1,"were":1,"has":1,"have":1}
                word_pos = pos_dict.get(word_pos, None)
                word_nonpos = nonpos_word.get(word_rep, None)
                if word_pos!=None and word_nonpos == None:
                    indices.append(w)
            #print(indices)
            if len(indices)!=0:
                indice_select=np.random.choice(indices,k)
                # indice = t + randomList[i]  # avoid repeated modification
                for indice in indice_select:
                    moves = self.get_transformations(
                                x.attacked_text,
                                original_text=initial_result.attacked_text,
                                indices_to_modify=indice
                                )
                # Skip words without candidates
                    if len(moves) == 0:
                        continue
                    y=xin.score
                    xNew, _ = self.get_goal_results(moves)
                    xNew_sorted = sorted(xNew, key=lambda x: -x.score)
                    #yNew = self._aim_function(initial_result, xNew_sorted[0])
                    yNew=xNew_sorted[0].score
                    if xNew_sorted[0].score!=0:
                        if yNew - y > 0:
                            xin = xNew_sorted[0]
                        else:
                            # metropolis principle
                            p = math.exp(-100/T)
                            r = np.random.uniform(low=0, high=1)
                            if r < p:
                                xin = xNew_sorted[0]
                    # If we succeeded, return the result.
            #t += 1
            #T = 1000/(1+t)  # quick annealing function
            #print(x.attacked_text)
            #print(x.score)
            #index_order,index_scores,search_over = self.get_index_order(x.attacked_text)
            #print(index_order)
            #print(index_scores)
            x=xin
            T=Ta*T
        return x

    @property
    def is_black_box(self):
        return False

    def extra_repr_keys(self):
        return ["beam_width"]
