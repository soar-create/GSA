"""

Determine successful in untargeted Classification
----------------------------------------------------
"""


from .classification_goal_function import ClassificationGoalFunction


class UntargetedClassification(ClassificationGoalFunction):
    """An untargeted attack on classification models which attempts to minimize
    the score of the correct label until it is no longer the predicted label.

    Args:
        target_max_score (float): If set, goal is to reduce model output to
            below this score. Otherwise, goal is to change the overall predicted
            class.
    """

    def __init__(self, *args, target_max_score=None, **kwargs):
        self.target_max_score = target_max_score
        super().__init__(*args, **kwargs)

    def _is_goal_complete(self, model_output, _):
        if self.target_max_score:
            return model_output[self.ground_truth_output] < self.target_max_score
        elif (model_output.numel() == 1) and isinstance(
            self.ground_truth_output, float
        ):
            return abs(self.ground_truth_output - model_output.item()) >= 0.5
        else:
            #执行这一条
            return model_output.argmax() != self.ground_truth_output

    def _get_score(self, model_output, attacked_text):
        # If the model outputs a single number and the ground truth output is
        # a float, we assume that this is a regression task.
        cur_words = attacked_text.words
        initial_words = self.initial_attacked_text.words
        initial_num_words=self.initial_attacked_text.num_words
        if self.ground_truth_output != model_output.argmax():
            return 0
        if (model_output.numel() == 1) and isinstance(self.ground_truth_output, float):
            return abs(model_output.item() - self.ground_truth_output)
        else:
            #return 1 - model_output[self.ground_truth_output]
            num_words_score = len(set(cur_words)-set(initial_words))
            model_score = model_output[self.ground_truth_output]
            #print(model_score)
            SCORE=(num_words_score + 2.5*model_score) / initial_num_words
            return SCORE
            #return (num_words_score/initial_num_words) + 0.5*model_score
