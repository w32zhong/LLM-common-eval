def Qv1_0shot(hypothesis, premise):
    return f'''Given a hypothesis: '{hypothesis}', please determine the truthfulness of the hypothesis by looking at the premise: '{premise}'. Output a label number 0, indicating that the hypothesis entails the premise. Or a label number 1, indicating that the premise and hypothesis neither entail nor contradict each other. Or 2, indicating that the hypothesis contradicts the premise.'''


def Qv1_0shot_cot(hypothesis, premise):
    return Qv1_0shot(hypothesis, premise) + " Let's think step by step. Output your selected label in the final step."
