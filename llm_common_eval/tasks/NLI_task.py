def Qv1_0shot(hypothesis, premise):
    return f'''Given a hypothesis: '{hypothesis}', please determine the truthfulness of the hypothesis by looking at the premise: '{premise}'. Output a label number 0, indicating that the hypothesis entails the premise. Or a label number 1, indicating that the premise and hypothesis neither entail nor contradict each other. Or 2, indicating that the hypothesis contradicts the premise.'''


def Qv1_0shot_cot(hypothesis, premise):
    return Qv1_0shot(hypothesis, premise) + " Let's think step by step. Output your selected label in the final step."


def Qv1_example(hypothesis, premise, label=None):
    if label is None:
        return f'Hypothesis: "{hypothesis}". Premise: "{premise}".'
    else:
        return f'Hypothesis: "{hypothesis}". Premise: "{premise}". Output label: "{label}".'


def Qv1_fewshot(hypothesis, premise, support_set):
    Q = f'''Given a hypothesis, please determine the truthfulness of the hypothesis by looking at the premise. Output a label number 0, indicating that the hypothesis entails the premise. Or a label number 1, indicating that the premise and hypothesis neither entail nor contradict each other. Or 2, indicating that the hypothesis contradicts the premise.'''
    Q += '\n'
    for i, support in enumerate(support_set):
        Q += f'Example#{i}:\n' + Qv1_example(*support) + '\n'
    Q += 'Now, given:\n' + Qv1_example(hypothesis, premise)
    return Q
