default_label_ids={
    'entailment': 0,
    'neutral': 1,
    'contradiction': 2,
    'not_entailment': 1
}

def Qv1_0shot(hypothesis, premise, label_ids=default_label_ids):
    return f'''Given a hypothesis: '{hypothesis}', please determine the truthfulness of the hypothesis by looking at the premise: '{premise}'. Output a label number {label_ids['entailment']}, indicating that the hypothesis entails the premise. Or a label number {label_ids['neutral']}, indicating that the premise and hypothesis neither entail nor contradict each other. Or {label_ids['contradiction']}, indicating that the hypothesis contradicts the premise.'''


def Qv1_0shot_cot(hypothesis, premise):
    return Qv1_0shot(hypothesis, premise) + " Let's think step by step. Output your selected label in the final step."


def Qv1_example(hypothesis, premise, label=None):
    if label is None:
        return f'Hypothesis: "{hypothesis}". Premise: "{premise}".'
    else:
        return f'Hypothesis: "{hypothesis}". Premise: "{premise}". Output label: "{label}".'


def Qv1_fewshot(hypothesis, premise, support_set, label_ids=default_label_ids):
    Q = f'''Given a hypothesis, please determine the truthfulness of the hypothesis by looking at the premise. Output a label number {label_ids['entailment']}, indicating that the hypothesis entails the premise. Or a label number {label_ids['neutral']}, indicating that the premise and hypothesis neither entail nor contradict each other. Or {label_ids['contradiction']}, indicating that the hypothesis contradicts the premise.'''
    Q += '\n'
    for i, support in enumerate(support_set):
        Q += f'Example#{i}:\n' + Qv1_example(*support) + '\n'
    Q += 'Now, given:\n' + Qv1_example(hypothesis, premise)
    return Q


def Qv1_RTE_0shot(hypothesis, premise, label_ids=default_label_ids):
    return f'''Given a hypothesis and a premise below, please determine the truthfulness of the hypothesis by looking at the premise. Output a label number {label_ids['entailment']}, indicating that the hypothesis entails the premise. Or {label_ids['not_entailment']}, indicating non-entailment.

## Premise:
{premise}

## Hypothesis:
{hypothesis}

## Response:'''
