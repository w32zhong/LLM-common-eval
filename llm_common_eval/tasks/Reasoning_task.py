def Qv1_COPA_0shot(premise, choices, cause_or_effect):
    return f'''Given a premise below, followed by two possible choices. Please determine determine the {cause_or_effect} of the premise, and output the choice number that you think is correct from these two possible choices.
## Premise:
{premise}

## Choices:
Choice 1: {choices[0]}
Choice 2: {choices[1]}

## Response:'''
