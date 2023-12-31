def Qv1_BoolQ_0shot(passage, question):
    return f'''Read a passage below, followed by a yes/no question about this passage. Please answer the question, your output should contain either "true" or "false".

## Passage:
{passage}

## Question:
{question}

## Response:'''


def Qv1_MultiRC_0shot(paragraph, question, answer):
    return f'''Read the paragraph below, followed by a question about this paragraph. For a given answer to the question, you must predict whether the given answer is true or false. Make sure your response contains either the prediction word "true" or "false".

## Paragraph:
{paragraph}

## Question:
{question}

## Answer:
{answer}

## Response:'''


def Qv1_ReCoRD_0shot(passage, query, entities):
    return f'''Read a passage below, followed by a query about this passage. In the query, an entity is masked out by a "@placeholder" mark. You must predict the masked-out entity from a list of possible entities, where the same entity may be expressed with multiple
different surface forms, which are all considered correct.
You have to indicate your selected entity at the very end of your output, in the form of "@placeholder = <your selected entity>". You must select only one entity from the provided entity list, for example, if your selected entity is "Donald Trump", please output (without wrapping up in quotes): @placeholder = Donald Trump.

## Passage:
{passage}

## Query:
{query}

## Possible entities:
{entities}

## Response:'''


def Qv1_ReCoRD_output_process(output):
    if '@placeholder' in output:
        output = output.split('@placeholder')[-1]
        output = output.replace('=', ' ')
        return output.split('.')[0]
    else:
        return output


def Qv1_WiC_0shot(snippet1, snippet2, word):
    return f'''Given two text snippets and a polysemous word that appears in both sentences, determine whether the word is used with the same sense in both snippets. Output either "True" or "False" in your response.

## Snippet 1:
{snippet1}

## Snippet 2:
{snippet2}

## Question
Look at the word "{word}" (its position is indicated by wrapping a pair of double square brackets "[[" and "]]" around it), is it used with the same sense in both snippets?

## Response:'''


def Qv1_WSC_0shot(text):
    return f'''Given a text snippet where a pair of co-referenced pronoun and noun phrase are highlighted. Their appearance is denoted by wrapping a pair of double square brackets "[[" and "]]" around each. Determine whether the word pair is used with the correct referent. Output either "True" or "False" in your response.

## Text snippet:
{text}

## Response:'''
