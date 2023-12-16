from transformers import StoppingCriteria
from transformers import StoppingCriteriaList

common_stops = ['<|endoftext|>', '<|im_end|>', '</s>']
newline_stops = ['\n']
double_newline_stops = ['\n\n']
newsect_stops = ['##']

class KeywordsStopper(StoppingCriteria):
    def __init__(self, tokenizer, keywords):
        self.keywords = keywords
        self.tokenizer = tokenizer

    def is_stop(self, text):
        for kw in self.keywords:
            if kw in text:
                return True
        return False

    def __call__(self, input_ids, scores, **kwargs):
        batch_text = self.tokenizer.batch_decode(input_ids)
        batch_stop = (self.is_stop(b) for b in batch_text)
        return all(batch_stop)

    @staticmethod
    def make_list(tokenizer, keywords):
        return StoppingCriteriaList([
            KeywordsStopper(tokenizer, keywords)
        ])
