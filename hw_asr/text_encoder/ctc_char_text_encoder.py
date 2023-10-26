from typing import List, NamedTuple
from pyctcdecode import build_ctcdecoder
from collections import defaultdict
import torch
from torch import Tensor
import numpy as np

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None, path_to_lm = None, path_to_tokenizer = None):
        super().__init__(alphabet)
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        
        self.lm_decoder = build_ctcdecoder(
            labels=[self.EMPTY_TOK] + list("".join(self.alphabet).upper()),
            kenlm_model_path=path_to_lm
        )
            

    def ctc_decode(self, inds: List[int]) -> str:
        # TODO: your code here
        result = []
        last_char = self.EMPTY_TOK
        for ind in inds:
            if self.ind2char[ind] == last_char:
                continue
            if self.ind2char[ind] != self.EMPTY_TOK:
                result.append(self.ind2char[ind]) 
            last_char = self.ind2char[ind]
        return ''.join(result)

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size  = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []
        # TODO: your code here
        probs = probs[:probs_length]
        state = {('', self.EMPTY_TOK): 1.0}
        for next_char_probs in probs:
            state = self.extend_and_merge(next_char_probs, state)
            state = self.truncate(state, beam_size)
        
        for path, prob in state.items():
            hypos.append(Hypothesis(path[0], prob))

        return sorted(hypos, key=lambda x: x.prob, reverse=True)
    
    def extend_and_merge(self, next_char_probs, src_paths):
        new_state = defaultdict(float)
        for idx, next_char_prob in enumerate(next_char_probs):
            next_char = self.ind2char[idx]

            for (text, last_char), path_prob in src_paths.items():
                new_pref = text if next_char == last_char else text + next_char
                new_pref = new_pref.replace(self.EMPTY_TOK, '')
                new_state[(new_pref, next_char)] += path_prob * next_char_prob
        return new_state
    
    def truncate(self, state, beam_size):
        return dict(sorted(state.items(), key=lambda x: -x[1])[:beam_size])
    
    def ctc_lm_beam_search(self, probs: torch.tensor, probs_length,
                           beam_size: int = 100) -> List[Hypothesis]:
        hypos: List[Hypothesis] = []
        probs = probs[:probs_length].cpu().detach().numpy()
        decoded_texts = self.lm_decoder.decode_beams(probs, beam_width=beam_size)
        hypos = [Hypothesis(text.lower(), np.exp(lm_logits)) for text, _, _, _, lm_logits in decoded_texts]
        return hypos
