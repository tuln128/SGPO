import torch
import numpy as np
from evodiff.utils import Tokenizer
from models.pretraining.collaters import ESMTokenizer

class pairedAntibodyTokenizer(Tokenizer):
    EXTRA_TOKENS = ['|']

    def __init__(self, sequences=False):
        super().__init__(sequences=sequences)

        # Extend alphabet as plain attribute
        self.alphabet = self.alphabet + ['|']

        # Rebuild mappings to include '|'
        self.a_to_i = {c: i for i, c in enumerate(self.alphabet)}
        self.i_to_a = {i: c for i, c in enumerate(self.alphabet)}

        # Set new token index
        self.concat_id = self.a_to_i['|']   # index 31

        # Update K
        self.K = len(self.alphabet)          # 32

    def tokenize(self, seq):
        if isinstance(seq, tuple):
            seq = seq[0]
        # ✅ Use self.a_to_i[self.mask] directly — avoids calling mask_id property
        #    which would call tokenize() again causing infinite recursion
        fallback = self.a_to_i[self.mask]
        return [self.a_to_i.get(c, fallback) for c in seq]

    def untokenize(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        result = []
        for t in tokens:
            t = int(t)
            if t == self.pad_id:
                continue
            result.append(self.i_to_a.get(t, 'X'))
        return ''.join(result)

# Customized tokenizer for ESM-based models
class PairedESMTokenizer(ESMTokenizer):
    """
    Extends ESMTokenizer with '|' as chain concatenation token
    for paired heavy and light chain antibody sequences.
    """
    def __init__(self, esm_model_name='esm2_t12_35M_UR50D', sequences=True):
        super().__init__(esm_model_name=esm_model_name, sequences=sequences)

        # Add '|' to ESM alphabet at index 33
        self.alphabet.all_toks.append('|')
        self.alphabet.tok_to_idx['|'] = len(self.alphabet.all_toks) - 1
        self.concat_id = self.alphabet.tok_to_idx['|']   # index 33

    @property
    def pad_id(self):
        return self.padding_idx   # index 1

    def tokenize(self, seq):
        if isinstance(seq, tuple):
            seq = seq[0]
        tokens = []
        for c in seq:
            if c == '|':
                tokens.append(self.concat_id)
            else:
                idx = self.alphabet.get_idx(c)
                tokens.append(idx if idx is not None
                              else self.alphabet.get_idx('<unk>'))
        return np.array(tokens)

    def untokenize(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        result = []
        for t in tokens:
            t = int(t)
            if t == self.pad_id:
                continue
            elif t == self.concat_id:
                result.append('|')
            else:
                result.append(self.alphabet.get_tok(t))
        result = [tok for tok in result
                  if tok not in ('<cls>', '<eos>', '<pad>', '<unk>')]
        return ''.join(result)
