import torch
import numpy as np
from evodiff.utils import Tokenizer

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