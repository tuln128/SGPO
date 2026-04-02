import hydra
import torch.nn as nn
import omegaconf
import torch
import torch.nn.functional as F
import numpy as np
import transformers

#from evodiff.model import ByteNetLMTime
from evodiff.pretrained import D3PM_UNIFORM_38M
from evodiff.losses import D3PMCELoss, D3PMLVBLoss
from sequence_models.metrics import MaskedAccuracy
from sequence_models.constants import MSA_ALPHABET

from models.pretraining.trainer import BaseModel

pretrained_mapping = {"D3PM_UNIFORM_38M": D3PM_UNIFORM_38M}

class ByteNetDiffusion(BaseModel):
    """
    ByteNet convolutional model with time, used in the diffusion model. Inherits from pl.LightningModule.
    """
    #TODO: uses src, src_onehot, timestep, tgt, tgt_onehot, Q, Q_bar, q = batch
    #outputs loss and any other relevant outputs
    def __init__(self, model_config, network, tokenizer, optimizer, lr_scheduler):
        """
        Initializes the ByteNetDiffusion model.
        config: Hydra config object
        network: ByteNetLMTime model
        tokenizer: Tokenizer object
        optimizer: Optimizer object
        lr_scheduler: Learning rate scheduler object
        device: torch.device
        """
        super().__init__()
        self.save_hyperparameters()

        #can remove this once we use the newer models    
        try: 
            model_config.pretrained_evodiff_ckpt
        except omegaconf.errors.ConfigAttributeError:
            model_config.pretrained_evodiff_ckpt = None

        if model_config.pretrained_evodiff_ckpt is not None:
            self.network, _, self.tokenizer, _ = pretrained_mapping[model_config.pretrained_evodiff_ckpt](return_all=False) #load pretrained evodiff checkpoint and finetune from there
            self.padding_idx = self.tokenizer.pad_id  # PROTEIN_ALPHABET.index(PAD)
            self.masking_idx = self.tokenizer.mask_id
        else:
            if model_config.mask == 'uniform':
                tokenizer = hydra.utils.instantiate(tokenizer, sequences=True)
            self.tokenizer = tokenizer
            self.padding_idx = self.tokenizer.pad_id  # PROTEIN_ALPHABET.index(PAD)
            self.masking_idx = self.tokenizer.mask_id
            self.network = hydra.utils.instantiate(network, n_tokens=len(MSA_ALPHABET), padding_idx=self.masking_idx)

            #network.n_tokens = len(MSA_ALPHABET)
            #network.padding_idx = self.masking_idx

        self.opt = hydra.utils.instantiate(optimizer, params=self.network.parameters())

        self.lr_scheduler = None
        if lr_scheduler:
            self.lr_scheduler = hydra.utils.instantiate(lr_scheduler, self.opt)
        
        self.loss_func1 = D3PMLVBLoss(tmax=self.network.embedder.timesteps, tokenizer=self.tokenizer)
        self.loss_func2 = D3PMCELoss(tokenizer=self.tokenizer)
        self._lambda = model_config._lambda

        self.accuracy_function = MaskedAccuracy()
    
    def forward(self, batch):
        
        ### Modified from https://github.com/microsoft/evodiff/blob/main/train.py ###
        src, src_onehot, timestep, tgt, tgt_onehot, Q, Q_bar, q = batch
        q = q.to(self.device)
        Q = Q.to(self.device)
        Q_bar = Q_bar.to(self.device)
        src_onehot = src_onehot.to(self.device)
        tgt_onehot = tgt_onehot.to(self.device)

        timestep = timestep.to(self.device)
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        input_mask = (src != self.padding_idx).float()

        n_tokens = input_mask.sum()

        n_processed = input_mask.sum()
        n_seqs = torch.tensor(len(src), device=self.device)

        outputs = self.network(src, timestep, input_mask=input_mask.unsqueeze(-1))

        lvb_loss = self.loss_func1(src_onehot, q, outputs, tgt, tgt_onehot, input_mask, timestep, Q, Q_bar)
        ce_loss = self.loss_func2(outputs, tgt, input_mask)
        lvb_loss = lvb_loss.to(torch.float32)
        ce_loss = ce_loss.to(torch.float32)

        #TODO: check if this is correct and what each loss is doing
        #evodiff seems to use loss, not nll_loss, but this does not look stable during training

        loss = (lvb_loss + (self._lambda * ce_loss)) * n_tokens
        nll_loss = ce_loss * n_tokens
        accuracy = self.accuracy_function(outputs, tgt, input_mask) * n_tokens

        out = {"loss": loss, "nll_loss": nll_loss, "accuracy": accuracy}
        return out