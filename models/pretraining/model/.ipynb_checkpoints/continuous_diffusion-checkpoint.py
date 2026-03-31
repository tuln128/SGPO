#Modified from https://github.com/ngruver/NOS/blob/main/seq_models/model/gaussian_diffusion.py#L302 ###
import os
import time
import tqdm
import wandb
import hydra
import math
import numpy as np

from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from transformers import AutoConfig, AutoModel
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertPooler,
    BertEncoder,
    BertEmbeddings,
    BertOnlyMLMHead,
)

import esm

from models.pretraining.trainer import BaseModel

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class RegressionHead(nn.Module):

    def __init__(
            self,
            config,
            out_channels,
            stop_grad=True,
    ):
        super().__init__()

        self.pooler = BertPooler(config)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.regression_dropout = nn.Dropout(classifier_dropout)
        self.regression_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, out_channels)
        )

        self.stop_grad = stop_grad

    def forward(self, sequence_output, attn_mask=None):
        if self.stop_grad:
            sequence_output = sequence_output.detach()
        pooled_output = sequence_output.mean(1)
        pooled_output = self.regression_dropout(pooled_output)
        regression_pred = self.regression_head(pooled_output)
        return regression_pred


class RegressionModel(nn.Module):

    def __init__(
            self,
            config,
            in_channels,
            model_channels,
            out_channels,
            stop_grad=True,
    ):
        super().__init__()

        self.model_channels = model_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, config.hidden_size),
        )

        self.input_up_proj = nn.Linear(in_channels, config.hidden_size)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.regression_transformers = BertEncoder(config)
        self.regression_head = RegressionHead(config, out_channels)

        self.stop_grad = stop_grad

    def forward(self, sequence_embeds, timesteps, attn_mask=None):
        if self.stop_grad:
            sequence_embeds = sequence_embeds.detach()

        t_feats = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_feats)

        emb_x = self.input_up_proj(sequence_embeds)
        seq_length = sequence_embeds.size(1)
        position_ids = self.position_ids[:, : seq_length]

        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        if attn_mask is None:
            attn_mask = torch.ones_like(sequence_embeds[..., 0])
        attn_mask = attn_mask[:, None, None, :]

        h = self.regression_transformers(emb_inputs, attention_mask=attn_mask).last_hidden_state
        return self.regression_head(h)


class GaussianDiffusionTransformer(nn.Module):
    def __init__(
            self,
            in_channels,
            vocab_size,
            dropout=0,
            bert_config_name='bert-base-uncased',
            target_channels=2,
            discr_stop_grad=True,
            esm_model_name=None,  # specify the ESM model to use; set to None to disable
            use_esm_head=False  # if True, use the pretrained ESM head
    ):
        super().__init__()

        config = AutoConfig.from_pretrained(bert_config_name)
        config.hidden_dropout_prob = dropout
        config.vocab_size = vocab_size  # evodiff num instead of vocab_size

        self.vocab_size = vocab_size

        self.dropout = dropout

        # NEW: Conditional branch based on whether an ESM model is provided.
        if esm_model_name is not None:
            if esm_model_name == 'facebook/esm2_t12_35M_UR50D':
                self.esm_model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()  # ADDED: using esm.pretrained
            else:
                # Fallback: you can add additional model options here if needed.
                ValueError(f"Unsupported ESM model name: {esm_model_name}")
                #self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()  # ADDED: fallback option
            # Optionally freeze the ESM model parameters:
            for p in self.esm_model.parameters():
                p.requires_grad = False
            # Use the ESM model's hidden size (no longer hyperparam)
            
            self.in_channels = self.esm_model.embed_dim
            if use_esm_head:
                config.hidden_size = self.esm_model.embed_dim #added this for now, goes from 512 to 480 which is not a big difference
        else:
            # Without esm:
            self.in_channels = in_channels
            self.word_embedding = nn.Embedding(vocab_size, self.in_channels)

        self.time_embed_dim = config.hidden_size
        self.time_embed = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.SiLU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size),
        )

        # If your in_channels no longer match, you may need a mapping layer.
        self.input_up_proj = nn.Linear(self.in_channels, config.hidden_size)

        # FIX: set _attn_implementation required by transformers >= 4.40
        config._attn_implementation = "eager"  # ← add this line

        #print(self.esm_model.__dict__)
        self.encoder = BertEncoder(config)
        #self.encoder = self.esm_model.layers

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if use_esm_head and esm_model_name is not None:
            self.cls = self.esm_model.lm_head #finetuning this is better than leaving it fixed, according to DiMA
        else:
            self.cls = BertOnlyMLMHead(config)

        if target_channels > 0:
            self.regression_head = RegressionHead(
                config,
                target_channels,
                stop_grad=discr_stop_grad
            )

    def get_embeds(self, input_ids):
        # When using ESM embeddings:
        if hasattr(self, 'esm_model'):
            # Ensure that input_ids are tokenized as expected by the ESM model.
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            with torch.no_grad():
                #esm_outputs = self.esm_model(input_ids) #hugging face method
                #embeds = esm_outputs.last_hidden_state  # shape: [B, T, d]
                embeds = self.esm_model.embed_tokens(input_ids)
            # Normalize each token embedding to have unit norm (with numerical stability)
            normed = embeds / (embeds.norm(dim=-1, keepdim=True) + 1e-8)
            # scale by sqrt(in_channels)
            return (np.sqrt(self.in_channels) * normed).squeeze(0)
        else:
            # Without esm: use the standard word embedding.
            embeds = self.word_embedding(input_ids)
            normed = embeds / embeds.norm(dim=-1, keepdim=True)
            return np.sqrt(self.in_channels) * normed

    def get_logits(self, hidden_repr):
        return self.cls(hidden_repr)

    def forward(
            self,
            x,
            timesteps,
            attn_mask=None,
    ):
        # Normalize input x and scale by sqrt(in_channels)
        x = x / x.norm(dim=-1, keepdim=True)
        x = np.sqrt(self.in_channels) * x

        time_emb = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))

        emb_x = self.input_up_proj(x)
        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length]

        emb_inputs = self.position_embeddings(position_ids) + emb_x
        emb_inputs = emb_inputs + time_emb.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        if attn_mask is not None: #added this line
            attn_mask = attn_mask[:, None, None, :]

        encoder_outputs = self.encoder(emb_inputs, attention_mask=attn_mask)
        sequence_output = encoder_outputs[0]

        prediction_scores = self.cls(sequence_output)

        out = {
            "logits": prediction_scores,
            "sequence_output": sequence_output,
        }

        return out

    def pred_xstart(
            self,
            x,
            timesteps,
            attn_mask=None,
            sequence_output=None,
            bad_word_ids=None,
            # NEW:
            classifier=None,
            guidance_scale=0.0,
            infill_mask=None  # no longer needed, but we keep the argument for compatibility
    ):
        """
        Predict x₀ (aka xstart) from xₜ. Optionally apply classifier guidance
        by modifying xstart = xstart + (guidance term).

        For the continuous model, the input x is in embedding space.
        Now, the classifier already selects the mutated residues, so we can use
        its residue selection (assumed to be available via classifier.residues)
        instead of using an external mask.
        """
        # 1) Compute the un-guided xstart from model's logits.
        if sequence_output is None:
            logits = self(x, timesteps, attn_mask=attn_mask)["logits"]
        else:
            logits = self.cls(sequence_output)

        if bad_word_ids is not None:
            logits[:, :, bad_word_ids] = -1e9

        # Compute probabilities over the vocabulary.
        probs = F.softmax(logits, dim=-1)
        # When using ESM, get the embedding table from the esm_model instead of self.word_embedding.
        if hasattr(self, 'esm_model'):
            all_embeds = self.esm_model.embed_tokens.weight  # shape: [vocab_size, embed_dim]
            all_embeds = all_embeds / (all_embeds.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            all_embeds = self.word_embedding.weight
            all_embeds = all_embeds / all_embeds.norm(dim=-1, keepdim=True)
        # Scale by sqrt(in_channels) as before.
        all_embeds = np.sqrt(self.in_channels) * all_embeds

        # Decode to get xstart in embedding space.
        xstart = probs @ all_embeds  # shape: [B, T, embed_dim]

        # 2) Apply classifier guidance if a classifier is provided.
        if (classifier is not None) and (guidance_scale > 0.0):
            # Clone x to enable gradients.
            x_ = x.detach().clone().requires_grad_(True)  # x_ is [B, T, C]

            # Instead of using an external infill_mask, use the classifier's residue selection.
            if hasattr(classifier, "residues") and (classifier.residues is not None):
                # classifier.residues is assumed to be a list or array of mutated residue indices (0-indexed).
                indices = torch.tensor(classifier.residues, device=x.device)
                masked_x = x_.index_select(1, indices)  # shape: [B, num_mutated, C]
            else:
                # If no residue information is provided, fall back to using the full x.
                masked_x = x_
                indices = torch.arange(x.shape[1], device=x.device)

            # Compute classifier output on the selected positions.
            with torch.enable_grad():
                # The classifier already selects the mutated residues in its forward pass,
                # so passing masked_x is sufficient.
                classifier_output = classifier(masked_x, timesteps)  # shape: [B, 1]
                gradients = torch.autograd.grad(classifier_output.sum(), masked_x)[0]  # shape: [B, num_mutated, C]

            # Now, create a full tensor of gradients of shape [B, T, C] (T = full sequence length)
            full_gradients = torch.zeros_like(xstart)
            full_gradients[:, indices, :] = gradients  # only mutated positions receive nonzero gradients

            # Add the guidance term to xstart.
            # print(gradients)
            xstart = xstart + guidance_scale * full_gradients

        out = {
            "probs": probs,
            "xstart": xstart
        }
        return out

    def posterior_sample(
            self,
            noise_schedule,
            x,
            t,
            attn_mask=None,
            infill_mask=None,
            corrupt_mask=None,
            gt_vals=None,
            sequence_output=None,
            bad_word_ids=None,
            # NEW:
            classifier=None,
            guidance_scale=0.0,
    ):

        out = self.pred_xstart(
            x,
            t,
            attn_mask=attn_mask,
            sequence_output=sequence_output,
            bad_word_ids=bad_word_ids,
            classifier=classifier,  # <--- new
            guidance_scale=guidance_scale,  # <--- new
            infill_mask=infill_mask
        )

        mean, _, logvar = noise_schedule.q_posterior_mean_variance(
            x_start=out['xstart'], x_t=x, t=t
        )

        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        sigma = torch.exp(0.5 * logvar)
        x = mean + nonzero_mask * sigma * noise

        if gt_vals is not None:
            noise_t = torch.maximum(t[:1] - 1, torch.zeros_like(t[:1]))
            noisy_gt = noise_schedule.q_sample(gt_vals, noise_t)
            noisy_gt = torch.where((corrupt_mask * nonzero_mask).bool(), noisy_gt, gt_vals)
            x = torch.where(infill_mask.bool(), x, noisy_gt)

        out = {
            "x": x,
            "probs": out["probs"],
            "mean": mean,
            "sigma": sigma,
        }

        return out

    def get_labels(
            self,
            x,
            timesteps,
            attn_mask=None,
            sequence_output=None
    ):
        if sequence_output is None:
            sequence_output = self.forward(x, timesteps, attn_mask)['sequence_output']
        return self.regression_head(sequence_output)

    def guidance_score(
            self,
            x,
            timesteps,
            attn_mask=None,
            sequence_output=None
    ):
        labels = self.get_labels(x, timesteps, attn_mask, sequence_output)
        return labels.sum(-1)


class GaussianDiffusion(BaseModel):

    def __init__(self,
                 model_config,
                 network,
                 noise_schedule,
                 optimizer,
                 lr_scheduler,
                 tokenizer):
        super().__init__()
        self.save_hyperparameters()

        self.network = hydra.utils.instantiate(network)
        self.noise_schedule = hydra.utils.instantiate(noise_schedule)
        self.opt = hydra.utils.instantiate(optimizer, params=self.parameters())
        self.lr_scheduler = None
        if lr_scheduler:
            self.lr_scheduler = hydra.utils.instantiate(lr_scheduler, self.opt)
        self.tokenizer = hydra.utils.instantiate(tokenizer, sequences=True)

    def freeze_for_discriminative(self):
        for _, p in enumerate(self.network.parameters()):
            p.requires_grad_(False)

        for _, p in enumerate(self.network.regression_model.parameters()):
            p.requires_grad_(True)

    def forward(self, batch):
        # 'batch' might be a dict: {"seq", "corrupt_mask", "attn_mask", "labels"}
        input_ids = batch["seq"]
        corrupt_mask = batch["corrupt_mask"]
        attn_mask = batch["attn_mask"]
        labels = batch.get("labels", None)

        # labels placeholder (should go in collater)
        # batch_size = batch["seq"].shape[0]
        # labels = torch.zeros(batch_size, 1, device=batch["seq"].device)

        return_by_timestep = False #added in

        timesteps = self.noise_schedule.timesteps
        t = torch.randint(
            timesteps,
            size=(input_ids.shape[0],),
            device=input_ids.device,
            dtype=torch.int64,
        )

        embeds = self.network.get_embeds(input_ids)
        x_t = self.noise_schedule.q_sample(embeds, t)
        x_t = torch.where(corrupt_mask[..., None].bool(), x_t, embeds)

        model_output = self.network.forward(x_t, t, attn_mask)
        logits = model_output['logits']

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        nll = loss_fct(logits.view(-1, logits.shape[-1]), input_ids.view(-1))
        nll = nll.view(*input_ids.shape[:2])

        loss_mask = attn_mask * corrupt_mask
        nll = (nll * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)
        accuracy = ((logits.argmax(-1) == input_ids) * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)
        loss = nll.mean()

        out = {}
        out["loss"] = loss.mean()
        out["nll"] = nll.mean()
        out["accuracy"] = accuracy.mean()

        if labels is not None:
            pred_labels = self.network.get_labels(x_t, t, attn_mask)
            regression_loss = (pred_labels - labels).pow(2)
            out["regression_mse"] = regression_loss.mean()
            out["regression_spearman"] = spearmanr(
                pred_labels[:, 0].detach().cpu().numpy(),
                labels[:, 0].detach().cpu().numpy(),
            ).correlation

        if not return_by_timestep:
            return out

        num_buckets = 4
        step_size = timesteps // num_buckets
        for t_lower in np.arange(0, timesteps, step_size):
            t_upper = t_lower + step_size
            t_mask = (t > t_lower) * (t < t_upper)

            tag = f"accuracy_{t_lower}-{t_upper}"
            out[tag] = accuracy[t_mask].mean()

            if labels is not None:
                tag = f"regression_mse_{t_lower}-{t_upper}"
                out[tag] = regression_loss[t_mask].mean()

        return out

    def guidance_steps(
            self,
            model_output,
            t,
            attn_mask,
            infill_mask,
            bad_word_ids,
            corrupt_mask=None,
            gt_vals=None,
            guidance_layer="first",
            step_size=0.1,
            stability_coef=1e-2,
            num_steps=5,
    ):
        if guidance_layer == "last":
            kl_loss = torch.nn.KLDivLoss(log_target=True)
            x_prev = model_output['x_prev'].detach()
            h = model_output['sequence_output'].detach()
            logits = model_output['logits'].detach()
            delta = torch.nn.Parameter(torch.zeros_like(h), requires_grad=True)
        elif guidance_layer == "first":
            x = model_output['x'].detach()
            mean = model_output['mean'].detach()
            sigma = model_output['sigma'].detach()
            delta = torch.nn.Parameter(torch.zeros_like(x), requires_grad=True)
        else:
            raise NotImplementedError()

        optimizer = torch.optim.Adagrad([delta], lr=step_size)

        with torch.enable_grad():
            # print("\n")
            for _ in range(num_steps):
                if guidance_layer == "last":
                    h_current = h + infill_mask * delta

                    target_loss = self.network.guidance_score(
                        None, t, attn_mask, sequence_output=h_current
                    ).sum()
                    new_logits = self.network.cls(h_current)

                    kl = kl_loss(new_logits, logits)
                    # print(kl.item())
                    loss = -target_loss + stability_coef * kl

                    # print(infill_mask[0,:,0])
                    # print(h_current[0,:,0][infill_mask[0,:,0]])
                    # print(target_loss.item())
                    # print(kl.item())
                    # print(delta.pow(2).mean())
                    # print("")

                elif guidance_layer == "first":
                    x_current = x + infill_mask * delta
                    target_loss = self.network.guidance_score(x_current, t, attn_mask).sum()
                    nll = ((x_current - mean) ** 2 / sigma).sum()
                    loss = -target_loss + stability_coef * nll

                    # print(target_loss.item())
                    # print(nll.item())
                    # print(delta.pow(2).mean())
                    # print(self.network.regression_head.stop_grad)
                    # print("")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # delta_grad_norm = delta.grad.norm(dim=(2), keepdim=True)
                # delta.grad /= delta_grad_norm.clamp_min(1e-7)

        if guidance_layer == "last":
            p_out = self.network.posterior_sample(
                self.noise_schedule,
                x_prev, t, attn_mask, infill_mask,
                corrupt_mask, gt_vals,
                sequence_output=(h + delta.data),
                bad_word_ids=bad_word_ids
            )
            x, probs = p_out['x'], p_out['probs']
        elif guidance_layer == "first":
            x = x + delta.data
            probs = self.network.pred_xstart(
                x, t, attn_mask=attn_mask, bad_word_ids=bad_word_ids
            )["probs"]

        return {
            "x": x,
            "probs": probs,
        }

    def sample(
            self,
            infill_seed,
            infill_mask,
            corrupt_mask,
            num_samples,
            guidance_kwargs=None,
            bad_word_ids=None,
    ):
        device = next(self.parameters()).device

        gt_vals = self.network.get_embeds(infill_seed)

        infill_mask = infill_mask[None, :, None]
        corrupt_mask = corrupt_mask[None, :, None]
        gt_vals = gt_vals[None]

        indices = list(range(self.noise_schedule.timesteps))[::-1]

        t = torch.tensor([indices[0]], device=device)
        noisy_gt = self.noise_schedule.q_sample(gt_vals, t)
        noisy_gt = torch.where(corrupt_mask, noisy_gt, gt_vals)

        x = self.noise_schedule.noise_scale * torch.randn(
            (num_samples, gt_vals.shape[1], gt_vals.shape[-1]), device=device
        )
        x = torch.where(infill_mask, x, noisy_gt)
        attn_mask = torch.ones(
            (num_samples, infill_mask.shape[1]), dtype=torch.bool, device=device
        )

        return_best = guidance_kwargs.pop("return_best", False) \
            if guidance_kwargs is not None else False

        traj = []
        for i in tqdm.tqdm(indices):
            t = torch.tensor([i] * num_samples, device=device)

            with torch.no_grad():
                f_out = self.network.forward(
                    x, t, attn_mask=attn_mask,
                )

                p_out = self.network.posterior_sample(
                    self.noise_schedule,
                    x, t, attn_mask, infill_mask,
                    corrupt_mask, gt_vals,
                    sequence_output=f_out['sequence_output'],
                    bad_word_ids=bad_word_ids
                )
                p_out['x_prev'] = x
                out = {**f_out, **p_out}

                x = out['x']
                probs = out['probs']

            if guidance_kwargs is not None:
                g_out = self.guidance_steps(
                    out, t, attn_mask, infill_mask, bad_word_ids,
                    **guidance_kwargs
                )

                x = g_out['x']
                probs = g_out['probs']

            pred_ids = probs.argmax(-1)  # greedy decoding
            pred_ids = torch.where(infill_mask.squeeze(-1), pred_ids, infill_seed[None])

            if guidance_kwargs is not None:
                labels = self.network.guidance_score(
                    self.network.get_embeds(pred_ids), t, attn_mask
                ).cpu().numpy()
                pred_ids = (pred_ids.cpu().numpy(), labels)
            else:
                pred_ids = pred_ids.cpu().numpy()

            traj.append(pred_ids)

        if return_best:
            best_idx = np.argmax(np.stack([t[1] for t in traj], axis=1), axis=1)
            samples = np.stack([
                traj[idx][0][i] for i, idx in enumerate(best_idx)
            ], axis=0)
        else:
            #samples = traj[-1][0] #original code only takes the first sample for some reason
            samples = traj[-1]

        return samples

    def guided_sample(
            self,
            infill_seed,
            infill_mask,
            corrupt_mask,
            num_samples,
            classifier,  # <--- NEW: The classifier function/nn.Module
            guidance_scale=1.0,  # <--- NEW: Guidance strength λ
            bad_word_ids=None,
            # You can add other optional kwargs if needed
    ):
        """
        A variant of 'sample' that applies classifier guidance.
        """
        device = next(self.parameters()).device

        # 1) Prepare the initial ground-truth embeddings and shapes
        gt_vals = self.network.get_embeds(infill_seed)  # shape: [seq_len, embed_dim]
        infill_mask = infill_mask[None, :, None]  # shape: [1, seq_len, 1]
        corrupt_mask = corrupt_mask[None, :, None]  # shape: [1, seq_len, 1]
        gt_vals = gt_vals[None]  # shape: [1, seq_len, embed_dim]

        indices = list(range(self.noise_schedule.timesteps))[::-1]

        # 2) Create fully (or partially) noised initial x
        #    "noisy_gt": your infilled seed with some noise
        t0 = torch.tensor([indices[0]], device=device)
        noisy_gt = self.noise_schedule.q_sample(gt_vals, t0)
        noisy_gt = torch.where(corrupt_mask.bool(), noisy_gt, gt_vals)

        # 3) Sample random noise (batch = num_samples), then inpaint with 'noisy_gt'
        x = self.noise_schedule.noise_scale * torch.randn(
            (num_samples, gt_vals.shape[1], gt_vals.shape[-1]),
            device=device
        )
        x = torch.where(infill_mask.bool(), x, noisy_gt)

        # 4) Basic attention mask = fully on
        attn_mask = torch.ones(
            (num_samples, infill_mask.shape[1]),
            dtype=torch.bool, device=device
        )

        # 5) Main reverse diffusion loop
        traj = []
        for i in tqdm.tqdm(indices):
            t = torch.tensor([i] * num_samples, device=device)

            with torch.no_grad():
                # Forward pass of the diffusion model
                f_out = self.network.forward(
                    x, t, attn_mask=attn_mask
                )

                # Posterior sample with guidance
                # We'll modify "posterior_sample" to accept classifier & guidance_scale
                p_out = self.network.posterior_sample(
                    self.noise_schedule,
                    x,
                    t,
                    attn_mask,
                    infill_mask,
                    corrupt_mask,
                    gt_vals,
                    sequence_output=f_out['sequence_output'],
                    bad_word_ids=bad_word_ids,
                    classifier=classifier,  # <--- pass in
                    guidance_scale=guidance_scale  # <--- pass in
                )
                p_out['x_prev'] = x
                out = {**f_out, **p_out}

                # Update x
                x = out['x']
                probs = out['probs']

            # (Optional) get argmax tokens for logging
            pred_ids = probs.argmax(-1)
            pred_ids = torch.where(infill_mask.squeeze(-1).bool(), pred_ids, infill_seed[None])
            pred_ids = pred_ids.cpu().numpy()
            traj.append(pred_ids)

        # Return the final sample (or entire trajectory) as you prefer
        final_tokens = traj[-1]  # shape: [num_samples, seq_len]
        return final_tokens

    def NOS_C_sample(
            self,
            infill_seed,
            infill_mask,
            corrupt_mask,
            num_samples,
            classifier=None,  # <--- NEW: The classifier function/nn.Module
            guidance_kwargs=None,
            bad_word_ids=None,
    ):

        device = next(self.parameters()).device

        classifier = classifier.to(device)
        self.network.regression_head = classifier

        gt_vals = self.network.get_embeds(infill_seed)

        infill_mask = infill_mask[None, :, None]
        corrupt_mask = corrupt_mask[None, :, None]
        gt_vals = gt_vals[None]

        indices = list(range(self.noise_schedule.timesteps))[::-1]

        t = torch.tensor([indices[0]], device=device)
        noisy_gt = self.noise_schedule.q_sample(gt_vals, t)
        noisy_gt = torch.where(corrupt_mask.bool(), noisy_gt, gt_vals)

        x = self.noise_schedule.noise_scale * torch.randn(
            (num_samples, gt_vals.shape[1], gt_vals.shape[-1]), device=device
        )
        x = torch.where(infill_mask.bool(), x, noisy_gt)
        attn_mask = torch.ones(
            (num_samples, infill_mask.shape[1]), dtype=torch.bool, device=device
        )

        return_best = guidance_kwargs.pop("return_best", False) \
            if guidance_kwargs is not None else False

        traj = []
        for i in tqdm.tqdm(indices):
            t = torch.tensor([i] * num_samples, device=device)

            with torch.no_grad():
                f_out = self.network.forward(
                    x, t, attn_mask=attn_mask,
                )

                p_out = self.network.posterior_sample(
                    self.noise_schedule,
                    x, t, attn_mask, infill_mask,
                    corrupt_mask, gt_vals,
                    sequence_output=f_out['sequence_output'],
                    bad_word_ids=bad_word_ids
                )
                p_out['x_prev'] = x
                out = {**f_out, **p_out}

                x = out['x']
                probs = out['probs']

            if guidance_kwargs is not None:
                g_out = self.guidance_steps(
                    out, t, attn_mask, infill_mask, bad_word_ids,
                    **guidance_kwargs
                )

                x = g_out['x']
                probs = g_out['probs']

            pred_ids = probs.argmax(-1)  # greedy decoding
            pred_ids = torch.where(infill_mask.squeeze(-1).bool(), pred_ids, infill_seed[None])

            if guidance_kwargs is not None:
                labels = self.network.guidance_score(
                    self.network.get_embeds(pred_ids), t, attn_mask
                ).cpu().numpy()
                pred_ids = (pred_ids.cpu().numpy(), labels)
            else:
                pred_ids = pred_ids.cpu().numpy()

            traj.append(pred_ids)

        if return_best:
            best_idx = np.argmax(np.stack([t[1] for t in traj], axis=1), axis=1)
            samples = np.stack([
                traj[idx][0][i] for i, idx in enumerate(best_idx)
            ], axis=0)
        else:
            #samples = traj[-1][0] #original code only takes the first sample for some reason
            samples = traj[-1][0]
        
        return samples