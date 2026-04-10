from .base import Algo
import torch
import tqdm
import torch.nn.functional as F
import numpy as np
    
class Classifier_Guidance_Inpaint(Algo):
    '''
        Implementation of Discrete Classifier Guidance with D3PM
        https://arxiv.org/abs/2406.01572
        forward_op is a predictor that outputs log p(y|x_t,t)
    '''
    def __init__(self, net, data_config,  forward_op=None, temperature=1, n_max_mutations=None, device='cuda'):
        super().__init__(net=net, forward_op=forward_op, data_config=data_config, n_max_mutations=n_max_mutations, device=device)
        if forward_op is not None:
            self.temperature = temperature
        else:
            self.temperature = 0.
        
    def update_model(self, classifier):
        self.forward_op = classifier.eval() #make sure the model is in eval mode

    def project(self, x):
        if self.full_seq is None:
            return x    # no projection needed for unconditional generation
        return x * self.mask + self.full_seq * (1 - self.mask)

    def inference(self, num_samples=1, verbose=False, detokenize=False, inpaint=False):
        timesteps = torch.linspace(self.net.timestep-1,1,int((self.net.timestep-1)/1), dtype=int) # iterate over reverse timesteps
        timesteps = tqdm.tqdm(timesteps) if verbose else timesteps
        x = self.net.get_start(num_samples)
        for t in timesteps:
            ### Simpler version doesn't seem to work right now. ###
            # x = self.net.p_sample(x, t, predictor_model=self.forward_op, guide_temp=self.temperature)

            x_next_prob = self.net.p_sample(x, t, hard=False)

            # x_next_prob: N x L x S, x: N x L in [0, S-1]
            # x_next_prob[x] -= 1 # prob difference
            x_next_prob[torch.arange(x_next_prob.shape[0])[:, None], torch.arange(x_next_prob.shape[1]), x] -= 1

            # guidance:
            # if unconditional, could skip computation of guided rates to speed up in future
            x_next_prob = self.net.get_guided_rates(self.forward_op, x, t, x_next_prob, guide_temp=self.temperature)
            ### TODO: check if mdlm and udlm work the same way ###

            # print(q_t.min(), x_next_prob.min())
            # x_next_prob = x_next_prob * (q_t**self.temperature)
            # print(x_next_prob.min())

            x_next_prob[torch.arange(x_next_prob.shape[0])[:, None], torch.arange(x_next_prob.shape[1]), x] += 1
            x_next_prob = x_next_prob.clamp(min=0)
            x_next_prob = x_next_prob / x_next_prob.sum(dim=-1, keepdim=True)
            x = torch.multinomial(x_next_prob.view(-1, x_next_prob.shape[-1]), num_samples=1).view(x_next_prob.shape[:-1])
            if inpaint:
                x = self.project(x) #project at every step of inference

        # project to only the positions under consideration
        # should maybe move this to within the timestep loop but then diversity is really bad
        x = self.project(x)

        # detokenize
        if detokenize:
            detokenized = [self.net.tokenizer.untokenize(s) for s in x]
            #replace special AA characters with a random ones
            #limit the number of mutations
            detokenized = self.project_sequences(detokenized)

            return x, detokenized
        else:
            return x


class Classifier_Guidance_Continuous(Algo):
    """
    Implementation of Classifier Guidance for a continuous diffusion model.
    forward_op is a predictor that outputs log p(y| x_t, t).
    """

    def __init__(self, net, data_config, forward_op=None, temperature=1.0, n_max_mutations=None, device='cuda'):
        super().__init__(net=net, forward_op=forward_op, data_config=data_config, n_max_mutations=n_max_mutations, device=device)

        if forward_op is not None:
            self.temperature = temperature #should rename this to be more accurate
        else:
            self.temperature = 0.

    def update_model(self, classifier):
        """ Update the classifier model used in forward_op. """
        self.forward_op = classifier

    def project(self, x):
        return x * self.mask + self.full_seq * (1 - self.mask)

    @torch.no_grad()
    def inference(self, num_samples=1, verbose=False, detokenize=False):
        """
        Generate samples from the continuous diffusion model with classifier guidance.
        We'll assume self.net has a method 'sample_guided(...)' that does the heavy lifting.
        """

        # x = self.net.guided_sample(
        #     forward_op=self.forward_op,
        #     temperature=self.temperature,
        #     mask=self.mask,  # optionally use the mask inside the sampler
        #     num_samples=num_samples,
        #     verbose=verbose
        # ) #fix to be guided_sample


        infill_seed = torch.randint(0, self.net.model.network.vocab_size, (self.seq_len,)).to(
            self.device)  # random seed of token ids for now
        # 1 if != pad, else 0
        infill_mask = (torch.ones(self.seq_len) != self.net.tokenizer.pad_id-100).to(
            self.device)  # switch 30 for self.net.tokenizer.pad_id

        #  changes infill_mask to only include the residues
        infill_mask = infill_mask * self.mask

        # corrupt_mask: 1 for real tokens, 0 for pad (Equivalent to "fully corrupt all real tokens")
        corrupt_mask = infill_mask.clone().to(self.device)  # (B, L), 1=corrupt, 0=pad
        
        x = self.net.guided_sample(
            infill_seed=infill_seed,
            infill_mask=infill_mask,
            corrupt_mask=corrupt_mask,
            num_samples=num_samples,
            classifier=self.forward_op,  # <--- NEW: The classifier function/nn.Module
            guidance_scale=self.temperature,  # <--- NEW: Guidance strength λ
            bad_word_ids=None,
        )

        # -- 2) (Optionally) force certain positions to remain unchanged after sampling. --
        x = torch.tensor(x) if isinstance(x, np.ndarray) else x
        x = self.project(x.to(device=self.device))

        # -- 3) Detokenize if needed. --
        if detokenize:
            # If x is integer tokens:
            detokenized = [self.net.tokenizer.untokenize(s) for s in x]

            detokenized = self.project_sequences(detokenized)
            return x, detokenized
        else:
            return x
