from .base import DiscreteData
import pandas as pd
import torch
import numpy as np
import copy
from oracle.inference_oracle import inference_oracle
import os
from Bio import SeqIO

class ProteinPredictorDataset(DiscreteData):
    def __init__(self, data_config, tokenizer, n_random_init=0, n_max_mutations=None, from_fasta=None):
        
        self.data_config = data_config
        self.df = pd.read_csv(data_config.data_path)
        self.name = data_config.name

        self.n_max_mutations = n_max_mutations
        self.full_seq_string = data_config.full_seq
        # self.full_seq =  [tokenizer.tokenize(s) for s in self.full_seq_string] #change
        # self.full_seq = torch.from_numpy(np.array(self.full_seq)).int().squeeze(-1)

        self.seq_len = data_config.seq_len
        self.alphabet_size = data_config.alphabet_size

        if data_config.residues is None:
            self.residues = np.arange(self.seq_len)
        else:
            self.residues = [r-1 for r in data_config.residues] #convert to zero indexing
        self.n_residues = len(self.residues)

        #TODO: make this or use the one in EvoDiff
        #or could just use a dictionary
        # self.tokenizer = data_config.tokenizer
        self.tokenizer = tokenizer
        self.mask = torch.zeros(self.seq_len)
        self.mask[self.residues] = 1
        self.mapping = 'ACDEFGHIKLMNPQRSTVWYBZXJOU'
        
        #keep track of queried samples
        self.summary_df = None
        self.seqs = None
        self.round_seqs = None #seqs for current round
        self.round = 0
        self.y = None   

        if from_fasta is not None:
            #load presampled sequences from fasta with SeqIO
            seqs = []
            for record in SeqIO.parse(from_fasta, "fasta"):
                seqs.append(str(record.seq))
            self.update_data(seqs, n_needed=len(seqs))
        elif n_random_init > 0:
            #initialize data as random n_samples from the dataset
            if self.n_residues == 4:
                self.df = self.df.sample(n=n_random_init, random_state=data_config.seed)
                self.df["full_sequence"] = self.df['Combo'].apply(self.get_full_sequence)
                seqs = self.df["full_sequence"].values
                #self.y = torch.tensor(self.df['fitness'].values).float()
            # elif self.n_residues != self.seq_len:
            else:
                seqs = self.sample_random_variants(n_random_init)
                #_, self.y = inference_oracle(seqs, model_path=data_config.oracle_model_path)
                #sample 4 sites and
            # self.x = self.df['full_sequence'].apply(self.tokenizer.tokenize).tolist()
            # get sequence:
            if len(seqs) < n_random_init:
                print(f"Only found {len(seqs)} unique samples for random initialization")
            self.update_data(seqs, n_needed=n_random_init)

    ### TODO: adapt this from https://github.com/AI4PDLab/DPO_pLM/blob/main/DPO_pLM.py#L105 if using paired loss ###
    # def prepare_pairs(hf_dataset):
    #     """
    #     Prepare data for the paired mode of DPO, calculating whether a sample belongs above or below a certain threshold in fitness.
    #     """
    #     # Sort the dataset by weight in descending order
    #     sorted_dataset = hf_dataset.sort("weight", reverse=False)

    #     # Split the dataset into two halves
    #     mid_point = len(sorted_dataset) // 2
    #     first_half = sorted_dataset.select(range(mid_point))
    #     second_half = sorted_dataset.select(range(mid_point, len(sorted_dataset)))

    #     # Create pairs of positive and negative sequences
    #     pairs = []
    #     for pos_example, neg_example in zip(first_half, second_half):
    #         pairs.append({
    #             "positive_sequence": pos_example["sequence"],
    #             "negative_sequence": neg_example["sequence"],
    #         })

    #     return Dataset.from_list(pairs)
    def get_full_sequence(self, combo):
        """
        Get the full sequence from a combo of only the mutated positions.
        """
        #substitute relevant residues in the full sequence
        seq = list(self.full_seq_string)
        for i, r in enumerate(self.residues):
            seq[r] = combo[i]
        return ''.join(seq)

    def sample_random_variants(self, n_samples):
        """
        Sample n_samples random variants from the dataset.
        n_max_mutations: maximum number of mutations to make when generating variants
        """
        mutation_choices = list("ACDEFGHIKLMNPQRSTVWY")
        variants = []

        # if self.n_max_mutations is None:
        #     for i in range(n_samples):
        #         seq = ''.join(np.random.choice(mutation_choices, self.seq_len))
        #         variants.append(seq)
        # else:
        
        if self.n_max_mutations == None:
                residues = self.residues
        else:
            if self.n_max_mutations > self.n_residues:
                self.n_max_mutations = self.n_residues

        for i in range(n_samples):
            #this is done a bit different from the other sampling functions
            if self.n_max_mutations != None:
                residues = np.random.choice(self.residues, self.n_max_mutations, replace=False)

            #covert full_seq string to a list
            seq_list = list(self.full_seq_string)
            for r in residues:
                #seq is a string, so replace the residue at position r with a random residue
                seq_list[r] = np.random.choice(mutation_choices)
            seq = ''.join(seq_list)
            variants.append(seq)
        #variants = list(set(variants)) #only take unique
        return variants

    # def project(self, x):
    #     ### TODO: update this function if needed ###
    #     return x * self.mask.to(x.device) + self.full_seq.to(x.device) * (1 - self.mask).to(x.device)
    
    def get_length(self):
        return self.n_residues

    def get_dim(self):
        return self.alphabet_size

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, i):
        return (self.seqs[i],), self.y[i]#to follow model input as pretraining

    def update_data(self, new_seqs, n_needed, BO=False, round=0, unique_only=True):
        """
        Update the dataset with new_seqs.
        BO: whether or not to keep duplicates relative to all previously sampled sequences. In BO, you will not keep them. 
        unique_only: whether or not to keep only unique sequences in the current round. To evaluate diversity of generated sequences, switch to False.
        """
        #reset tracker if new round
        if round != self.round:
            self.round = round
            self.round_seqs = None

        new_seqs = list(new_seqs)
        
        #remove rows from new_seqs if it is already in self.seqs
        if self.seqs is not None:
            if BO:
                new_seqs = list(set(new_seqs))
                new_seqs = [s for s in new_seqs if s not in self.seqs]
            elif self.round_seqs is not None and unique_only:
                new_seqs = list(set(new_seqs))
                new_seqs = [s for s in new_seqs if s not in self.round_seqs] #only keep unique sequences relative to the current round

        n_samples_unique = len(new_seqs)
        #print(f"Reduced {n_samples} to {n_samples_unique} unique samples")

        if n_needed < n_samples_unique:
            new_seqs = new_seqs[:n_needed] #only take part of the batch if we're at the end of sampling

        self.seqs = new_seqs if self.seqs is None else self.seqs + new_seqs
        self.round_seqs = new_seqs if self.round_seqs is None else self.round_seqs + new_seqs

        #only take the mutated residues from self.residues
        new_combos = [''.join([seq[r] for r in self.residues]) for seq in new_seqs]

        #update fitness values
        if self.n_residues == 4:
            fitness = []
            for combo in new_combos:
                if combo not in self.df['Combo'].values:
                    print("Warning: combo not found in fitness dataset.")
                    v = 0 #all fitness values should be interpolated, so this should not be used
                else:
                    v = self.df[self.df['Combo'] == combo]['fitness'].values.item()
                fitness.append(v)
            y_new = torch.from_numpy(np.array(fitness)).float()
        else:
            if new_combos == []:
                y_new = torch.zeros(0)
            else:
                #_, y_new = inference_oracle(new_combos, protein=self.data_config.name, model_path=self.data_config.oracle_model_path)          

                # ── SKIP oracle if no oracle path configured ──────────────
                if self.data_config.oracle_model_path is None:
                    y_new = torch.zeros(len(new_combos))   # dummy scores
                    print("WARNING: No oracle model path set, using dummy scores")
                else:
                    _, y_new = inference_oracle(
                        new_combos,
                        protein=self.data_config.name,
                        model_path=self.data_config.oracle_model_path
                    )
                # ─────────────────────────────────────────────────────────
        
        self.y = y_new if self.y is None else torch.cat([self.y, y_new], dim=0)

        #update_summary_df
        df = pd.DataFrame({'sequence': new_seqs, 'round': round, 'fitness': y_new})
        if self.summary_df is None:
            self.summary_df = df
        else:
            self.summary_df = pd.concat([self.summary_df, df])

        return len(new_seqs)
    
    def save_data(self, save_path):
        """
        Save the dataset to a file.
        """
        self.summary_df.to_csv(save_path, index=False)