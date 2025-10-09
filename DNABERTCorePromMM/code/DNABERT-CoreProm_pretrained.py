import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple
from copy import deepcopy
from multiprocessing import Pool
import torch.nn as nn

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers.modeling_bert import BertEmbeddings, BertOnlyMLMHead, BertPreTrainedModel, BertModel
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    DNATokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    BertPreTrainedModel,
    BertModel
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

class ChemicalBertEmbeddings(BertEmbeddings):
    def __init__(self, config, chemical_embeddings: nn.ModuleDict, chemical_properties: dict, dinucleotide_to_idx: dict):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.sequence_prop_projection = nn.Linear(10, config.hidden_size) 
        self.chemical_embeddings = chemical_embeddings
        self.chemical_properties = chemical_properties
        self.dinucleotide_to_idx = dinucleotide_to_idx
        self.tokenizer = None
        
    def get_sequence_properties(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        if self.tokenizer is None:
            raise ValueError("tokenizer not found")
        
        # getting mask token id 
        # mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
            
        sequence_props = torch.zeros(batch_size, seq_length, 10, device=device)
        
        for b in range(batch_size):
            tokens = [self.tokenizer.convert_ids_to_tokens(int(x.item())) for x in input_ids[b]]
            for i in range(seq_length):
                # skipping for mask or pad token 
                # if input_ids[b, i] == mask_token_id:
                #     continue
                if tokens[i] == self.tokenizer.pad_token:
                    continue
                token_seq = tokens[i].replace(" ", "")
                if token_seq and not token_seq.startswith("["):
                    props = compute_sequence_properties(token_seq)
                    sequence_props[b, i] = props
                    
        return sequence_props

    def get_dinucleotide_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        chemical_embeds = torch.zeros(batch_size, seq_length, self.hidden_size, device=device)

        if self.tokenizer is None:
            raise ValueError("tokenizer not found")
        
        # getting mask token id 
        # mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        for b in range(batch_size):
            tokens = [self.tokenizer.convert_ids_to_tokens(int(x.item())) for x in input_ids[b]]
            for i in range(seq_length):
                # skip if the token is a mask token 
                # if input_ids[b, i] == mask_token_id:
                #     continue
                token = tokens[i]
                dinucs = []
                if len(token) >= 2:
                    for j in range(len(token) - 1):
                        dinucs.append(token[j:j+2])
                if i < seq_length - 1:
                    next_token = tokens[i+1]
                    if len(token) > 0 and len(next_token) > 0:
                        dinucs.append(token[-1] + next_token[0])
                embed_sum = torch.zeros(self.hidden_size, device=device)
                count = 0
                for dinuc in dinucs:
                    for prop_name, prop_values in self.chemical_properties.items():
                        if dinuc in prop_values:
                            idx = self.dinucleotide_to_idx[dinuc]
                            embed_sum += self.chemical_embeddings[prop_name](torch.tensor(idx, device=device))
                            count += 1
                if count > 0:
                    chemical_embeds[b, i] = embed_sum / count
        return chemical_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        token_type_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if (input_ids is not None) == (inputs_embeds is not None):
            raise ValueError('must specify either input_ids or inputs_embeds')
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
    
        if token_type_ids is None:
            if hasattr(self, 'token_type_ids'):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                token_type_ids = buffered_token_type_ids.expand(input_shape[0], seq_length)
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.word_embeddings.weight.device)
    
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
    
        # Get chemical embeddings
        chemical_embeds = self.get_dinucleotide_embeddings(input_ids)
    
        # Get sequence property embeddings
        sequence_props = self.get_sequence_properties(input_ids)
        sequence_prop_embeddings = self.sequence_prop_projection(sequence_props)
    
        # Zero out chemical and sequence property embeddings for masked tokens
        mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        mask_positions = (input_ids == mask_token_id)  # Find positions of masked tokens
    
        # Apply mask to chemical and sequence property embeddings
        chemical_embeds[mask_positions] = 0  # Zero out chemical embeddings for masked tokens
        sequence_prop_embeddings[mask_positions] = 0  # Zero out sequence property embeddings for masked tokens
    
        # Combine embeddings
        embeddings = embeddings + chemical_embeds + sequence_prop_embeddings
    
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ChemicalBertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.chemical_embeddings = nn.ModuleDict({
            'protein_dna_twist': nn.Embedding(16, config.hidden_size),
            'z_dna_stabilizing_energy': nn.Embedding(16, config.hidden_size),
            'A_philicity': nn.Embedding(16, config.hidden_size),
            'Base_stacking': nn.Embedding(16, config.hidden_size),
            'B_DNA_twist': nn.Embedding(16, config.hidden_size),
            'DNA_bending_stiffness': nn.Embedding(16, config.hidden_size),
            'DNA_denaturation': nn.Embedding(16, config.hidden_size),
            'Duplex_stability_disrupt_energy': nn.Embedding(16, config.hidden_size),
            'Duplex_stability_free_energy': nn.Embedding(16, config.hidden_size),
            'Propeller_twist': nn.Embedding(16, config.hidden_size),
            'Helical_slide': nn.Embedding(16, config.hidden_size),
            'Helical_shift': nn.Embedding(16, config.hidden_size),
            'Helical_twist': nn.Embedding(16, config.hidden_size),
            'Helical_roll': nn.Embedding(16, config.hidden_size),
            'Helical_tilt': nn.Embedding(16, config.hidden_size),
            'Helical_rise': nn.Embedding(16, config.hidden_size),
            'protein_induced_deformability': nn.Embedding(16, config.hidden_size)
        })
        
        self.dinucleotide_to_idx = {
            'AA': 0, 'AC': 1, 'AG': 2, 'AT': 3,
            'CA': 4, 'CC': 5, 'CG': 6, 'CT': 7,
            'GA': 8, 'GC': 9, 'GG': 10, 'GT': 11,
            'TA': 12, 'TC': 13, 'TG': 14, 'TT': 15
        }
        
        self.chemical_properties = {
            'protein_dna_twist': {
                'AA': 0.3647, 'AC': -0.4824, 'AG': -0.3882, 'AT': -1.0000,
                'CA': 0.8824, 'CC': -0.1529, 'CG': 0.6000, 'CT': -0.3882,
                'GA': 0.6471, 'GC': 0.0118, 'GG': -0.1529, 'GT': -0.4824,
                'TA': 1.0000, 'TC': 0.6471, 'TG': 0.8824, 'TT': 0.3647
            },
            'z_dna_stabilizing_energy': {
                'AA': 0.2308, 'AC': 0.5000, 'AG': 0.0385, 'AT': 1.0000,
                'CA': -0.7692, 'CC': -0.3462, 'CG': -1.0000, 'CT': 0.0385,
                'GA': 0.0385, 'GC': 0.2692, 'GG': -0.3462, 'GT': 0.5000,
                'TA': -0.3077, 'TC': 0.0385, 'TG': -0.7692, 'TT': 0.2308
            },
            'A_philicity': {
                'AA': 0.8462, 'AC': -1.0000, 'AG': -0.5604, 'AT': -0.0110,
                'CA': 1.0000, 'CC': -0.8681, 'CG': -0.1429, 'CT': -0.5604,
                'GA': 0.8681, 'GC': 0.3187, 'GG': -0.8681, 'GT': -1.0000,
                'TA': 0.3187, 'TC': 0.8681, 'TG': 1.0000, 'TT': 0.8462
            },
            'Base_stacking': {
                'AA': 0.7124, 'AC': -0.2412, 'AG': 0.4508, 'AT': 0.4898,
                'CA': 0.4898, 'CC': 0.1763, 'CG': -0.0891, 'CT': 0.4508,
                'GA': -0.1113, 'GC': -1.0000, 'GG': 0.1763, 'GT': -0.2412,
                'TA': 1.0000, 'TC': -0.1113, 'TG': 0.4898, 'TT': 0.7124
            },
            'B_DNA_twist': {
                'AA': -0.2222, 'AC': -0.6032, 'AG': -1.0000, 'AT': 1.0000,
                'CA': 0.1270, 'CC': -0.2540, 'CG': -0.8889, 'CT': -1.0000,
                'GA': 0.4286, 'GC': 0.2381, 'GG': -0.2540, 'GT': -0.6032,
                'TA': -0.8413, 'TC': 0.4286, 'TG': 0.1270, 'TT': -0.2222
            },
            'DNA_bending_stiffness': {
                'AA': -0.7273, 'AC': -0.2727, 'AG': -0.2727, 'AT': -1.0000,
                'CA': -0.2727, 'CC': 1.0000, 'CG': 0.1818, 'CT': -0.2727,
                'GA': -0.2727, 'GC': 0.1818, 'GG': 1.0000, 'GT': -0.2727,
                'TA': -1.0000, 'TC': -0.2727, 'TG': -0.2727, 'TT': -0.7273
            },
            'DNA_denaturation': {
                'AA': -0.6174, 'AC': 0.3693, 'AG': -0.1832, 'AT': -0.4825,
                'CA': -0.6545, 'CC': 0.1479, 'CG': -0.0964, 'CT': -0.1832,
                'GA': -0.3019, 'GC': 1.0000, 'GG': 0.1479, 'GT': 0.3693,
                'TA': -1.0000, 'TC': -0.3019, 'TG': -0.6545, 'TT': -0.6174
            },
            'Duplex_stability_disrupt_energy': {
                'AA': -0.2593, 'AC': -0.7037, 'AG': -0.4815, 'AT': -1.0000,
                'CA': -0.2593, 'CC': 0.6296, 'CG': 1.0000, 'CT': -0.4815,
                'GA': -0.4815, 'GC': 0.6296, 'GG': 0.6296, 'GT': -0.7037,
                'TA': -0.5556, 'TC': -0.4815, 'TG': -0.2593, 'TT': -0.2593
            },
            'Duplex_stability_free_energy': {
                'AA': 0.6842, 'AC': 0.3684, 'AG': 0.3684, 'AT': 1.0000,
                'CA': 0.1579, 'CC': -0.4737, 'CG': -1.0000, 'CT': 0.3684,
                'GA': 0.3684, 'GC': -0.4737, 'GG': -0.4737, 'GT': 0.3684,
                'TA': 1.0000, 'TC': 0.3684, 'TG': 0.1579, 'TT': 0.6842
            },
            'Propeller_twist': {
                'AA': -1.0000, 'AC': 0.0540, 'AG': -0.1166, 'AT': -0.3081,
                'CA': 0.7460, 'CC': 1.0000, 'CG': 0.6360, 'CT': -0.1166,
                'GA': -0.0180, 'GC': 0.4370, 'GG': 1.0000, 'GT': 0.0540,
                'TA': 0.2910, 'TC': -0.0180, 'TG': 0.7460, 'TT': -1.0000
            },
            'Helical_slide': {
                'AA': 0.3793, 'AC': 0.4737, 'AG': 0.0529, 'AT': 1.0000,
                'CA': -0.6057, 'CC': 0.8524, 'CG': -1.0000, 'CT': 0.0529,
                'GA': -0.2414, 'GC': 0.0573, 'GG': 0.8524, 'GT': 0.4737,
                'TA': -0.6316, 'TC': -0.2414, 'TG': -0.6057, 'TT': 0.3793
            },
            'Helical_shift': {
                'AA': 0.9770, 'AC': -0.6262, 'AG': 0.1726, 'AT': -1.0000,
                'CA': 0.0297, 'CC': 1.0000, 'CG': -0.6125, 'CT': 0.1726,
                'GA': -0.4005, 'GC': 0.4372, 'GG': 1.0000, 'GT': -0.6262,
                'TA': -0.1496, 'TC': -0.4005, 'TG': 0.0297, 'TT': 0.9770
            },
            'Helical_twist': {
                'AA': 0.7993, 'AC': 1.0000, 'AG': 0.6559, 'AT': 0.8136,
                'CA': -1.0000, 'CC': 0.9498, 'CG': -0.8781, 'CT': 0.6559,
                'GA': 0.5197, 'GC': 0.5125, 'GG': 0.9498, 'GT': 1.0000,
                'TA': 0.0538, 'TC': 0.5197, 'TG': -1.0000, 'TT': 0.7993
            },
            'Helical_roll': {
                'AA': 0.4245, 'AC': 0.8849, 'AG': 0.3094, 'AT': 0.9568,
                'CA': -0.3094, 'CC': 0.5108, 'CG': -0.7554, 'CT': 0.3094,
                'GA': 0.0791, 'GC': 1.0000, 'GG': 0.5108, 'GT': 0.8849,
                'TA': -1.0000, 'TC': 0.0791, 'TG': -0.3094, 'TT': 0.4245
            },
            'Helical_tilt': {
                'AA': 0.7041, 'AC': 0.9645, 'AG': 0.4911, 'AT': 0.8817,
                'CA': -0.6450, 'CC': 1.0000, 'CG': -0.6095, 'CT': 0.4911,
                'GA': 0.7396, 'GC': 0.7870, 'GG': 1.0000, 'GT': 0.9645,
                'TA': -1.0000, 'TC': 0.7396, 'TG': -0.6450, 'TT': 0.7041
            },
            'Helical_rise': {
                'AA': 0.0577, 'AC': 0.7980, 'AG': -0.3460, 'AT': 1.0000,
                'CA': -0.3778, 'CC': -0.0359, 'CG': -1.0000, 'CT': -0.3460,
                'GA': 0.2897, 'GC': 0.9285, 'GG': -0.0359, 'GT': 0.7980,
                'TA': -0.7823, 'TC': 0.2897, 'TG': -0.3778, 'TT': 0.0577
            },
            'protein_induced_deformability': {
                'AA': 1.0000, 'AC': 0.5619, 'AG': -0.1048, 'AT': -0.9048,
                'CA': -0.1429, 'CC': -0.7524, 'CG': -0.4476, 'CT': -1.0000,
                'GA': -0.8667, 'GC': -0.5429, 'GG': -0.1429, 'GT': -0.9048,
                'TA': -0.8667, 'TC': -0.4476, 'TG': 0.5619, 'TT': -0.7714
            }
        }
        
        self.bert.embeddings = ChemicalBertEmbeddings(
            config,
            chemical_embeddings=self.chemical_embeddings,
            chemical_properties=self.chemical_properties,
            dinucleotide_to_idx=self.dinucleotide_to_idx
        )
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()
    
    @property
    def tokenizer(self):
        return self._tokenizer
    
    @tokenizer.setter
    def tokenizer(self, tokenizer):
        self._tokenizer = tokenizer
        self.bert.embeddings.tokenizer = tokenizer


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]
        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (loss,) + outputs
        return outputs


MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    # "dna": (BertConfig, BertForMaskedLM, DNATokenizer),
    "dna": (BertConfig, ChemicalBertForMaskedLM, DNATokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
}

MASK_LIST = {
    "3": [-1, 1],
    "4": [-1, 1, 2],
    "5": [-2, -1, 1, 2],
    "6": [-2, -1, 1, 2, 3]
}

def compute_sequence_properties(sequence: str) -> torch.Tensor:
    seq_len = len(sequence)
    if seq_len == 0:
        return torch.zeros(10, dtype=torch.float)
    
    A_count = sequence.count('A')
    C_count = sequence.count('C')
    G_count = sequence.count('G')
    T_count = sequence.count('T')
    A_fract = A_count / seq_len
    C_fract = C_count / seq_len
    G_fract = G_count / seq_len
    T_fract = T_count / seq_len

    purpyr_fract = (A_count + G_count - C_count - T_count) / seq_len
    amke_fract = (A_count + C_count - G_count - T_count) / seq_len
    west_fract = (A_count + T_count - C_count - G_count) / seq_len
    
    if seq_len > 1:
        CG_count = sequence.count("CG")
        GC_count = sequence.count("GC")
        cpg1 = (2 * (CG_count + GC_count)) / (seq_len - 1)
    else:
        cpg1 = 0.0
    
    if seq_len > 2:
        ACG_count = sequence.count("ACG")
        AGC_count = sequence.count("AGC")
        CAG_count = sequence.count("CAG")
        CCG_count = sequence.count("CCG")
        CGA_count = sequence.count("CGA")
        CGC_count = sequence.count("CGC")
        CGG_count = sequence.count("CGG")
        CGT_count = sequence.count("CGT")
        CTG_count = sequence.count("CTG")
        GAC_count = sequence.count("GAC")
        GCA_count = sequence.count("GCA")
        GCC_count = sequence.count("GCC")
        GCG_count = sequence.count("GCG")
        GCT_count = sequence.count("GCT")
        GGC_count = sequence.count("GGC")
        GTC_count = sequence.count("GTC")
        TCG_count = sequence.count("TCG")
        TGC_count = sequence.count("TGC")
        cpg2 = (ACG_count + AGC_count + CAG_count + CCG_count + CGA_count + CGC_count +
                2 * CGG_count + CGT_count + CTG_count + GAC_count + GCA_count +
                2 * GCC_count + GCG_count + GCT_count + 2 * GGC_count +
                GTC_count + TCG_count + TGC_count) / (seq_len - 2)
        cpg3 = (4 * CAG_count + CCG_count + CGG_count + 4 * CTG_count +
                4 * GAC_count + GCC_count + GGC_count + 4 * GTC_count) / (seq_len - 2)
    else:
        cpg2 = 0.0
        cpg3 = 0.0

    features = [A_fract, C_fract, G_fract, T_fract, purpyr_fract, amke_fract, west_fract, cpg1, cpg2, cpg3]
    return torch.tensor(features, dtype=torch.float)

class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

def convert_line_to_example(tokenizer, lines, max_length, add_special_tokens=True):
    examples = tokenizer.batch_encode_plus(lines, add_special_tokens=add_special_tokens, max_length=max_length)["input_ids"]
    return examples

class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", file_path)

            with open(file_path, encoding="utf-8") as f:
                lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            
            if args.n_process == 1:
                self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]
            else:
                n_proc = args.n_process
                p = Pool(n_proc)
                indexes = [0]
                len_slice = int(len(lines)/n_proc)
                for i in range(1, n_proc+1):
                    if i != n_proc:
                        indexes.append(len_slice*(i))
                    else:
                        indexes.append(len(lines))
                results = []
                for i in range(n_proc):
                    results.append(p.apply_async(convert_line_to_example,[tokenizer, lines[indexes[i]:indexes[i+1]], block_size,]))
                    print(str(i) + " start")
                p.close() 
                p.join()

                self.examples = []
                for result in results:
                    ids = result.get()
                    self.examples.extend(ids)

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted

def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    
    mask_list = MASK_LIST[tokenizer.kmer]

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()

    # change masked indices
    masks = deepcopy(masked_indices)
    for i, masked_index in enumerate(masks):
        end = torch.where(probability_matrix[i]!=0)[0].tolist()[-1]
        mask_centers = set(torch.where(masked_index==1)[0].tolist())
        new_centers = deepcopy(mask_centers)
        for center in mask_centers:
            for mask_number in mask_list:
                current_index = center + mask_number
                if current_index <= end and current_index >= 1:
                    new_centers.add(current_index)
        new_centers = list(new_centers)
        masked_indices[i][new_centers] = True
    

    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(args.beta1,args.beta2))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    model.tokenizer = tokenizer
    
    # multi-gpu training (should be after apex fp16 initialization)    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility
    ids_set = {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0}
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
            # print(inputs.shape)
            # print(inputs)
            # for i in range(len(inputs)):
            #     for j in range(len(inputs[i])):
            #         ids_set[str(int(inputs[i][j]))] += 1
            # print(ids_set)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write(str(float(perplexity)) + "\n")
            # writer.write("%s = %s\n" % (key, str(result[key])))

    return result

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--beta1", default=0.9, type=float, help="Beta1 for Adam optimizer.")
    parser.add_argument("--beta2", default=0.999, type=float, help="Beta2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--n_process", type=int, default=1, help="")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()


    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    # text = "C G A T A T A G"
    # print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)))

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_class(config=config)

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.tokenizer = tokenizer
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results

if __name__ == "__main__":
    main()
