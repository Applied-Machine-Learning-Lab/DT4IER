"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
from math import log2
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

import collections
from torch import nn
from d2l import torch as d2l
import time

from torch.nn.modules.activation import Sigmoid

logger = logging.getLogger(__name__)

import numpy as np

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class GPTConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                                     .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        reward_mask = torch.ones_like(self.mask)
        for i in range(reward_mask.shape[-1]):
            reward_mask[:,:,::3,i] = 0
        reward_mask+=self.mask 
        att = att.masked_fill(reward_mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
        
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model_type = config.model_type

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # self.state_encoder = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # self.action_encoder = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.state_encoder = Seq2SeqEncoder(config.vocab_size, config.n_embd, config.n_embd, 2,
                        0.2)
        self.action_encoder = Seq2SeqEncoder(config.vocab_size, config.n_embd, config.n_embd, 2,
                        0.2)
        self.decoder = Seq2SeqDecoder(config.vocab_size, config.n_embd, config.n_embd, 2,
                        0.2)
        # self.net = d2l.EncoderDecoder(self.action_encoder, self.decoder)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)


        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        bucket_number = 100
        self.ret_emb = RewardEmbedding(config, bucket_number)
        self.state_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)


        #self.weight_gen_module = WeightGenerationModule(input_dim=256, hidden_dim=64, sequence_length=20)


    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = sequence + [0] * pad_len 
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)


        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        for i in (param_dict.keys() - union_params):
            no_decay.add(str(i))
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # state, action, and return
    def forward(self, states, actions, actions_neg, y_len, targets, rtgs,return_step, timesteps=None):
        device=rtgs.device
        state_embeddings=torch.zeros([states.shape[0], states.shape[1], 128])
        for i in range(states.shape[1]):
            states_seq=states[:,i,:].type(torch.long).squeeze(1)
            output, state = self.state_encoder(states_seq)
            context=state.permute(1, 0, 2) 
            state_embeddings[:,i,:]=context[:,-1,:]
        
        action_embeddings=torch.zeros([actions.shape[0], actions.shape[1], 128])
        state_allstep=[]
        for i in range(actions.shape[1]):
            action_seq=actions[:,i,:].type(torch.long).squeeze(1)
            output, state = self.action_encoder(action_seq)
            context=state.permute(1, 0, 2) 
            action_embeddings[:,i,:]=context[:,-1,:]
            state_allstep.append(state)
        #rtg_neg=torch.zeros([rtgs.shape[0]*8,rtgs.shape[1],rtgs.shape[2]],device=device)
        #for i in range(8):
        #    for j in range(rtgs.shape[0]):
        #        rtg_neg[i*rtgs.shape[0]+j,:-1,0]=rtgs[j,1:,0]+i
        #        rtg_neg[i*rtgs.shape[0]+j,-1,0]=rtgs[j,-1,0]
        
        
        #concatenated_embeddings = torch.cat((state_embeddings, action_embeddings), dim=2)

        #adaptive_weights = self.weight_gen_module(concatenated_embeddings).sum()
                

        if actions is not None and self.model_type == 'reward_conditioned': 
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            rtg_embeddings = rtg_embeddings.squeeze(2)
            #rtg_embeddings = rtg_embeddings.squeeze(2) * adaptive_weights
            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*3 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::3,:] = rtg_embeddings
            token_embeddings[:,1::3,:] = state_embeddings
            token_embeddings[:,2::3,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        elif actions is None and self.model_type == 'reward_conditioned': # only happens at very first timestep of evaluation
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            rtg_embeddings = rtg_embeddings.squeeze(2)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = rtg_embeddings # really just [:,0,:]
            token_embeddings[:,1::2,:] = state_embeddings # really just [:,1,:]
        elif actions is not None and self.model_type == 'naive':
            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = state_embeddings
            token_embeddings[:,1::2,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        elif actions is None and self.model_type == 'naive': # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, n_embd
        position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]
        token_embeddings=token_embeddings.to(device)


        token_neg_embeddings=torch.repeat_interleave(token_embeddings,8,0)
        #rtg_neg_embeddings = self.ret_emb(rtg_neg.type(torch.float32))
        #token_neg_embeddings[:,::3,:] = rtg_neg_embeddings
        token_all = torch.cat((token_embeddings, token_neg_embeddings), 0)
        position_all = torch.repeat_interleave(position_embeddings,9,0)


        x = self.drop(token_all + position_all)
        logits = self.blocks(x)
        if actions is not None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1::3, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1:, :]
        elif actions is not None and self.model_type == 'naive':
            logits = logits[:, ::2, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'naive':
            logits = logits # for completeness
        else:
            raise NotImplementedError()

        # if we are given some desired targets also calculate the loss
        loss_func = MaskedSoftmaxCELoss()    
        loss=[]

        for i in range(actions.shape[1]):
            logits_new=logits[:,i,:].squeeze(1)
            targets_seq=targets[:,i,:].type(torch.long).squeeze(1) 
            neg_seq=actions_neg[:,i,:].type(torch.long).squeeze(1) 
            pos_seq=actions[:,i,:].type(torch.long).squeeze(1) 
            bos = torch.tensor([5011] * targets_seq.shape[0]).reshape(-1, 1).to(device)
            dec_input = torch.cat([bos, targets_seq[:, :-1]], 1)
            logits_new_pos=logits_new[:actions.shape[0]]
            Y_hat,_,Y_emb = self.decoder(dec_input, state_allstep[i], logits_new_pos)

            dec_input_neg = torch.repeat_interleave(dec_input,8,0)
            state_neg = torch.repeat_interleave(state_allstep[i],8,1)
            logits_new_neg=logits_new[actions.shape[0]:]
            Y_hat_all,_,Y_emb_all = self.decoder(dec_input_neg, state_neg, logits_new_neg)
            # neg_seq_emb = Y_emb_all[actions.shape[0]:]
            neg_seq_emb = Y_emb_all
            y_len_step=y_len[:,i]

            loss_step1 = loss_func(Y_hat, targets_seq, y_len_step)
            loss_step1=loss_step1.mean()
            pos_seq_emb=self.action_encoder.embedding(pos_seq)
            pos_score = bleu_emb_pos(Y_emb, pos_seq_emb, y_len_step)
            return_step_one=return_step[:,i]

            neg_score = bleu_emb(Y_emb, neg_seq_emb, y_len_step, return_step_one)
            loss_step=loss_step1
            loss.append(loss_step)

        loss_mean=sum(loss)/len(loss)
        return logits[:actions.shape[0]], loss_mean
    #@save

    def predict_seq2seq(self, states, actions, actions_len, targets, rtgs, r_step, timesteps, num_steps,
                        device, save_attention_weights=False):
        # 在预测时将net设置为评估模式
        device=rtgs.device
        state_embeddings=torch.zeros([states.shape[0], states.shape[1], 128])
        for i in range(states.shape[1]):
            states_seq=states[:,i,:].type(torch.long).squeeze(1)
            output, state = self.state_encoder(states_seq)
            context=state.permute(1, 0, 2) 
            state_embeddings[:,i,:]=context[:,-1,:]

        action_embeddings=torch.zeros([actions.shape[0], actions.shape[1], 128])
        state_allstep=[]
        for i in range(actions.shape[1]):
            action_seq=actions[:,i,:].type(torch.long).squeeze(1)
            output, state = self.action_encoder(action_seq)
            context=state.permute(1, 0, 2) 
            action_embeddings[:,i,:]=context[:,-1,:]
            state_allstep.append(state)
            

        if actions is not None and self.model_type == 'reward_conditioned': 
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            rtg_embeddings = rtg_embeddings.squeeze(2)
            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*3 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::3,:] = rtg_embeddings
            token_embeddings[:,1::3,:] = state_embeddings
            token_embeddings[:,2::3,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        elif actions is None and self.model_type == 'reward_conditioned': # only happens at very first timestep of evaluation
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            rtg_embeddings = rtg_embeddings.squeeze(2)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = rtg_embeddings # really just [:,0,:]
            token_embeddings[:,1::2,:] = state_embeddings # really just [:,1,:]
        elif actions is not None and self.model_type == 'naive':
            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = state_embeddings
            token_embeddings[:,1::2,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        elif actions is None and self.model_type == 'naive': # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, n_embd
        position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]
        token_embeddings=token_embeddings.to(device)
        x = self.drop(token_embeddings + position_embeddings)
        logits = self.blocks(x)

        if actions is not None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1::3, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1:, :]
        elif actions is not None and self.model_type == 'naive':
            logits = logits[:, ::2, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'naive':
            logits = logits # for completeness
        else:
            raise NotImplementedError()
        loss_func = MaskedSoftmaxCELoss()
        y_pred=torch.zeros_like(actions)
        for j in range(actions.shape[1]):
            logits_new=logits[:,j,:].squeeze(1)
            targets_seq=targets[:,j,:].type(torch.long).squeeze(1)

            #batch
            dec_state=state_allstep[j]
            output_seq, attention_weight_seq = [], []
            dec_X = torch.tensor([5011] * targets_seq.shape[0]).reshape(-1, 1).to(device)
            for k in range(num_steps):
                Y, dec_state, Y_emb = self.decoder(dec_X, dec_state, logits_new)
                dec_X = Y.argmax(dim=2)
                pred = dec_X.type(torch.int32)
                if save_attention_weights:
                    attention_weight_seq.append(self.decoder.attention_weights)
                y_pred[:,j,k]=pred.squeeze(1)
        score = []
        for i in range(y_pred.shape[0]):
            score_i = bleu_seq(y_pred[i,-1,:],targets[i,-1,:])
            score.append(score_i)
        score = sum(score)/len(score)
        
        rouge_score = []
        for i in range(y_pred.shape[0]):
            rouge_score_i = ROUGE(y_pred[i,-1,:],targets[i,-1,:], actions_len[i,-1])
            rouge_score.append(rouge_score_i)
        rouge_score = sum(rouge_score)/len(rouge_score)
        
        hr_score = []
        for i in range(y_pred.shape[0]):
            hr_i = hit_rate(y_pred[i,-1,:],targets[i,-1,:],20)
            hr_score.append(hr_i)
        hr_score = sum(hr_score)/len(hr_score)
        
        ndcg_score = []
        for i in range(y_pred.shape[0]):
            ndcg_i = ndcg(y_pred[i,-1,:],targets[i,-1,:],20)
            ndcg_score.append(ndcg_i)
        ndcg_score = sum(ndcg_score)/len(ndcg_score)
        
        recall = []
        for i in range(y_pred.shape[0]):
            recall_i = compute_recall(y_pred[i,-1,:],targets[i,-1,:])
            recall.append(recall_i)
        recall = sum(recall)/len(recall)
        
        precision = []
        for i in range(y_pred.shape[0]):
            precision_i = compute_precision(y_pred[i,-1,:],targets[i,-1,:])
            precision.append(precision_i)
        precision = sum(precision)/len(precision)
        
        #ndcg = []
        #for i in range(y_pred.shape[0]):
        #    ndcg_i = ndcg(y_pred[i,-1,:],targets[i,-1,:], actions_len[i,-1])
        #    ndcg.append(ndcg_i)
        #ndcg = sum(ndcg)/len(ndcg)
        
        return score, rouge_score, hr_score, ndcg_score, precision


# Reward Embedding Block
class RewardEmbedding(nn.Module):
    def __init__(self, config, bucket_number):
        super().__init__()
        self.bucket = nn.Sequential(nn.Linear(2, config.n_embd))
        self.ret_emb_score = nn.Sequential(nn.Linear(2, bucket_number, bias=False), nn.LeakyReLU())
        self.res = nn.Linear(bucket_number, bucket_number, bias=False)
        self.temp = nn.Sequential(
            nn.Linear(2, bucket_number, bias=False), 
            nn.LeakyReLU(), 
            nn.Linear(bucket_number, bucket_number, bias=False),
            nn.Sigmoid()
            )


    def forward(self, x, layer_past=None):
        bucket_value = torch.arange(0, 1400, 7).to(x.device).reshape(100,2).type(torch.float32)
        Meta_emb = self.bucket(bucket_value)
        t = self.temp(x)
        x = self.ret_emb_score(x)
        x = x + self.res(x)
        max_value,_ = torch.max(x, dim=2, keepdim=True)
        x = torch.exp(x - max_value)
        soft_sum = torch.sum(x, dim=2).unsqueeze(2)
        x = x / soft_sum
        x = torch.einsum('ncdk,km->ncdm', [x, Meta_emb])
        return x



class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


#@save
class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        #print("Max index in X:", X.max().item())
       # print("Min index in X:", X.min().item())
        #print("Embedding weight shape:", self.embedding.weight.shape)

        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        return output, state

class Seq2SeqDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state, logits_new):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        context = logits_new.unsqueeze(0).repeat(X.shape[0], 1, 1)
        # context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        state=state.contiguous()
        output, state = self.rnn(X_and_context, state)
        output_emb = output.permute(1, 0, 2)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        return output, state, output_emb

#@save
def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

#@save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss
    

def bleu(pred_seq, label_seq, y_len):  #@save
    retA=0
    for i in range(pred_seq.shape[0]):
        for j in range(y_len[i]):
            if pred_seq[i,j]==label_seq[i,j]:
                retA+=1
                break
    score=retA/sum(y_len)
    return score



def bleu_emb_pos(pred_seq, label_seq, y_len):  #@save
    retA=0

    score_step=torch.abs(torch.cosine_similarity(pred_seq, label_seq,dim=-1))

    for i in range(pred_seq.shape[0]):
        score_batch=torch.sum(score_step[i,:y_len[i]])/y_len[i]

        retA+=score_batch
    score=retA/y_len.shape[0]
    return score


def bleu_emb(pred_seq, label_seq, y_len,return_step_one):  #@save
    retA=0

    score_neg=torch.zeros([8,pred_seq.shape[0],pred_seq.shape[1]],device=y_len.device)
    for i in range(8):
        score_neg[i,:,:]=torch.abs(torch.cosine_similarity(pred_seq, label_seq[i*pred_seq.shape[0]:(i+1)*pred_seq.shape[0]],dim=-1)) * (15-2*i) / 8
        if i > 4:
            score_neg[i,:,:]=torch.abs(torch.cosine_similarity(pred_seq, label_seq[i*pred_seq.shape[0]:(i+1)*pred_seq.shape[0]],dim=-1)) * 0


    for i in range(pred_seq.shape[0]):
        score_batch=torch.sum(score_neg[:,i,:y_len[i]],dim=1)/y_len[i]
        score_batch_neg=(torch.sum(score_batch)-score_batch[int(return_step_one[i].item())])/7

        retA+=score_batch_neg
    score=retA/y_len.shape[0]
    return score

def InfoNCE(pred_seq, pos_seq, neg_seq, y_len,return_step_one):
    
    score_neg=torch.zeros([8,pred_seq.shape[0],pred_seq.shape[1]],device=y_len.device)
    for i in range(8):
        score_neg[i,:,:]=torch.cosine_similarity(pred_seq, neg_seq[i*pred_seq.shape[0]:(i+1)*pred_seq.shape[0]],dim=-1)
    # score_step=torch.abs(torch.cosine_similarity(pred_seq, label_seq,dim=-1))
    l_neg = torch.zeros([pred_seq.shape[0],7],device=y_len.device)
    for i in range(pred_seq.shape[0]):
        score_batch=torch.sum(score_neg[:,i,:y_len[i]],dim=1)/y_len[i]
        index = list(set(range(8))-set([int(return_step_one[i].item())]))
        l_neg[i] = score_batch[index]
    
    pos_score_step=torch.cosine_similarity(pred_seq, pos_seq,dim=-1)
    l_pos = torch.zeros([pred_seq.shape[0],1],device=y_len.device)
    for i in range(pred_seq.shape[0]):
        pos_score_batch=torch.sum(pos_score_step[i,:y_len[i]])/y_len[i]
        l_pos[i] = pos_score_batch
    logits = torch.cat([l_pos, l_neg], dim=1)
    # apply temperature
    T = 0.07
    logits /= T
	
    # labels: positive key indicators
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    loss = criterion(logits, labels)
    return loss

class WeightGenerationModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, sequence_length):
        super(WeightGenerationModule, self).__init__()
        # Define the layers of the module
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, sequence_length)

    def forward(self, concatenated_embeddings):
        # Forward pass through the network
        x = F.relu(self.fc1(concatenated_embeddings))
        x = F.relu(self.fc2(x))
        weights = F.softmax(self.fc3(x), dim=1)
        return weights
        

def ROUGE(y_pred,y,y_len):
    score_sum=0
    for i in range(y_len):
        for j in range(y_pred.shape[0]):
            if y[i]==y_pred[j]:
                score_sum+=1
                break
    score=score_sum/y_len.item()
    return score

def bleu_seq(y_pred,y):
    score_sum=0
    for i in range(y_pred.shape[0]):
        for j in range(y.shape[0]):
            if y_pred[i]==y[j]:
                score_sum+=1
                break
    score=score_sum/y_pred.shape[0]
    return score

def hit_rate(y_pred, y, k=20):
    hr = 0
    for i in range(k):
        for j in range(k):
            if y_pred[i]==y[j]:
                hr+=1
                break
    hr=hr/k
    return hr

def ndcg(y_pred, y, k=20):
    dcg = 0
    idcg = 0
    for i in range(k):
        for j in range(k):
            if y_pred[i] == y[j]:
                dcg += 1.0 / log2(i + 2)
                break

    # Perfect ranking would rank all relevant items at the top
    for i in range(min(k, len(y))):
        idcg += 1.0 / log2(i + 2)
    
    if idcg == 0:  # Handle case where IDCG is 0 to avoid division by zero
        return 0

    return dcg / idcg

def compute_recall(y_pred, states):
    # Create binary ground truth labels for actions
    y_true = torch.where((states.unsqueeze(0) == y_pred.unsqueeze(1)) & (states != 0), 1, 0).sum(dim=1)
    
    # Compute the total number of positive instances in the ground truth
    true_positives = y_true.sum().item()
    
    # Compute recall
    total_actual_positives = len(states) - (states == 0).sum().item()
    recall = true_positives / total_actual_positives if total_actual_positives != 0 else 0

    return recall


def compute_precision(y_pred, states):
    # Create binary ground truth labels for actions
    y_true = torch.where((states.unsqueeze(0) == y_pred.unsqueeze(1)) & (states != 0), 1, 0).sum(dim=1)
    
    # Compute the total number of true positives
    true_positives = y_true.sum().item()

    # Compute precision
    total_predicted_positives = len(y_pred) - (y_pred == 0).sum().item()  # Remove padding zeros from count
    precision = true_positives / total_predicted_positives if total_predicted_positives != 0 else 0

    return precision

