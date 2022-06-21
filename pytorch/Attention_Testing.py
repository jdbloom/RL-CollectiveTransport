import numpy as np
import pandas as pd
import argparse
import json
import os
import re
import math
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt



class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed Size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.query = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.fc_out = nn.Linear(self.heads*self.head_dim, embed_size)

        self.softmax = nn.Softmax(dim = 3)

    def forward(self, values, keys, query, mask=None):
        # mask is only for the decoder
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = T.einsum("nqhd, nkhd->nhqk", [queries, keys])
        # query shape = N, query_len, heads, heads_dim
        # key shape = N, key_len, heads, heads_dim
        # energy shape = N, heads, query_len, key_len
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = self.softmax(energy / (self.embed_size**(1/2)))

        out = T.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        # attention shape = N, heads, querry_len, key_len
        # values shape = N, value_len, heads, heads_dim
        # out = N, query_len, heads, heads_dim, then flatten last two dims

        return self.fc_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        # mask is only for the decoder
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query)) # skip connection in encoder block
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward+x)) # skip connection in encoder block
        return out

class Encoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length, src_vocab_size=None):
        super().__init__()
        # masked_length is the max length of a sequence, so for us it is however long we want our sequences to be when training 
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Sequential(nn.Linear(6, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 1))
        self.position_embedding = nn.Embedding(max_length, embed_size) # We need this to propagate the causality
        self.fc_out = nn.Linear(embed_size*max_length, 1) # Transform to angle
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion,)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

        self.optimizer = optim.Adam(self.parameters(), lr = 0.0001  , weight_decay = 1e-4)
        self.to(self.device)

    def forward(self, x, mask = None):
        N, seq_len, obs_size = x.shape
        positions = T.arange(0, seq_len).expand(N, seq_len).to(self.device)

        out = self.word_embedding(x) + self.position_embedding(positions) 
        
        for layer in self.layers:
            mp = layer(out, out, out, mask)
        out = self.fc_out(mp.view(-1))
        out = self.tanh(out) # converts to single number in range (-1, 1) to represent angle
        return out


# embed_size = 256, num_layes = 6, forward_expansion = 4, heads, = 8, dropout = 0, device = "cuda", max_length = 100

intention = Encoder(embed_size=256,
                    num_layers = 6,
                    heads = 8,
                    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu'),
                    forward_expansion = 4,
                    dropout = 0,
                    max_length = 100)

parser = argparse.ArgumentParser()
parser.add_argument("data_path")

args = parser.parse_args()

data_path = args.data_path + "Data/"

Loss = nn.MSELoss()

Y_history = []
Pred_history = []
episode_reward_history = []

print("... Loading CSV Paths ...")
file_names = []
for file in os.listdir(data_path):
    file_names.append(file)

for episode in range(len(file_names)-1):
    print('[EPISODE]', episode)
    history = [[0 for _ in range(6)] for _ in range(100)]
    Y_actual = []
    episode_reward = 0
    name = 'Data_Episode_'+str(episode)+'.csv'
    file_path = data_path + name
    df = pd.read_csv(file_path)
    for t in range(len(df[:-1])):
        #print(episode, t)
        cx = df['cyl_x_pos'][t+1]
        cx_ = df['cyl_x_pos'][t]
        cy = df['cyl_y_pos'][t+1]
        cy_ = df['cyl_y_pos'][t]
        obs = df['env_observations'][t]
        obs = [[float(d) for d in e] for e in [e.split(',') for e in (re.sub(r"[\n\t\s]*", "", (obs[8:-18]))).split('],dtype=float32),array([')]]
        P = [sum(obs[i][-24:])/24 for i in range(len(obs))] #Normalized proximity readings (1 per robot (0, 1))
        attention_observation = [cx_, cy_] + P

        history.append(attention_observation)
        if len(history) > 100:
            history.pop(0)

        Y = T.Tensor([math.atan2((cy_ - cy), (cx_ - cx))/math.pi]).to(intention.device)

        observation = T.Tensor(history).unsqueeze(0).to(intention.device)

        predicted_heading = intention(observation)
        loss = Loss(predicted_heading, Y)
        loss.backward()
        intention.optimizer.step()

        Y_history.append(Y.cpu().detach().numpy())
        Pred_history.append(predicted_heading.cpu().detach().numpy())
        
        x_y = math.cos(Y.cpu().detach().numpy() * math.pi)
        y_y = math.sin(Y.cpu().detach().numpy() * math.pi)

        x_pred = math.cos(predicted_heading.cpu().detach().numpy() * math.pi)
        y_pred = math.sin(predicted_heading.cpu().detach().numpy() * math.pi)

        diff = np.dot([x_y, y_y], [x_pred, y_pred])
        episode_reward += diff
        
    episode_reward_history.append(episode_reward/len(df[:-1]))
    plt.clf()
    #plt.plot(Y_history, label = 'Y')
    #plt.plot(Pred_history, label = 'Pred')
    plt.plot(episode_reward_history)
    #plt.legend()
    plt.savefig('python_code/Data/Figures/Attention_Testing.png')


        
        
    
    