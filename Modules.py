import torch
from torch import nn
import numpy as np
from utils import *

class Encoder(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Encoder, self).__init__()
        self.lookup_table = nn.Parameter(torch.randn((output_size,input_size)))

    def forward(self, input):
        return torch.mm(self.lookup_table,input.view(input.shape[0],1)).flatten()
    
class Decoder(nn.Module):
    def __init__(self, input_size, num_actions):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(input_size,num_actions)

    def forward(self, inputs):
        n_agents = inputs.size()[0]
        out = torch.stack([torch.nn.functional.softmax(self.linear(inputs[i]),dim=-1) for i in range(n_agents)])
        return out


class LinearLayer(nn.Module):
    def __init__(self, num_agents):
        super(LinearLayer,self).__init__()
        
        self.num_agents=num_agents
        self.C =nn.Parameter(torch.randn((num_agents,num_agents)))
        self.H = nn.Parameter(torch.randn((num_agents,num_agents)))
    
    def forward(self, h, c):
        hidden = torch.tanh(torch.mm(self.H,h)+torch.mm(self.C,c))
        order = [[hidden[i] for j in range(self.num_agents) if j!= i] for i in range(self.num_agents)]
        comm = torch.stack([np.sum(o)/(self.num_agents-1) for o in order])
        return hidden, comm



class CommNetMLP(nn.Module):
    def __init__(self, num_agents, input_size, output_size, num_actions, comm_steps):
        super(CommNetMLP,self).__init__()

        #Hyper Parameters
        self.num_agents = num_agents
        self.output_size = output_size
        self.num_actions = num_actions
        self.comm_steps = comm_steps

        #Architecture
        self.encoder = Encoder(input_size, output_size)
        self.linear_layers = nn.ModuleList([LinearLayer(num_agents) for _ in range(self.comm_steps)])
        self.decoder = Decoder(output_size, num_actions)
        self.comm = nn.Parameter(torch.zeros((num_agents,output_size)))
        
        
    def forward(self, states):
        S = [torch.tensor(s) for s in states]
        h0 = torch.stack([self.encoder(s) for s in S])
        for l in self.linear_layers:
            h, c = l(h0,self.comm)
        self.action_distributions = self.decoder(h).detach().numpy()
        return self.action_distributions
        

if __name__=="__main__":
    num_agents = 3
    output_size = 5
    S = [[0.,1.],[0.,1.],[2.,4.]]
    
    C = CommNetMLP(num_agents, 2, 10, 10, 3)
    a = C(S)
    


        
