import torch
import torch.nn as nn
import torch.nn.functional as F
from random import choice
from Utils import NLP
from torch.optim import SGD
from math import log

class Transformer(nn.Module):

    def __init__(self):

        super().__init__()

        self.Wq = nn.Parameter(torch.rand(384,384),requires_grad=True)
        self.Wk = nn.Parameter(torch.rand(384,384),requires_grad=True)
        self.Wv = nn.Parameter(torch.rand(384,384),requires_grad=True)
        self.ffws = nn.Parameter(torch.rand(384),requires_grad=True)

        #self.positional_encoding = nn.Parameter(torch.rand(384),requires_grad=True)

    def weighted_sum(self,soft_sim, xv):

        N = choice(soft_sim.size())
        zs = []
        
        for i in range(N):
            tot = torch.zeros(xv.size()[1])
            for j in range(N):
                tot += soft_sim[i][j]*xv[j]
            zs.append(tot.unsqueeze(0))

        return torch.cat(tuple(zs),0)

    def forward(self,x):
        
        #block 1
        Q = self.Wq @ x.t()
        K = self.Wk @ x.t()
        V = self.Wv @ x.t()

        QK = Q.t() @ K
        QK = F.softmax(Q.t() @ K)
        z = self.weighted_sum(QK,V.t())

        #block 2
        Q = self.Wq @ z.t()
        K = self.Wk @ z.t()
        V = self.Wv @ z.t()

        QK = Q.t() @ K
        QK = F.softmax(Q.t() @ K)
        z = self.weighted_sum(QK,V.t())

        o = torch.sigmoid(sum(self.ffws * z[-1]))

        return o

        
def main():

    torch.manual_seed(0)

    dummy_data = ['you won a billion dollars great work',
                  'click here for cs685 midterm answers',
                  'read important cs685 news',
                  'send me your bank account info asap']

    dummy_labels = torch.tensor([1, 1, 0, 1],dtype=torch.float64,requires_grad=True)

    model = Transformer()

    optimizer = SGD(model.parameters(), lr=0.1)

    for epoch in range(10):

        i = 0
        
        for x in dummy_data:
            tokens = x.split(' ')
            embeddings = [NLP.embedding(token) for token in tokens]
            CLS_embedding = torch.rand(384)
            embeddings = [list(embedding) for embedding in embeddings] + [list(CLS_embedding)]
            embeddings = torch.tensor(embeddings)
            output = model(embeddings)

            loss = -1 * ((dummy_labels[i]*log(output + 0.00001)) + ((1-dummy_labels[i])*log(1-output+0.00001)))
            print (loss)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            i += 1


main()

    
