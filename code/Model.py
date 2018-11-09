############################################################
# Hierarchical Reinforcement Learning for Relation Extraction
# Multiprocessing with CUDA
# Require: PyTorch 0.3.0
# Author: Tianyang Zhang, Ryuichi Takanobu
# E-mail: keavilzhangzty@gmail.com, truthless11@gmail.com
############################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class TopModel(nn.Module):
    def __init__(self, dim, statedim, rel_count):
        super(TopModel, self).__init__()
        self.dim = dim
        self.hid2state = nn.Linear(dim*3 + statedim, statedim)
        self.state2prob = nn.Linear(statedim, rel_count+1)

    def forward(self, top_word_vec, rel_vec, memory, training): 
        inp = torch.cat([top_word_vec, rel_vec, memory])
        outp = F.dropout(F.tanh(self.hid2state(inp)), training=training) 
        prob = F.softmax(self.state2prob(outp), dim=0)
        return outp, prob 

class BotModel(nn.Module):
    def __init__(self, dim, statedim, rel_count):
        super(BotModel, self).__init__()
        self.dim = dim
        self.hid2state = nn.Linear(dim*3 + statedim*2, statedim)
        self.state2probL = nn.ModuleList([nn.Linear(statedim, 7) for i in range(0, rel_count)])

    def forward(self, ent_vec, bot_word_vec, memory, rel, target, training): 
        inp = torch.cat([bot_word_vec, ent_vec, memory, target])
        outp = F.dropout(F.tanh(self.hid2state(inp)), training=training)
        prob = F.softmax(self.state2probL[rel-1](outp), dim=0)
        return outp, prob 

class Model(nn.Module):
    def __init__(self, lr, dim, statedim, wv, rel_count):
        super(Model, self).__init__()
        self.dim = dim
        self.statedim = statedim
        self.rel_count = rel_count
        self.topModel = TopModel(dim, statedim, rel_count)
        self.botModel = BotModel(dim, statedim, rel_count)
        wvTensor = torch.FloatTensor(wv)
        self.wordvector = nn.Embedding(wvTensor.size(0), wvTensor.size(1))
        self.wordvector.weight = nn.Parameter(wvTensor)
        self.relationvector = nn.Embedding(rel_count+1, dim)
        self.entitytypevector = nn.Embedding(7, dim)
        self.preLSTML = nn.LSTMCell(dim, dim)
        self.preLSTMR = nn.LSTMCell(dim, dim)
        self.top2target = nn.Linear(statedim, statedim)
        self.top2bot = nn.Linear(statedim, statedim)
        self.bot2top = nn.Linear(statedim, statedim)
    
    def sample(self, prob, training, preoptions, position):
        if not training:
            return torch.max(prob, 0)[1]
        elif preoptions is not None:
            return autograd.Variable(torch.cuda.LongTensor(1, ).fill_(preoptions[position]))
        else:
            return torch.multinomial(prob, 1)

    def forward(self, mode, text, preoptions=None, preactions=None):
        textin = torch.cuda.LongTensor(text)
        wvs = self.wordvector(autograd.Variable(textin))
        top_action, top_actprob = [], []
        bot_action, bot_actprob = [], [] 
        training = True if "test" not in mode else False

        #-----------------------------------------------------------------
        # Prepare
        prehid = autograd.Variable(torch.cuda.FloatTensor(self.dim, ).fill_(0))
        prec = autograd.Variable(torch.cuda.FloatTensor(self.dim, ).fill_(0))
        front, back = [0 for i in range(len(text))], [0 for i in range(len(text))]
        for x in range(len(text)):
            prehid, prec = self.preLSTML(wvs[x], (prehid, prec))
            front[x] = prehid
        prehid = autograd.Variable(torch.cuda.FloatTensor(self.dim, ).fill_(0))
        prec = autograd.Variable(torch.cuda.FloatTensor(self.dim, ).fill_(0))
        for x in range(len(text))[::-1]:
            prehid, prec = self.preLSTMR(wvs[x], (prehid, prec))
            back[x] = prehid
        wordin = []
        for x in range(len(text)):
            wordin.append(torch.cat([front[x], back[x]]))
        #------------------------------------------------------------------
        # First Layer
        mem = autograd.Variable(torch.cuda.FloatTensor(self.statedim, ).fill_(0))
        action = autograd.Variable(torch.cuda.LongTensor(1, ).fill_(0))
        rel_action = autograd.Variable(torch.cuda.LongTensor(1, ).fill_(0)) 
        for x in range(len(text)):                   
            mem, prob = self.topModel(wordin[x],\
                    self.relationvector(rel_action)[0], mem, training)
            action = self.sample(prob, training, preoptions, x)
            if action.data[0] != 0: 
                rel_action = action
            actprob = prob[action]
            top_action.append(action.cpu().data[0])
            if not training:
                top_actprob.append(actprob.cpu().data[0])
            else:
                top_actprob.append(actprob)

            #----------------------------------------------------------------
            # Second Layer
            if "NER" in mode and action.data[0] > 0:
                rel = action.data[0]
                target = self.top2target(mem)
                actionb = autograd.Variable(torch.cuda.LongTensor(1, ).fill_(0))
                actions, actprobs = [], []
                mem = self.top2bot(mem)
                for y in range(len(text)):
                    mem, probb = self.botModel(\
                            self.entitytypevector(actionb)[0], wordin[y], \
                            mem, rel, target, training)
                    actionb = self.sample(probb, training, preactions[x] if preactions is not None else None, y)
                    actprobb = probb[actionb]
                    actions.append(actionb.cpu().data[0])
                    if not training:
                        actprobs.append(actprobb.cpu().data[0]) 
                    else:
                        actprobs.append(actprobb)
                mem = self.bot2top(mem)
                bot_action.append(actions)
                bot_actprob.append(actprobs)
        return top_action, top_actprob, bot_action, bot_actprob

