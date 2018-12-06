############################################################
# Hierarchical Reinforcement Learning for Relation Extraction
# Multiprocessing with CUDA
# Require: PyTorch 0.3.0
# Author: Tianyang Zhang, Ryuichi Takanobu
# E-mail: keavilzhangzty@gmail.com, truthless11@gmail.com
############################################################

import numpy as np
import torch
import torch.autograd as autograd
from AccCalc import calc_acc, calcF1

def calcTopReward(top_action, gold_labels):
    lenth = len(top_action)
    r = [0. for i in range(lenth)]
    rem = [0 for i in range(len(gold_labels))]
    for i in range(lenth)[::-1]:
        if top_action[i] > 0:
            ok = -1
            for j, label in enumerate(gold_labels):
                if label['type'] == top_action[i]:
                    if rem[j] == 0:
                        ok = 0.5
                        rem[j] = 1
                        break
                    else:
                        ok = -0.2
            r[i] = ok
    return r

def calcTopFinalReward(top_action, gold_labels, top_bias = 0.):
    r = 0.
    a1, t1, c1 = calc_acc(top_action, None, gold_labels, ["RE"])
    if c1 != 0:
        r = calcF1(a1, c1, t1, beta=0.9)
    else:
        r = -2
    if c1 > t1:
        r -= 0.5 * (c1 - t1)
    r *= len(top_action)
    return r - top_bias

def calcBotReward(top_action, bot_action, gold_labels):
    lenth = len(top_action)
    r = [[0. for i in range(lenth)] for j in range(len(bot_action))]
    j = 0
    for i in range(lenth):
        if top_action[i] > 0:
            for label in gold_labels:
                if label['type'] == top_action[i]:
                    for t in range(lenth):
                        if label['tags'][t] == bot_action[j][t]:
                            if label['tags'][t] in [4, 5, 6]:
                                r[j][t] = 0.5
                            elif label['tags'][t] in [1, 2, 3]:
                                r[j][t] = 0.2
                        else:
                            r[j][t] = -0.5
            j += 1
    return r

def calcBotFinalReward(top_action, bot_action, gold_labels, bot_bias = 0.):
    lenth = len(top_action)
    r = [0. for j in range(len(bot_action))]
    j = 0
    for i in range(lenth):
        if top_action[i] > 0:
            r[j] = -1.0
            for label in gold_labels:
                if label['type'] == top_action[i]:
                    ok = True
                    for t in range(lenth):
                        if label['tags'][t] != bot_action[j][t]:
                            ok = False;
                            break;
                    if ok:
                        r[j] = 1.0
            j += 1
    for j in range(len(bot_action)):
        r[j] -= bot_bias
    return r

def calcTopGrad(top_action, top_actprob, top_reward, top_final_reward, pretrain=False):
    lenth = len(top_action)
    decay_reward = top_final_reward 
    grads = autograd.Variable(torch.cuda.FloatTensor(1, ).fill_(0))
    for i in range(lenth)[::-1]:
        decay_reward = decay_reward * 0.95 + top_reward[i]
        to_grad = -torch.log(top_actprob[i])
        if not pretrain:
            to_grad *= autograd.Variable(torch.cuda.FloatTensor(1, ).fill_(decay_reward))
        if top_action[i] == 0:
            to_grad *= 0.3
        grads = grads + to_grad
    return grads

def calcBotGrad(top_action, bot_action, bot_actprob, bot_reward, bot_final_reward, pretrain=False):
    lenth = len(top_action)
    bot_tot_reward = [0. for i in range(lenth)]
    grads = autograd.Variable(torch.cuda.FloatTensor(1, ).fill_(0))
    j = 0
    for i in range(lenth):
        if top_action[i] > 0:
            bot_tot_reward[i] = sum(bot_reward[j]) / lenth + bot_final_reward[j]#
            for k in range(lenth)[::-1]:
                to_grad = -torch.log(bot_actprob[j][k]) 
                if not pretrain:
                    to_grad *= autograd.Variable(torch.cuda.FloatTensor(1, ).fill_(bot_tot_reward[i]))
                if bot_action[j][k] == 0:
                    to_grad *= 0.3 
                elif bot_action[j][k] == 3 or bot_action[j][k] == 6:
                    to_grad *= 0.7 
                else:
                    to_grad *= 1.0
                grads = grads + to_grad
            j += 1
    return bot_tot_reward, grads

def optimize(model, top_action, top_actprob, bot_action, bot_actprob, gold_labels, mode, top_bias = 0., bot_bias = 0.):
    lenth = len(top_action)
    top_reward = calcTopReward(top_action, gold_labels)
    top_final_reward = calcTopFinalReward(top_action, gold_labels, top_bias)
    pretrain = True if "pretrain" in mode else False
    if "NER" in mode:
        bot_reward = calcBotReward(top_action, bot_action, gold_labels)
        bot_final_reward = calcBotFinalReward(top_action, bot_action, gold_labels, bot_bias)
        bot_tot_reward, grads = calcBotGrad(top_action, bot_action, bot_actprob, bot_reward, bot_final_reward, pretrain)
        for i in range(lenth):
            top_reward[i] += bot_tot_reward[i]
    else:
        grads = autograd.Variable(torch.cuda.FloatTensor(1, ).fill_(0))
    if "RE" in mode:
        grads += calcTopGrad(top_action, top_actprob, top_reward, top_final_reward, pretrain)
    loss = grads.cpu().data[0]
    grads.backward()
    return loss

def optimize_round(model, top_actions, top_actprobs, bot_actions, bot_actprobs, gold_labels, mode):
    sample_round = len(top_actions)
    if "RE" in mode:
        top_bias = 0.
        for i in range(sample_round):
            top_bias += calcTopFinalReward(top_actions[i], gold_labels, 0.)
        top_bias /= sample_round
    else:
        top_bias = 0.
    if "NER" in mode:
        bot_bias, bot_cnt = 0., 0
        for i in range(sample_round):
            tmp = calcBotFinalReward(top_actions[i], bot_actions[i], gold_labels, 0.)
            bot_cnt += len(tmp)
            bot_bias += np.sum(tmp)
        if bot_cnt != 0:
            bot_bias /= bot_cnt
    else:
        bot_bias = 0.
    loss = .0
    for i in range(sample_round):
        loss += optimize(model, top_actions[i], top_actprobs[i], bot_actions[i], \
                bot_actprobs[i], gold_labels, mode, top_bias, bot_bias)
    return loss / sample_round
