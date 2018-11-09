############################################################
# Hierarchical Reinforcement Learning for Relation Extraction
# Multiprocessing with CUDA
# Require: PyTorch 0.3.0
# Author: Tianyang Zhang, Ryuichi Takanobu
# E-mail: keavilzhangzty@gmail.com, truthless11@gmail.com
############################################################

import time
import torch.optim as optim
import queue
from AccCalc import calc_acc, rule_actions
from Optimize import optimize_round

def workProcess(model, datas, sample_round, mode):
    acc, cnt, tot = 0, 0, 0
    loss = .0
    for data in datas:
        top_actions, top_actprobs, bot_actions, bot_actprobs = [], [], [], []
        preoptions, preactions = rule_actions(data['relations'])   
        for i in range(sample_round):
            if "pretrain" in mode and "test" not in mode:
                top_action, top_actprob, bot_action, bot_actprob = \
                        model(mode, data['text'], \
                        preoptions, preactions)
            else:
                top_action, top_actprob, bot_action, bot_actprob = \
                        model(mode, data['text'])
            top_actions.append(top_action)
            top_actprobs.append(top_actprob)
            bot_actions.append(bot_action)
            bot_actprobs.append(bot_actprob)
            acc1, tot1, cnt1 = calc_acc(top_action, bot_action, \
                    data['relations'], mode)
            acc += acc1
            tot += tot1
            cnt += cnt1
        if "test" not in mode:
            loss += optimize_round(model, top_actions, top_actprobs, bot_actions,\
                    bot_actprobs, data['relations'], mode)
    return acc, cnt, tot, loss / len(datas)

def worker(model, rank, dataQueue, resultQueue, freeProcess, lock, flock, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print("Process ", rank, " start service.")
    flock.acquire()
    freeProcess.value += 1
    flock.release()
    while True:
        datas, sample_round, mode, dataID = dataQueue.get()
        flock.acquire()
        freeProcess.value -= 1
        flock.release()
        model.zero_grad()
        acc, cnt, tot, loss = workProcess(model, datas, sample_round, mode)
        resultQueue.put((acc, cnt, tot, dataID, rank, loss))
        if not "test" in mode:
            lock.acquire()
            optimizer.step()
            lock.release()
        flock.acquire()
        freeProcess.value += 1
        flock.release()

def train(dataID, model, datas, sample_round, mode, dataQueue, resultQueue, freeProcess, lock, numprocess):
    dataPerProcess = len(datas) // numprocess
    while freeProcess.value != numprocess:
        pass

    acc, cnt, tot = 0, 0, 0
    loss = .0
    for r in range(numprocess):
        endPos = ((r+1)*dataPerProcess if r+1 != numprocess else len(datas))
        data = datas[r*dataPerProcess: endPos]
        dataQueue.put((data, sample_round, mode, dataID))
    lock.acquire()
    try:
        for r in range(numprocess):
            while True:
                item = resultQueue.get()
                if item[3] == dataID:
                    break
                else:
                    print ("receive wrong dataID: ", item[3], "from process ", item[4])
            acc += item[0]
            cnt += item[1]
            tot += item[2]
            loss += item[5]
    except queue.Empty:
        print("The result of some process missed...")
        print(freeProcess.value)
        lock.release()
        time.sleep(2)
        print(freeProcess.value)
        while True:
            pass

    lock.release()
    if dataID > 0 and dataID % 20 == 0:
        print (acc, cnt, tot, loss / numprocess)
    return (acc, cnt, tot)

def test(dataID, model, datas, mode, dataQueue, resultQueue, freeProcess, lock, numprocess):
    testmode = mode + ["test"]
    if dataID < -2:
        print(testmode)
    return train(-dataID-1, model, datas, 1, testmode, dataQueue, resultQueue, freeProcess, lock, numprocess)

