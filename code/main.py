############################################################
# Hierarchical Reinforcement Learning for Relation Extraction
# Multiprocessing with CUDA
# Require: PyTorch 0.3.0
# Author: Tianyang Zhang, Ryuichi Takanobu
# E-mail: keavilzhangzty@gmail.com, truthless11@gmail.com
############################################################

import random, sys, time
import torch
from DataManager import DataManager
from Model import Model
from Parser import Parser
from TrainProcess import train, test, worker
import torch.multiprocessing as mp
from AccCalc import calcF1

def work(mode, train_data, test_data, dev_data, model, args, sampleround, epoch):
    for e in range(epoch):
        random.shuffle(train_data)
        print("training epoch ", e)
        batchcnt = (len(train_data) - 1) // args.batchsize + 1
        for b in range(batchcnt):
            start = time.time()
            data = train_data[b * args.batchsize : (b+1) * args.batchsize]
            acc, cnt, tot = train(b, model, data, sampleround, \
                    mode, dataQueue, resultQueue, freeProcess, lock, args.numprocess)
            trainF1 = calcF1(acc, cnt, tot)
            if b % args.print_per_batch == 0:
                print("    batch ", b, ": F1:", trainF1, "    time:", (time.time()-start))
        batchcnt = (len(dev_data) - 1) // args.batchsize_test + 1
        acc, cnt, tot = 0, 0, 0
        for b in range(batchcnt):
            data = dev_data[b * args.batchsize_test : (b+1) * args.batchsize_test]
            acc_, cnt_, tot_ = test(b, model, data, mode, \
                    dataQueue, resultQueue, freeProcess, lock, args.numprocess)
            acc += acc_
            cnt += cnt_ 
            tot += tot_
        devF1 = calcF1(acc, cnt, tot)
        batchcnt = (len(test_data) - 1) // args.batchsize_test + 1
        acc, cnt, tot = 0, 0, 0
        for b in range(batchcnt):
            data = test_data[b * args.batchsize_test : (b+1) * args.batchsize_test]
            acc_, cnt_, tot_ = test(b, model, data, mode, \
                    dataQueue, resultQueue, freeProcess, lock, args.numprocess)
            acc += acc_
            cnt += cnt_ 
            tot += tot_
        testF1 = calcF1(acc, cnt, tot)
        f = open("checkpoints/"+args.logfile+".log", 'a')
        print("epoch ", e, ": dev F1: ", devF1, ", test F1: ", testF1)
        f.write("epoch "+ str(e)+ ": dev F1: "+ str(devF1)+ ", test F1: "+ str(testF1)+ "\n")
        f.close()
        torch.save(model, "checkpoints/model_"+args.logfile+"_"+str(e))

if __name__ == "__main__":
    torch.manual_seed(1)

    argv = sys.argv[1:]
    parser = Parser().getParser()
    args, _ = parser.parse_known_args(argv)

    print("Load data start...")
    dm = DataManager(args.datapath, args.testfile)
    wv = dm.vector

    train_data, test_data, dev_data = dm.data['train'], dm.data['test'], dm.data['dev']
    print("train_data count: ", len(train_data))
    print("test_data  count: ", len(test_data))
    print("dev_data   count: ", len(dev_data))

    model = Model(args.lr, args.dim, args.statedim, wv, dm.relation_count)
    model.cuda()
    if args.start != '':
        pretrain_model = torch.load(args.start) 
        model_dict = model.state_dict() 
        pretrained_dict = pretrain_model.state_dict() 
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} 
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict) 
    model.share_memory()
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    for name, param in model.named_parameters():
        print (name, param.size(), param.get_device())
    
    processes = []
    dataQueue = mp.Queue()
    resultQueue = mp.Queue()
    freeProcess = mp.Manager().Value("freeProcess", 0)
    lock = mp.Lock()
    flock = mp.Lock()
    print("Starting training service, overall process number: ", args.numprocess)
    for r in range(args.numprocess):
        p = mp.Process(target=worker, args= \
                (model, r, dataQueue, resultQueue, freeProcess, lock, flock, args.lr))
        p.start()
        processes.append(p)
                
    if args.test == True:
        batchcnt = (len(test_data) - 1) // args.batchsize_test + 1
        acc, cnt, tot = 0, 0, 0
        for b in range(batchcnt):
            data = test_data[b * args.batchsize_test : (b+1) * args.batchsize_test]
            acc_, cnt_, tot_ = test(b, model, data, ["RE","NER"], \
                    dataQueue, resultQueue, freeProcess, lock, args.numprocess)
            acc += acc_
            cnt += cnt_ 
            tot += tot_
            print(acc, cnt, tot)
        testF1 = calcF1(acc, cnt, tot)
        print("test P: ", acc/cnt, "test R: ", acc/tot, "test F1: ", testF1)
    elif args.pretrain == True:
        work(["RE", "NER", "pretrain"], train_data, test_data, dev_data, model, args, 1, args.epochPRE)
    else:
        work(["RE", "NER"], train_data, test_data, dev_data, model, args, args.sampleround, args.epochRL)
    for p in processes:
        p.terminate()
