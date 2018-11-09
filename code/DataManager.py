############################################################
# Hierarchical Reinforcement Learning for Relation Extraction
# Multiprocessing with CUDA
# Require: PyTorch 0.3.0
# Author: Tianyang Zhang, Ryuichi Takanobu
# E-mail: keavilzhangzty@gmail.com, truthless11@gmail.com
############################################################

import numpy as np
import json

class DataManager:
    def __init__(self, path, testfile):

        #read data
        self.data = {}
        for name in ["train", "test"]:
            self.data[name] = []
            filename = testfile if name == "test" else name
            with open(path+(filename+".json")) as fl:
                for line in fl.readlines():
                    self.data[name].append(json.loads(line))
        trainlen = len(self.data["train"])
        self.data['dev'] = self.data['train'][:int(trainlen*0.05)]
        self.data['train'] = self.data['train'][int(trainlen*0.05):]
        
        #arrange words
        wordsdic = {}
        for name in ["train", "dev"]:
            datas = self.data[name]
            for item in datas:
                for word in item['sentext'].strip().split():
                    word = word.lower()
                    if word in wordsdic:
                        wordsdic[word] = wordsdic[word] + 1
                    else:
                        wordsdic[word] = 1
        wordssorted = sorted(wordsdic.items(), key = lambda d: (d[1],d[0]), reverse=True) 
        self.words = {}
        for i in range(len(wordssorted)):
            self.words[wordssorted[i][0]] = i
        OOV = len(wordssorted)

        #get text
        for name in ["train", "test", "dev"]:
            for item in self.data[name]:
                item['text'] = []
                for word in item['sentext'].strip().split():
                    if name == "test" and word.lower() not in self.words:
                        item['text'].append(OOV)
                    else:
                        item['text'].append(self.words[word.lower()])
        
        #load word vector
        self.vector = np.random.rand(len(self.words)+1, 300) * 0.1
        with open(path+("vector.txt")) as fl:
            for line in fl.readlines():
                vec = line.strip().split()
                word = vec[0].lower()
                vec = list(map(float, vec[1:]))
                if word in self.words:
                    self.vector[self.words[word]] = np.asarray(vec)
        self.vector = np.asarray(self.vector)

        #get relation count
        self.relationcnt = {}
        self.relations = []
        for name in ['train','dev']:
            for item in self.data[name]:
                for t in item['relations']:
                    rel = t['rtext']
                    if not rel in self.relations:
                        self.relations.append(rel)
                        self.relationcnt[rel] = 1
                    else:
                        self.relationcnt[rel] += 1
        self.relation_count = len(self.relations)
        for name in ['train', 'test', 'dev'] :
            for item in self.data[name]:
                for t in item['relations']:
                    t['type'] = self.relations.index(t['rtext']) + 1
        print(self.relationcnt)
        print(self.relations)
