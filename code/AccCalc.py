############################################################
# Hierarchical Reinforcement Learning for Relation Extraction
# Multiprocessing with CUDA
# Require: PyTorch 0.3.0
# Author: Tianyang Zhang, Ryuichi Takanobu
# E-mail: keavilzhangzty@gmail.com, truthless11@gmail.com
############################################################

def calcF1(acc, cnt, tot, beta=1.0):
    if cnt == 0 or tot == 0:
        return 0
    precision = float(acc) / float(cnt)
    recall = float(acc) / float(tot)
    if precision + recall < 1e-5:
        return 0
    return (1+beta*beta) * precision * recall / (beta*beta*precision + recall)

def calc_acc(top_action, bot_action, gold_labels, mode):
    acc, cnt, tot = 0, 0, len(gold_labels)
    used = [0 for i in range(len(top_action))]
    for label in gold_labels:
        tp, tags = label['type'], label['tags']
        j, ok = 0, 0
        for i in range(len(top_action)):
            if top_action[i] == tp and tp > 0 and used[i] == 0 and ok == 0:
                match = 1
                if "NER" in mode:
                    for k in range(len(bot_action[j])):
                        if tags[k] == 4 and bot_action[j][k] != 4:
                            match = 0
                        if tags[k] != 4 and bot_action[j][k] == 4:
                            match = 0
                        if tags[k] == 5 and bot_action[j][k] != 5:
                            match = 0
                        if tags[k] != 5 and bot_action[j][k] == 5:
                            match = 0
                if match == 1:
                    ok = 1
                    used[i] = 1
            if top_action[i] > 0:
                j += 1
                cnt += 1
        acc += ok
    cnt //= tot
    return acc, tot, cnt

def find_tail(tags, num):
    last = False
    for i, x in enumerate(tags):
        if x != num and last:
            return i-1
        if x == num+3:
            last = True
    return len(tags)-1 if last else -1
    
def rule_actions(gold_labels):
    length = len(gold_labels[0]['tags'])
    options = [0 for i in range(length)]
    actions = [[] for i in range(length)]
    for label in gold_labels:
        tp, tags = label['type'], label['tags']
        entity_1 = find_tail(tags, 1)
        assert entity_1 != -1
        entity_2 = find_tail(tags, 2)
        assert entity_2 != -1
        pos = max(entity_1, entity_2)
        while pos < len(tags) and options[pos] != 0:
            pos += 1
        if pos != len(tags):
            options[pos] = tp
            actions[pos] = tags
        else:
            pos = max(entity_1, entity_2) - 1
            while pos >= 0 and options[pos] != 0:
                pos -= 1
            if pos != -1:
                options[pos] = tp
                actions[pos] = tags
    return options, actions	
				
