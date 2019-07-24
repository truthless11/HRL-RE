# HRL-RE
Codes for the paper "A Hierarchical Framework for Relation Extraction with Reinforcement Learning", and you can find our paper at [arxiv](https://arxiv.org/abs/1811.03925)

Cite this paper :
```
@inproceedings{takanobu2019hierarchical,
  title={A Hierarchical Framework for Relation Extraction with Reinforcement Learning},
  author={Takanobu, Ryuichi and Zhang, Tianyang and Liu, Jiexi and Huang, Minlie},
  booktitle={AAAI},
  pages={7072--7079},
  year={2019}
}
```

## Data

NYT10 is originally released by the paper "Modeling relations and their mentions without labeled text." [Download](http://iesl.cs.umass.edu/riedel/ecml/) , while NYT11 is provided by the paper "CoType: Joint Extraction of Typed Entities and Relations with Knowledge Bases"  [Download](https://drive.google.com/drive/folders/0B--ZKWD8ahE4UktManVsY1REOUk?usp=sharing) 

The preprocessed dataset that we used for relation extraction are provided in corresponding subdirectories under ``HRL-RE/data``. 

Each relational triple is formatted as follows:

- rtext : relation type
- em1 : source entity mention
- em2 : target entity mention
- tags : the proposed entity annotation scheme for the sentence
  - 0 :  $O$ non-entity 
  - 1 : $S_I$ inside of a source entity
  - 2 : $T_I$ inside of a target entity
  - 3 : $O_I$ inside of not-concerned entity
  - 4 : $S_B$ begin of a source entity
  - 5 : $T_B$ begin of a target entity
  - 6 : $O_B$ begin of not-concerned entity

## Run

Command

```
cd code
python main.py {--[option1]=[value1] --[option2]=[value2] ... }
```

Change the corresponding options to set hyper-parameters:

```python
parser.add_argument('--logfile', type=str, default='HRL', help="Filename of log file")
parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
parser.add_argument('--epochPRE', type=int, default=15, help="Number of epoch on pretraining")
parser.add_argument('--epochRL', type=int, default=10, help="Number of epoch on training with RL")
parser.add_argument('--dim', type=int, default=300, help="Dimension of hidden layer")
parser.add_argument('--statedim', type=int, default=300, help="Dimension of state")
parser.add_argument('--batchsize', type=int, default=16, help="Batch size on training")
parser.add_argument('--batchsize_test', type=int, default=64, help="Batch size on testing")
parser.add_argument('--print_per_batch', type=int, default=200, help="Print results every XXX batches")
parser.add_argument('--sampleround', type=int, default=5, help="Sample round in RL")
parser.add_argument('--numprocess', type=int, default=4, help="Number of process")
parser.add_argument('--start', type=str, default='', help="Directory to load model")
parser.add_argument('--test', type=bool, default=False, help="Set to True to inference")
parser.add_argument('--pretrain', type=bool, default=False, help="Set to True to pretrain")
parser.add_argument('--datapath', type=str, default='../data/NYT10/', help="Data directory")
parser.add_argument('--testfile', type=str, default='test', help="Filename of test file")
```
**NOTE**: please start with pretraining, e.g.

    python main.py --datapath ../data/NYT10/ --pretrain True

then,

    python main.py --lr 2e-5 --datapath ../data/NYT10/ --start checkpoints/model_HRL_10


## Requirements

- python 3
- pytorch = 0.3 