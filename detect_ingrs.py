
import numpy as np
import json
import random
from tqdm import *

vocab_file = 'vocab.txt'
with open(vocab_file) as f_vocab:
    ingr_vocab = {w.rstrip(): i for i, w in enumerate(f_vocab)}
    #ingr_vocab['</i>'] = 1
  

def detect_ingrs(recipe):

    ingr_names = recipe
    ts = [',','.',';','(',')','?','!','&','%',':','*','"','\'','-','_',chr(169)]
    detected = [0]
    #detected.add(0)
    undetected = ''

    for names in ingr_names:
      for t in ts:
        names = names.replace(t,' '+t+' ')

      for name in names.split(' '):
        name_ind = ingr_vocab.get(name.lower())
        #print(name, name_ind)

        if name_ind:
          detected.append(name_ind)
        else:
          undetected += name +','
    return list(detected) + [ingr_vocab['</s>']]


#data = json.load(open('./dataset/train.json','r'))
#for i, entry in tqdm(enumerate(data)):
#     ingr_detections = detect_ingrs(entry)
