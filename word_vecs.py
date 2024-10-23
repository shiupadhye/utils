import torch
import numpy as np
import torch.nn as nn
from gensim.models import FastText
from gensim.utils import simple_preprocess


def train_fasttext(corpus_file,embedding_dim,min_count,model_path):
    with open(corpus_file, 'r', encoding='utf-8') as file:
        corpus = [simple_preprocess(line) for line in file]
    model = FastText(sentences=corpus, vector_size=embedding_dim, min_count=min_count, sg=1)
    model.save(model_path)


def vectorized_cosine_similarity(M,N,distance=False):
    # compute magnitude
    mag_M = torch.norm(M,dim=1).unsqueeze(1)
    mag_N = torch.norm(N,dim=1).unsqueeze(0)
    # dot prod
    dot_prod = M @ N.T
    # compute cosine similarity
    cos_sim = dot_prod / (mag_M * mag_N)
    if distance:
        return 1 - cos_sim
    else:
        return cos_sim
    

def add_gaussian_noise(v,mean,stddev):
    noisy_v = v + np.random.normal(mean,stddev,v.shape[0])
    return torch.tensor(noisy_v)



