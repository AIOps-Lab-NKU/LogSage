import os
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch_geometric.data import Data

class GraphConstructor:
    def __init__(self, alpha=0.3, beta=0.7, k=15):
        self.alpha = alpha  
        self.beta = beta  
        self.k = k        

    def compute_word_frequencies(self, cases):
        word_frequencies = []
        for explanation in cases.values():
            word_count = {}
            words = " ".join(explanation).split()
            for word in words:
                word_count[word] = word_count.get(word, 0) + 1
            word_frequencies.append(word_count)
        return word_frequencies

    def construct_graph(self, features, word_frequencies):
        num_nodes = features.shape[0]
        similarity_matrix = cosine_similarity(features)
        edge_index, edge_attr = [], []

        for i in tqdm(range(num_nodes), desc="Constructing graph"):
            neighbors = np.argsort(-similarity_matrix[i])[:self.k + 1]
            for neighbor in neighbors:
                if i == neighbor:
                    continue
                
                common_words = set(word_frequencies[i].keys()) & set(word_frequencies[neighbor].keys())
                cooccur_score = sum(word_frequencies[i][w] + word_frequencies[neighbor][w] for w in common_words)
                cooccur_score /= (sum(word_frequencies[i].values()) + sum(word_frequencies[neighbor].values()))
                
                weight = self.alpha * similarity_matrix[i, neighbor] + self.beta * cooccur_score
                
                edge_index.append([i, neighbor])
                edge_attr.append(weight)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        return Data(x=torch.tensor(features, dtype=torch.float), 
                  edge_index=edge_index, 
                  edge_attr=edge_attr)

    def save_graph(self, graph, path):
        torch.save(graph, path)

    def load_graph(self, path):
        return torch.load(path)