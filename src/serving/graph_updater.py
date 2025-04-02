import os
import time
import threading
import numpy as np
from typing import Dict, List
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from KPRCA_stage.graph_constructor import GraphConstructor

class GraphUpdater:
    def __init__(self, 
                 model_path: str,
                 data_dir: str,
                 update_interval: int = 3600,
                 max_features: int = 5000,
                 k_neighbors: int = 15):

        self.model_path = model_path
        self.data_dir = data_dir
        self.update_interval = update_interval
        self.max_features = max_features
        self.k_neighbors = k_neighbors
        self.graph_constructor = GraphConstructor(alpha=0.3, beta=0.7, k=k_neighbors)
        self.current_graph = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.Lock()
        self._stop_event = threading.Event()

    def start(self):
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

    def stop(self):
        self._stop_event.set()
        self.update_thread.join()

    def _update_loop(self):
        while not self._stop_event.is_set():
            try:
                self._perform_update()
            except Exception as e:
                print(f"Graph update failed: {str(e)}")
            time.sleep(self.update_interval)

    def _perform_update(self):

        new_cases = self._load_new_cases()
        if not new_cases:
            return

        all_logs = [" ".join(logs) for logs in new_cases.values()]
        if not hasattr(self.tfidf_vectorizer, 'vocabulary_'):

            tfidf_features = self.tfidf_vectorizer.fit_transform(all_logs).toarray()
        else:
            tfidf_features = self.tfidf_vectorizer.transform(all_logs).toarray()

        word_frequencies = self.graph_constructor.compute_word_frequencies(new_cases)

        subgraph = self.graph_constructor.construct_graph(tfidf_features, word_frequencies)

        with self.lock:
            if self.current_graph is None:
                self.current_graph = subgraph
            else:
                self._merge_graphs(subgraph)

        print(f"Graph updated at {datetime.now()}, current nodes: {self.current_graph.num_nodes}")

    def _load_new_cases(self) -> Dict[str, List[str]]:
        cases = {}
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.log'):
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    cases[filename] = f.readlines()
        return cases

    def _merge_graphs(self, subgraph):
        self.current_graph.x = torch.cat([self.current_graph.x, subgraph.x], dim=0)
        
        offset = self.current_graph.num_nodes
        subgraph.edge_index += offset
        self.current_graph.edge_index = torch.cat(
            [self.current_graph.edge_index, subgraph.edge_index], dim=1)
        
        self.current_graph.edge_attr = torch.cat(
            [self.current_graph.edge_attr, subgraph.edge_attr], dim=0)

    def get_current_graph(self):
        with self.lock:
            return self.current_graph

    def save_current_graph(self, path: str):
        with self.lock:
            if self.current_graph is not None:
                self.graph_constructor.save_graph(self.current_graph, path)