import numpy as np
import torch
import random

def seed_worker(self, worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)