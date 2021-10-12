import ray
import torch
import os
import datetime

@ray.remote
class SharedStorage:
    """
    Class which run in a dedicated thread to store the network weights and some information.
    """

    def __init__(self, weights, config):
        self.config = config
        self.weights = weights
        self.info = {
            "warm_up": True,
            "training_step": 0,
            "lr": 0,
            "task_id": None,
            "nlm_loss_outdegree": 0,
            "nlm_loss_adjacent": 0,
            "nlm_loss_connectivity": 0,
            "nlm_loss_hfather": 0,
            "nlm_loss_hsister": 0,
            "nlm_loss_grandparents": 0,
            "nlm_loss_uncle": 0,
            "nlm_loss_MGuncle": 0,
            "acc_train_outdegree": 0,
            "acc_train_adjacent": 0,
            "acc_train_connectivity": 0,
            "acc_train_hfather": 0,
            "acc_train_hsister": 0,
            "acc_train_grandparents": 0,
            "acc_train_uncle": 0,
            "acc_train_MGuncle": 0,
            "TrainStep_outdegree": 0,
            "TrainStep_adjacent": 0,
            "TrainStep_connectivity": 0,
            "TrainStep_hfather": 0,
            "TrainStep_hsister": 0,
            "TrainStep_grandparents": 0,
            "TrainStep_uncle": 0,
            "TrainStep_MGuncle": 0,
        }

    def get_weights(self):
        return self.weights

    def set_weights(self, weights, path=None):
        self.weights = weights
        if not path:
            path = os.path.join(self.config.results_path, "model.weights")
        torch.save(self.weights, path)

    def get_info(self):
        return self.info

    def set_info(self, key, value):
        self.info[key] = value
