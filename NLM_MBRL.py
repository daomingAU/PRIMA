import copy
import os
import pickle
import time

import numpy
import ray
import torch

import models
import replay_buffer
import self_play
import shared_storage
import trainer
from NLM_MBRL_config import NLM_MBRL_Config
import numpy as np
from os import path
import argparse

parser = argparse.ArgumentParser(description="PRIMA for all tasks")
parser.add_argument("--lr_val", type=float, default=0.075)
parser.add_argument("--lr_pol", type=float, default=0.075)
parser.add_argument("--lr_nlm", type=float, default=0.004)
parser.add_argument("--train_len", type=int, default=400000)
parser.add_argument("--runs", type=int, default=1)
parser.add_argument("--rwd_temp", type=int, default=5)
parser.add_argument("--gen_graph_method", default='edges')
parser.add_argument("--gen_directed", type=bool, default=False)
parser.add_argument("--residual", type=bool, default=False)
parser.add_argument("--connectivity_dist_limit", type=int, default=4)
parser.add_argument("--gen_graph_pmax", type=float, default=0.3)
parser.add_argument("--gen_graph_pmin", type=float, default=0.)
parser.add_argument("--breadth", type=int, default=3)
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--outdegree_n", type=int, default=1)
parser.add_argument("--num_actors", type=int, default=8)
parser.add_argument("--batch_train", type=int, default=16)
parser.add_argument("--c_init", type=float, default=30.)
parser.add_argument("--task_mode", default='train')
ARGS = parser.parse_args()



class NLM_MBRL:
    """
    Main class to manage PRIMA/PRISA.
    """

    def __init__(self, config):

        self.config = config

        # Weights and replay buffer used to initialize workers
        self.model_weights = models.NLM_MBRL_Network(self.config).get_weights()
        self.replay_buffer = None

    def train(self):
        print("---- Training ----")
        ray.init()
        os.makedirs(self.config.results_path, exist_ok=True)

        ### Initialize workers
        training_worker = trainer.Trainer.options(num_gpus=1 if "cuda" in self.config.training_device else 0).remote(copy.deepcopy(self.model_weights), self.config)
        shared_storage_worker = shared_storage.SharedStorage.remote(copy.deepcopy(self.model_weights), self.config)

        replay_buffer_workers = [replay_buffer.ReplayBuffer.remote(self.config) for _ in range(self.config.len_tasks)]
        self_play_workers = [self_play.SelfPlay.remote(copy.deepcopy(self.model_weights), seed, self.config) for seed in range(self.config.num_actors)]

        ### Launch workers
        [self_play_worker.continuous_self_play.remote(shared_storage_worker, replay_buffer_workers) for self_play_worker in self_play_workers]
        training_worker.continuous_update_weights.remote(replay_buffer_workers, shared_storage_worker)

        self._logging_loop(shared_storage_worker, replay_buffer_workers)

        self.model_weights = ray.get(shared_storage_worker.get_weights.remote())

        ### End running actors
        ray.shutdown()

    def _logging_loop(self, shared_storage_worker, replay_buffer_workers):
        # Loop for updating the training performance
        info = ray.get(shared_storage_worker.get_info.remote())
        try:
            nlm_loss_outdegree = {}
            nlm_loss_adjacent = {}
            nlm_loss_connectivity = {}
            nlm_loss_hfather = {}
            nlm_loss_hsister = {}
            nlm_loss_grandparents = {}
            nlm_loss_uncle = {}
            nlm_loss_MGuncle = {}

            acc_train_outdegree = {}
            acc_train_adjacent = {}
            acc_train_connectivity = {}
            acc_train_hfather = {}
            acc_train_hsister = {}
            acc_train_grandparents = {}
            acc_train_uncle = {}
            acc_train_MGuncle = {}

            count_outdegree = 0
            count_adjacent = 0
            count_connectivity = 0
            count_hfather = 0
            count_hsister = 0
            count_grandparents = 0
            count_uncle = 0
            count_MGuncle = 0

            Task = None

            while info["training_step"] < self.config.training_steps:
                increm = info["training_step"]
                info = ray.get(shared_storage_worker.get_info.remote())
                idx_task = info["task_id"]

                count_outdegree = ray.get(replay_buffer_workers[0].get_self_play_count.remote())
                count_adjacent = ray.get(replay_buffer_workers[1].get_self_play_count.remote())
                count_connectivity = ray.get(replay_buffer_workers[2].get_self_play_count.remote())
                count_hfather = ray.get(replay_buffer_workers[3].get_self_play_count.remote())
                count_hsister = ray.get(replay_buffer_workers[4].get_self_play_count.remote())
                count_grandparents = ray.get(replay_buffer_workers[5].get_self_play_count.remote())
                count_uncle = ray.get(replay_buffer_workers[6].get_self_play_count.remote())
                count_MGuncle = ray.get(replay_buffer_workers[7].get_self_play_count.remote())

                if idx_task == 0:
                    TrainStep = info["TrainStep_outdegree"]
                    acc_train = info["acc_train_outdegree"]
                    nlm_loss = info["nlm_loss_outdegree"]
                    nlm_loss_outdegree[TrainStep] = nlm_loss
                    acc_train_outdegree[TrainStep] = acc_train
                    Task = self.config.TASKs[idx_task]
                elif idx_task == 1:
                    TrainStep = info["TrainStep_adjacent"]
                    acc_train = info["acc_train_adjacent"]
                    nlm_loss = info["nlm_loss_adjacent"]
                    nlm_loss_adjacent[TrainStep] = nlm_loss
                    acc_train_adjacent[TrainStep] = acc_train
                    Task = self.config.TASKs[idx_task]
                elif idx_task == 2:
                    TrainStep = info["TrainStep_connectivity"]
                    acc_train = info["acc_train_connectivity"]
                    nlm_loss = info["nlm_loss_connectivity"]
                    nlm_loss_connectivity[TrainStep] = nlm_loss
                    acc_train_connectivity[TrainStep] = acc_train
                    Task = self.config.TASKs[idx_task]
                elif idx_task == 3:
                    TrainStep = info["TrainStep_hfather"]
                    acc_train = info["acc_train_hfather"]
                    nlm_loss = info["nlm_loss_hfather"]
                    nlm_loss_hfather[TrainStep] = nlm_loss
                    acc_train_hfather[TrainStep] = acc_train
                    Task = self.config.TASKs[idx_task]
                elif idx_task == 4:
                    TrainStep = info["TrainStep_hsister"]
                    acc_train = info["acc_train_hsister"]
                    nlm_loss = info["nlm_loss_hsister"]
                    nlm_loss_hsister[TrainStep] = nlm_loss
                    acc_train_hsister[TrainStep] = acc_train
                    Task = self.config.TASKs[idx_task]
                elif idx_task == 5:
                    TrainStep = info["TrainStep_grandparents"]
                    acc_train = info["acc_train_grandparents"]
                    nlm_loss = info["nlm_loss_grandparents"]
                    nlm_loss_grandparents[TrainStep] = nlm_loss
                    acc_train_grandparents[TrainStep] = acc_train
                    Task = self.config.TASKs[idx_task]
                elif idx_task == 6:
                    TrainStep = info["TrainStep_uncle"]
                    acc_train = info["acc_train_uncle"]
                    nlm_loss = info["nlm_loss_uncle"]
                    nlm_loss_uncle[TrainStep] = nlm_loss
                    acc_train_uncle[TrainStep] = acc_train
                    Task = self.config.TASKs[idx_task]
                elif idx_task == 7:
                    TrainStep = info["TrainStep_MGuncle"]
                    acc_train = info["acc_train_MGuncle"]
                    nlm_loss = info["nlm_loss_MGuncle"]
                    nlm_loss_MGuncle[TrainStep] = nlm_loss
                    acc_train_MGuncle[TrainStep] = acc_train
                    Task = self.config.TASKs[idx_task]
                elif idx_task == None:
                    nlm_loss = 0.
                    acc_train = 0.
                    TrainStep = 0.
                    Task = 'Warm-up'
                else:
                    raise NotImplementedError

                print(
                    "Task: {}, Loss: {:.4f}, Acc: {:.3f}, train-step: {}. Played: T0:{} T1:{} T2:{} T3:{} T4:{} T5:{} T6:{} T7:{}, training progress: {}/{}".format(
                        Task,
                        nlm_loss,
                        acc_train,
                        TrainStep,
                        count_outdegree,
                        count_adjacent,
                        count_connectivity,
                        count_hfather,
                        count_hsister,
                        count_grandparents,
                        count_uncle,
                        count_MGuncle,
                        increm,
                        self.config.training_steps,
                    ),
                    end="\r",
                )

        except KeyboardInterrupt as err:
            # Comment the line below to be able to stop the training but keep running
            # raise err
            pass


def test(weights_path, lr_RL, lr_pol,training_steps, rwd_temp, breadth, depth, gen_graph_pmax, gen_graph_pmin,
         outdegree_n, connectivity_dist_limit, gen_directed, gen_graph_method, residual, lr_nlm, num_actors, batch_train, c_init):
    """
    Test the model in a dedicated thread.
    """
    config = NLM_MBRL_Config(
                            lr_RL,
                            lr_pol,
                            training_steps,
                            rwd_temp,
                            breadth,
                            depth,
                            gen_graph_pmax,
                            gen_graph_pmin,
                            outdegree_n,
                            connectivity_dist_limit,
                            gen_directed,
                            gen_graph_method,
                            residual,
                            lr_nlm,
                            num_actors,
                            batch_train,
                            c_init,
                            )
    print("\nTesting...")

    if os.path.exists(weights_path):
        model_weights = torch.load(weights_path)
        print("\nLoad weights done!")

    ray.init()
    self_play_workers = self_play.SelfPlay.remote(
            copy.deepcopy(model_weights),
            numpy.random.randint(1000),
            config,
    )
    steps = 2000

    runs = 1
    warm_up = False

    count_outdegree = 0
    count_adjacent = 0
    count_connectivity = 0
    count_hfather = 0
    count_hsister = 0
    count_grandparents = 0
    count_uncle = 0
    count_MGuncle = 0

    acc_test_outdegree = []
    acc_test_adjacent = []
    acc_test_connectivity = []
    acc_test_hfather = []
    acc_test_hsister = []
    acc_test_grandparents = []
    acc_test_uncle = []
    acc_test_MGuncle = []

    for i_r in range(runs):
        # print("------ i_r", i_r)
        for ee in range(steps):
            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>> ee", ee)
            history, task_id = ray.get(self_play_workers.play_game.remote(warm_up, None, config.temperature_threshold, False, "self", config.muzero_player, "test"))

            if task_id == 0:
                count_outdegree += 1
                acc_test_outdegree.append(history.accuracy)
            elif task_id == 1:
                count_adjacent += 1
                acc_test_adjacent.append(history.accuracy)
            elif task_id == 2:
                count_connectivity += 1
                acc_test_connectivity.append(history.accuracy)
            elif task_id == 3:
                count_hfather += 1
                acc_test_hfather.append(history.accuracy)
            elif task_id == 4:
                count_hsister += 1
                acc_test_hsister.append(history.accuracy)
            elif task_id == 5:
                count_grandparents += 1
                acc_test_grandparents.append(history.accuracy)
            elif task_id == 6:
                count_uncle += 1
                acc_test_uncle.append(history.accuracy)
            elif task_id == 7:
                count_MGuncle += 1
                acc_test_MGuncle.append(history.accuracy)


    print("acc_test_outdegree:", np.array(acc_test_outdegree).mean())
    print("acc_test_adjacent:", np.array(acc_test_adjacent).mean())
    print("acc_test_connectivity:", np.array(acc_test_connectivity).mean())
    print("acc_test_hfather:", np.array(acc_test_hfather).mean())
    print("acc_test_hsister:", np.array(acc_test_hsister).mean())
    print("acc_test_grandparents:", np.array(acc_test_grandparents).mean())
    print("acc_test_uncle:", np.array(acc_test_uncle).mean())
    print("acc_test_MGuncle:", np.array(acc_test_MGuncle).mean())

    print("count_outdegree:", count_outdegree)
    print("count_adjacent:", count_adjacent)
    print("count_connectivity:", count_connectivity)
    print("count_hfather:", count_hfather)
    print("count_hsister:", count_hsister)
    print("count_grandparents:", count_grandparents)
    print("count_uncle:", count_uncle)
    print("count_MGuncle:", count_MGuncle)

    ray.shutdown()



if __name__ == "__main__":

    runs = ARGS.runs
    training_steps = ARGS.train_len
    lr_RL = ARGS.lr_val
    lr_pol = ARGS.lr_pol
    rwd_temp = ARGS.rwd_temp
    breadth = ARGS.breadth
    depth = ARGS.depth
    gen_graph_pmax = ARGS.gen_graph_pmax
    gen_graph_pmin = ARGS.gen_graph_pmin
    outdegree_n = ARGS.outdegree_n
    connectivity_dist_limit = ARGS.connectivity_dist_limit
    gen_directed = ARGS.gen_directed
    gen_graph_method = ARGS.gen_graph_method
    residual = ARGS.residual
    lr_nlm = ARGS.lr_nlm
    num_actors = ARGS.num_actors
    batch_train = ARGS.batch_train
    c_init = ARGS.c_init
    task_mode = ARGS.task_mode

    if task_mode == 'train':
        ##### training entry #####
        for _ in range(runs):
            config = NLM_MBRL_Config(
                                    lr_RL,
                                    lr_pol,
                                    training_steps,
                                    rwd_temp,
                                    breadth,
                                    depth,
                                    gen_graph_pmax,
                                    gen_graph_pmin,
                                    outdegree_n,
                                    connectivity_dist_limit,
                                    gen_directed,
                                    gen_graph_method,
                                    residual,
                                    lr_nlm,
                                    num_actors,
                                    batch_train,
                                    c_init,
                                        )
            nlm_muzero = NLM_MBRL(config)
            nlm_muzero.train()
        print(">>>> Multi-task learning done! <<<<")
    elif task_mode == 'test':
        ##### testing entry #####
        weights_path = "saved_model/model.weights"
        test(weights_path, lr_RL, lr_pol, training_steps, rwd_temp, breadth, depth, gen_graph_pmax,
             gen_graph_pmin, outdegree_n, connectivity_dist_limit, gen_directed, gen_graph_method, residual, lr_nlm,
             num_actors, batch_train, c_init)
    else:
        raise NotImplementedError
