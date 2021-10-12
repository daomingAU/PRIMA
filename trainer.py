import time
import numpy
import ray
import torch
import models
from torch.optim import AdamW
from OpsAsAct_net.thutils import binary_accuracy
from OpsAsAct_net.nn.neural_logic.modules._utils import meshgrid_exclude_self
import jacinle.random as jrandom
import copy

@ray.remote
class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(self, initial_weights, config):
        self.config = config
        self.training_step = 0
        self.TrainStep_outdegree = 0
        self.TrainStep_adjacent = 0
        self.TrainStep_connectivity = 0
        self.TrainStep_hfather = 0
        self.TrainStep_hsister = 0
        self.TrainStep_grandparents = 0
        self.TrainStep_uncle = 0
        self.TrainStep_MGuncle = 0

        # Initialize the network
        self.model = models.NLM_MBRL_Network(self.config)
        self.model.set_weights(initial_weights)
        self.model.to(torch.device(config.training_device))
        self.model.train()


        if self.config.optimizer == "AdamW":
            nlm_params = list(self.model.LogMac_layers.parameters()) + list(self.model.pred_adjacent.parameters()) + list(self.model.pred_outdegree.parameters()) \
                        + list(self.model.pred_hfather.parameters()) + list(self.model.pred_hsister.parameters()) + list(self.model.pred_connectivity.parameters()) \
                        + list(self.model.pred_grandparents.parameters()) + list(self.model.pred_uncle.parameters()) + list(self.model.pred_MGuncle.parameters())
            pol_params = list(self.model.prediction_policy_network.parameters())
            val_params = list(self.model.prediction_value_network.parameters())
            self.optimizer = AdamW([
                {'params': val_params},
                {'params': pol_params, 'lr': self.config.lr_pol},
                {'params': nlm_params, 'lr': self.config.lr_nlm},
            ], lr=self.config.lr_RL)
        else:
            raise NotImplementedError("{} is not implemented. You can change the optimizer manually in trainer.py.")

    def continuous_update_weights(self, replay_buffer_list, shared_storage_worker):
        while ray.get(replay_buffer_list[0].get_self_play_count.remote()) < self.config.num_warm_ups \
                or ray.get(replay_buffer_list[1].get_self_play_count.remote()) < self.config.num_warm_ups\
                or ray.get(replay_buffer_list[2].get_self_play_count.remote()) < self.config.num_warm_ups\
                or ray.get(replay_buffer_list[3].get_self_play_count.remote()) < self.config.num_warm_ups\
                or ray.get(replay_buffer_list[4].get_self_play_count.remote()) < self.config.num_warm_ups\
                or ray.get(replay_buffer_list[5].get_self_play_count.remote()) < self.config.num_warm_ups\
                or ray.get(replay_buffer_list[6].get_self_play_count.remote()) < self.config.num_warm_ups\
                or ray.get(replay_buffer_list[7].get_self_play_count.remote()) < self.config.num_warm_ups:
            time.sleep(0.01)

        shared_storage_worker.set_info.remote("warm_up", False)
        self.model.prediction_policy_network.act_mask = True

        #### Training loop
        while self.training_step < self.config.training_steps:
            idx_task = numpy.random.choice(self.config.len_tasks, 1, p=self.config.prob_task_train).item()
            #################
            index_batch, batch = ray.get(replay_buffer_list[idx_task].get_batch.remote(self.model.get_weights()))
            #################
            (priorities,
             acc_train,
             total_loss,
             nlm_loss,
             muzero_loss,
             value_loss,
             reward_loss,
             policy_loss,
             ) = self.update_weights(batch)

            # Save to the shared storage
            if self.training_step == self.config.training_steps:
                shared_storage_worker.set_weights.remote(self.model.get_weights())
            shared_storage_worker.set_info.remote("training_step", self.training_step)
            shared_storage_worker.set_info.remote("task_id", idx_task)
            if idx_task == 0:
                self.TrainStep_outdegree += 1
                shared_storage_worker.set_info.remote("TrainStep_outdegree", self.TrainStep_outdegree)
                shared_storage_worker.set_info.remote("acc_train_outdegree", acc_train)
                shared_storage_worker.set_info.remote("nlm_loss_outdegree", nlm_loss)
            elif idx_task == 1:
                self.TrainStep_adjacent += 1
                shared_storage_worker.set_info.remote("TrainStep_adjacent", self.TrainStep_adjacent)
                shared_storage_worker.set_info.remote("acc_train_adjacent", acc_train)
                shared_storage_worker.set_info.remote("nlm_loss_adjacent", nlm_loss)
            elif idx_task == 2:
                self.TrainStep_connectivity += 1
                shared_storage_worker.set_info.remote("TrainStep_connectivity", self.TrainStep_connectivity)
                shared_storage_worker.set_info.remote("acc_train_connectivity", acc_train)
                shared_storage_worker.set_info.remote("nlm_loss_connectivity", nlm_loss)
            elif idx_task == 3:
                self.TrainStep_hfather += 1
                shared_storage_worker.set_info.remote("TrainStep_hfather", self.TrainStep_hfather)
                shared_storage_worker.set_info.remote("acc_train_hfather", acc_train)
                shared_storage_worker.set_info.remote("nlm_loss_hfather", nlm_loss)
            elif idx_task == 4:
                self.TrainStep_hsister += 1
                shared_storage_worker.set_info.remote("TrainStep_hsister", self.TrainStep_hsister)
                shared_storage_worker.set_info.remote("acc_train_hsister", acc_train)
                shared_storage_worker.set_info.remote("nlm_loss_hsister", nlm_loss)
            elif idx_task == 5:
                self.TrainStep_grandparents += 1
                shared_storage_worker.set_info.remote("TrainStep_grandparents", self.TrainStep_grandparents)
                shared_storage_worker.set_info.remote("acc_train_grandparents", acc_train)
                shared_storage_worker.set_info.remote("nlm_loss_grandparents", nlm_loss)
            elif idx_task == 6:
                self.TrainStep_uncle += 1
                shared_storage_worker.set_info.remote("TrainStep_uncle", self.TrainStep_uncle)
                shared_storage_worker.set_info.remote("acc_train_uncle", acc_train)
                shared_storage_worker.set_info.remote("nlm_loss_uncle", nlm_loss)
            elif idx_task == 7:
                self.TrainStep_MGuncle += 1
                shared_storage_worker.set_info.remote("TrainStep_MGuncle", self.TrainStep_MGuncle)
                shared_storage_worker.set_info.remote("acc_train_MGuncle", acc_train)
                shared_storage_worker.set_info.remote("nlm_loss_MGuncle", nlm_loss)
            else:
                raise NotImplementedError

            time.sleep(self.config.sleep_time(idx_task))

    def update_weights(self, batch):
        """
        Perform one training step.
        """
        (
            feature_axis,
            task_id,
            observation_batch,
            action_batch,
            target_value,
            target_reward,
            target_policy,
            weight_batch,
            gradient_scale_batch,
            target_batch,
        ) = batch

        target_value_scalar = numpy.array(target_value, dtype=numpy.float32)
        priorities = numpy.zeros_like(target_value_scalar)

        device = next(self.model.parameters()).device
        if self.config.PER:
            weight_batch = torch.tensor(weight_batch.copy()).float().to(device)
        observation_batch = observation_batch
        action_batch = torch.tensor(action_batch).to(device) ## exclude the first action in history, but add a random action instead
        target_value = torch.tensor(target_value).float().to(device)
        target_reward = torch.tensor(target_reward).float().to(device)
        target_policy = torch.tensor(target_policy).float().to(device)
        gradient_scale_batch = torch.tensor(gradient_scale_batch).float().to(device)

        target_batch = torch.cat(target_batch, dim=0).float().to(device)

        target_value = models.scalar_to_support(target_value, self.config.support_size)
        target_reward = models.scalar_to_support(target_reward, self.config.support_size)

        len_act_batch = len(action_batch[0])
        act_batch = []
        for la in range(len_act_batch):
            act_batch.append([action_batch[ba][la] for ba in range(self.config.batch_size_opt)])

        len_obs_batch = len(observation_batch[0])
        obs_batch = []
        for ll in range(len_obs_batch):
            obs_batch.append(torch.cat([observation_batch[jj][ll] for jj in range(self.config.batch_size_opt)], dim=0).float().to(device))
        #################
        ## Generate predictions
        value, reward, policy_logits, hidden_state = self.model.initial_inference(obs_batch)
        predictions = [(value, reward, policy_logits)]

        for i in range(len(act_batch) - 1):
            value, _, reward, policy_logits, hidden_state = self.model.recurrent_inference(i, hidden_state, act_batch[i])
            predictions.append((value, reward, policy_logits))

        LogMac_outputs = hidden_state
        feature_mlp = LogMac_outputs[feature_axis]

        ## Compute losses
        value_loss, reward_loss, policy_loss = (0, 0, 0)

        for i in range(len(predictions)):
            value, reward, policy_logits = predictions[i]
            if i == 0:
                (current_value_loss, _, current_policy_loss) = self.loss_function(
                    value.squeeze(-1),
                    None,
                    policy_logits,
                    target_value[:, i],
                    None,
                    target_policy[:, i],
                )
            elif i == len(predictions) - 1:
                (current_value_loss, _, current_policy_loss) = self.loss_function(
                    None,
                    None,
                    None,
                    None,
                    target_reward[:, i],
                    None,
                )
            else:
                (current_value_loss, _, current_policy_loss) = self.loss_function(
                    value.squeeze(-1),
                    None,
                    policy_logits,
                    target_value[:, i],
                    target_reward[:, i],
                    target_policy[:, i],
                )

            value_loss += current_value_loss
            policy_loss += current_policy_loss

            # Compute priorities for the prioritized replay (See paper appendix Training)
            pred_value_scalar = (
                models.support_to_scalar(value, self.config.support_size)
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze()
            )

            priorities[:, i] = (
                numpy.abs(pred_value_scalar - target_value_scalar[:, i])
                ** self.config.PER_alpha
            )

        muzero_loss = value_loss * self.config.value_loss_weight + reward_loss* self.config.reward_loss_weight + policy_loss* self.config.policy_loss_weight
        if self.config.PER:
            muzero_loss *= weight_batch

        if feature_axis == 1:
            if task_id == 1:
                pred_nlm = self.model.pred_adjacent(feature_mlp)
            elif task_id == 0:
                pred_nlm = self.model.pred_outdegree(feature_mlp).squeeze(-1)
            elif task_id == 3:
                pred_nlm = self.model.pred_hfather(feature_mlp).squeeze(-1)
            elif task_id == 4:
                pred_nlm = self.model.pred_hsister(feature_mlp).squeeze(-1)
            else:
                raise NotImplementedError
        elif feature_axis == 2:
            if task_id == 2:
                pred_nlm = self.model.pred_connectivity(feature_mlp).squeeze(-1)
                pred_nlm = meshgrid_exclude_self(pred_nlm)
            elif task_id == 5:
                pred_nlm = self.model.pred_grandparents(feature_mlp).squeeze(-1)
            elif task_id == 6:
                pred_nlm = self.model.pred_uncle(feature_mlp).squeeze(-1)
            elif task_id == 7:
                pred_nlm = self.model.pred_MGuncle(feature_mlp).squeeze(-1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        nlm_loss = self.model.loss(pred_nlm, target_batch)
        acc_train = binary_accuracy(target_batch, torch.sigmoid(pred_nlm))['accuracy']
        nlm_loss_weight = 1000 if task_id == 5 or task_id == 6 or task_id == 7 else 100
        ### total loss = muzero_loss + nlm_loss
        total_loss = muzero_loss.mean() * self.config.muzero_loss_weight + nlm_loss * nlm_loss_weight
        #################
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        #################
        self.training_step += 1

        return (
            priorities,
            # For log purpose
            acc_train,
            total_loss.item(),
            nlm_loss.item(),
            muzero_loss.mean().item(),
            value_loss.mean().item(),
            reward_loss,
            policy_loss.mean().item(),
        )


    @staticmethod
    def loss_function(value, reward, policy_logits, target_value, target_reward, target_policy):
        if value is not None and target_value is not None:
            value_loss = (-target_value * torch.nn.LogSoftmax(dim=1)(value)).sum(1)
        else:
            value_loss = 0

        if reward is not None and target_reward is not None:
            reward_loss = (-target_reward * torch.nn.LogSoftmax(dim=1)(reward)).sum(1)
        else:
            reward_loss = 0

        if policy_logits is not None and target_policy is not None:
            policy_loss = (-target_policy * policy_logits).sum(1)
        else:
            policy_loss = 0
        return value_loss, reward_loss, policy_loss
