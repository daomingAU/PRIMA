import copy
import math
import numpy
import ray
import torch
import models
from OpsAsAct_net.dataset.graph import GraphOutDegreeDataset, \
    GraphConnectivityDataset, GraphAdjacentDataset, FamilyTreeDataset
from OpsAsAct_net.thutils import binary_accuracy
from OpsAsAct_net.nn.neural_logic.modules._utils import meshgrid_exclude_self
from jactorch.data.dataloader import JacDataLoader
from jacinle.utils.container import GView
import jacinle.random as jrandom
import gc

@ray.remote
class SelfPlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, initial_weights, seed, config):
        self.config = config

        # Fix random generator seed
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        jrandom.reset_global_seed(seed)

        # Initiaget_self_play_countize the network
        self.model = models.NLM_MBRL_Network(self.config)
        self.model.set_weights(initial_weights)
        self.model.to(torch.device("cpu"))
        self.model.eval()

    def continuous_self_play(self, shared_storage, replay_buffer_list, test_mode=False):

        while True:
            self.model.set_weights(copy.deepcopy(ray.get(shared_storage.get_weights.remote())))

            if not test_mode:
                ct_outdegree = ray.get(shared_storage.get_info.remote())["TrainStep_outdegree"]
                ct_adjacent = ray.get(shared_storage.get_info.remote())["TrainStep_adjacent"]
                ct_connectivity = ray.get(shared_storage.get_info.remote())["TrainStep_connectivity"]
                ct_hfather = ray.get(shared_storage.get_info.remote())["TrainStep_hfather"]
                ct_hsister = ray.get(shared_storage.get_info.remote())["TrainStep_hsister"]
                ct_grandparents = ray.get(shared_storage.get_info.remote())["TrainStep_grandparents"]
                ct_uncle = ray.get(shared_storage.get_info.remote())["TrainStep_uncle"]
                ct_MGuncle = ray.get(shared_storage.get_info.remote())["TrainStep_MGuncle"]

                ct_list = [ct_outdegree, ct_adjacent, ct_connectivity, ct_hfather, ct_hsister, ct_grandparents, ct_uncle, ct_MGuncle]

                game_history, task_id = self.play_game(
                    ray.get(shared_storage.get_info.remote())["warm_up"],
                    ct_list,
                    self.config.temperature_threshold,
                    False,
                    "self",
                    0,
                    "train",
                )
                replay_buffer_list[task_id].save_game.remote(game_history)

    def make_dataset(self, epoch_size, is_train):
        idx_task = numpy.random.choice(self.config.len_tasks, 1, p=self.config.prob_task_uniform).item()
        pmin, pmax = self.config.gen_graph_pmin, self.config.gen_graph_pmax
        if idx_task == 0:
            if is_train:
                n = self.config.NumObj_graph_train
            else:
                n = self.config.NumObj_graph_test
            return GraphOutDegreeDataset(
                self.config.outdegree_n,
                epoch_size,
                n,
                pmin=pmin,
                pmax=pmax,
                directed=self.config.gen_directed,
                gen_method=self.config.gen_graph_method)
        elif idx_task == 1:
            if is_train:
                n = self.config.NumObj_graph_train
            else:
                n = self.config.NumObj_graph_test
            return GraphAdjacentDataset(
                self.config.gen_graph_colors,
                epoch_size,
                n,
                pmin=pmin,
                pmax=pmax,
                directed=self.config.gen_directed,
                gen_method=self.config.gen_graph_method,
                is_train=is_train,
                is_mnist_colors=False)
        elif idx_task == 2:
            if is_train:
                n = self.config.NumObj_graph_train
            else:
                n = self.config.NumObj_graph_test
            nmin, nmax = n, n
            if is_train and self.config.NLM_recursion:
                nmin = 2
            return GraphConnectivityDataset(
                self.config.connectivity_dist_limit,
                epoch_size,
                nmin,
                pmin,
                nmax,
                pmax,
                directed=self.config.gen_directed,
                gen_method=self.config.gen_graph_method)
        else:
            if is_train:
                n = self.config.NumObj_FTree_train
            else:
                n = self.config.NumObj_FTree_test
            return FamilyTreeDataset(self.config.TASKs[idx_task], epoch_size, n, p_marriage=1.0, balance_sample=False)

    data_iterator = {}
    def prepare_dataset(self, mode='train'):
        assert mode in ['train', 'test']
        if mode == 'train':
            batch_size = self.config.batchsize_NLM
            epoch_size = self.config.epoch_size
        else:
            batch_size = self.config.test_batchsize
            epoch_size = self.config.epoch_size

        dataset = self.make_dataset(epoch_size * batch_size, mode == 'train')
        dataloader = JacDataLoader(
            dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=max(epoch_size, 1))
        self.data_iterator[mode] = dataloader.__iter__()

    def play_game(self, warm_up, step_list, temperature_threshold, render, opponent, muzero_player, mode):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """

        game_history = GameHistory()
        game_history.action_history.append(0)
        game_history.reward_history.append(0)
        game_history.to_play_history.append(0)
        ###############################
        self.prepare_dataset(mode)
        feed_dict = self.data_iterator[mode].next()
        feed_dict = GView(feed_dict)
        number = feed_dict.n.item()
        feature_axis = feed_dict.out_arity.item()
        task_id = feed_dict.task_id.item()
        game_history.feature_axis = feature_axis
        game_history.task_id = task_id
        if step_list is not None:
            temperature = self.config.visit_softmax_temperature_fn(step_list[task_id], self.config.prob_task_train[task_id])
        else:
            temperature = 0
        ###############################
        if self.config.TASKs[task_id] == "adjacent":
            states = feed_dict.states.float()
        else:
            states = None
        nullary = feed_dict.task_OneHot.float()
        relations = feed_dict.relations.float()
        target = feed_dict.target.float()
        ##################################
        b0 = torch.zeros(self.config.batchsize_NLM, self.config.nlm_attributes)
        b0[:, :nullary.size(-1)] = nullary
        b1 = torch.zeros(self.config.batchsize_NLM, number, self.config.nlm_attributes)
        if states is not None:
            b1[:, :, :states.size(-1)] = states
        b2 = torch.zeros(self.config.batchsize_NLM, number, number,
                         self.config.nlm_attributes)
        b2[:, :, :, :relations.size(-1)] = relations

        observation = [None for _ in range(self.config.breadth + 1)]
        observation[0] = b0
        observation[1] = b1
        observation[2] = b2

        if self.config.breadth == 3:
            b3 = torch.zeros(self.config.batchsize_NLM, number, number, number, self.config.nlm_attributes)
            observation[3] = b3
        elif self.config.breadth == 4:
            b3 = torch.zeros(self.config.batchsize_NLM, number, number, number, self.config.nlm_attributes)
            observation[3] = b3
            b4 = torch.zeros(self.config.batchsize_NLM, number, number,
                             number, number, self.config.nlm_attributes)
            observation[4] = b4
        #########
        game_history.observation_history.append(observation)
        game_history.target = target
        #########

        with torch.no_grad():

            f = observation.copy()
            for i in range(self.config.depth):
                # Choose the action
                if opponent == "self" or muzero_player == 0:
                    if mode == 'train':
                        root, mcts_info = MCTS(self.config).run(
                            task_id,
                            feature_axis,
                            self.model,
                            f,
                            self.config.action_space,
                            0,
                            False if temperature == 0 else True,
                            i,
                            target,
                            game_history.reward_history[i],
                            int(self.config.num_simulations[i]) if not warm_up else 200,
                        )
                        action = self.select_action(
                            root,
                            temperature
                            if not temperature_threshold
                               or len(game_history.action_history) < temperature_threshold
                            else 0,
                        )
                    elif mode == 'test':
                        policy_logits = self.model.prediction_policy_network(f)
                        prob = policy_logits.exp()
                        action = prob.argmax().item()
                    else:
                        raise NotImplementedError

                    if type(action) is not int:
                        action = action.item()

                    f, reward_layer, _ = self.model.dynamics(i, f, action)

                    game_history.action_history.append(action)
                    game_history.observation_history.append(f)
                    game_history.reward_history.append(reward_layer)

                    if i == (self.config.depth - 1):
                        LogMac_outputs = f

                        feature_mlp = LogMac_outputs[feature_axis]
                        if feature_axis == 1:
                            if task_id == 1:
                                pred_mlp = self.model.pred_adjacent(feature_mlp)
                            elif task_id == 0:
                                pred_mlp = self.model.pred_outdegree(feature_mlp).squeeze(-1)
                            elif task_id == 3:
                                pred_mlp = self.model.pred_hfather(feature_mlp).squeeze(-1)
                            elif task_id == 4:
                                pred_mlp = self.model.pred_hsister(feature_mlp).squeeze(-1)
                            else:
                                raise NotImplementedError
                        elif feature_axis == 2:
                            if task_id == 2:
                                pred_mlp = self.model.pred_connectivity(feature_mlp).squeeze(-1)
                                pred_mlp = meshgrid_exclude_self(pred_mlp)
                            elif task_id == 5:
                                pred_mlp = self.model.pred_grandparents(feature_mlp).squeeze(-1)
                            elif task_id == 6:
                                pred_mlp = self.model.pred_uncle(feature_mlp).squeeze(-1)
                            elif task_id == 7:
                                pred_mlp = self.model.pred_MGuncle(feature_mlp).squeeze(-1)
                            else:
                                raise NotImplementedError
                        else:
                            raise NotImplementedError

                        accuracy = binary_accuracy(target, torch.sigmoid(pred_mlp))['accuracy']
                        game_history.accuracy = accuracy
                        game_history.reward_history[-1] = game_history.reward_history[-1] + (
                                    accuracy ** self.config.rwd_temp) * self.config.final_reward_weight

                    if mode == 'train':
                        game_history.store_search_statistics(root, self.config.action_space)
                    game_history.to_play_history.append(0)
                    ############
                    if mode == 'train':
                        del root
                        gc.collect()
                    ############
        return game_history, task_id


    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function 
        in the config.
        """
        visit_counts = numpy.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[numpy.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = numpy.random.choice(actions)
        else:
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = numpy.random.choice(actions, p=visit_count_distribution)

        return action

class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config

    def run(
        self,
        task_id,
        feature_axis,
        model,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
        i_layer,
        target,
        reward,
        num_simulations,
    ):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        root = Node(0)
        observation = [ele.float().to(next(model.parameters()).device) if type(ele) is torch.Tensor else ele for ele in observation]
        root_predicted_value, _, policy_logits, hidden_state = model.initial_inference(observation)
        root_predicted_value = models.support_to_scalar(root_predicted_value, self.config.support_size).item()

        root.expand(legal_actions, to_play, reward, policy_logits, hidden_state)

        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()
        max_tree_depth = 0
        for k in range(num_simulations):
            virtual_to_play = to_play
            node = root
            search_path = [node]
            current_tree_depth = 0

            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)

                if virtual_to_play + 1 < len(self.config.players):
                    virtual_to_play = self.config.players[virtual_to_play + 1]
                else:
                    virtual_to_play = self.config.players[0]

            ## Inside the search tree we use the dynamics function to obtain the next hidden
            ## state given an action and the previous hidden state
            parent = search_path[-2]
            value, reward, _, policy_logits, hidden_state = model.recurrent_inference(
                i_layer + current_tree_depth - 1,
                parent.hidden_state,
                action.item(),
            )

            if current_tree_depth < self.config.depth - i_layer:
                value = models.support_to_scalar(value, self.config.support_size).item()
                node.expand(
                    self.config.action_space,
                    virtual_to_play,
                    reward,
                    policy_logits,
                    hidden_state,
                )
            else:
                LogMac_outputs = hidden_state

                feature_mlp = LogMac_outputs[feature_axis]
                if feature_axis == 1:
                    if task_id == 1:
                        pred_mlp = model.pred_adjacent(feature_mlp)
                    elif task_id == 0:
                        pred_mlp = model.pred_outdegree(feature_mlp).squeeze(-1)
                    elif task_id == 3:
                        pred_mlp = model.pred_hfather(feature_mlp).squeeze(-1)
                    elif task_id == 4:
                        pred_mlp = model.pred_hsister(feature_mlp).squeeze(-1)
                    else:
                        raise NotImplementedError
                elif feature_axis == 2:
                    if task_id == 2:
                        pred_mlp = model.pred_connectivity(feature_mlp).squeeze(-1)
                        pred_mlp = meshgrid_exclude_self(pred_mlp)
                    elif task_id == 5:
                        pred_mlp = model.pred_grandparents(feature_mlp).squeeze(-1)
                    elif task_id == 6:
                        pred_mlp = model.pred_uncle(feature_mlp).squeeze(-1)
                    elif task_id == 7:
                        pred_mlp = model.pred_MGuncle(feature_mlp).squeeze(-1)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
                accuracy = binary_accuracy(target, torch.sigmoid(pred_mlp))['accuracy']
                reward = reward + (accuracy ** self.config.rwd_temp) * self.config.final_reward_weight
                value = 0
                node.reward = reward

            self.backpropagate(search_path, value, virtual_to_play, min_max_stats)
            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        return root, extra_info


    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        max_ucb = max(
            self.ucb_score(node, child, min_max_stats)
            for action, child in node.children.items()
        )
        action = numpy.random.choice(
            [
                action
                for action, child in node.children.items()
                if self.ucb_score(node, child, min_max_stats) == max_ucb
            ]
        )
        return action, node.children[action]


    def ucb_score(self, parent, child, min_max_stats):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )

        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        if child.visit_count > 0:
            value_score = min_max_stats.normalize(child.value())
        else:
            value_score = 0
        return prior_score + value_score

    def backpropagate(self, search_path, value, to_play, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        if len(self.config.players) == 1:
            for node in reversed(search_path):
                if type(node.reward) is torch.Tensor:
                    node.reward = node.reward.item()
                value = node.reward + self.config.discount * value

                node.value_sum += value
                node.visit_count += 1
                min_max_stats.update(node.value())

        else:
            raise NotImplementedError("More than two player mode not implemented.")

THRESHOLD = 5e-5

class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, to_play, reward, policy_logits, hidden_state):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state
        policy_values = policy_logits[0].exp().tolist()
        policy = {a: policy_values[i] for i, a in enumerate(actions)}

        for action, p in policy.items():
            self.children[action] = Node(p)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            if self.children[a].prior > THRESHOLD:
                self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.child_visits = []
        self.root_values = []
        self.reanalysed_predicted_root_values = None
        self.priorities = None
        self.game_priority = None
        self.target = None
        self.accuracy = None
        self.total_ops = None
        ### task
        self.feature_axis = None
        self.task_id = None

    def store_search_statistics(self, root, action_space):
        # Turn visit count from root into a policy
        if root is not None:
            sum_visits = sum(child.visit_count for child in root.children.values())
            self.child_visits.append(
                [
                    root.children[a].visit_count / sum_visits
                    if a in root.children
                    else 0
                    for a in action_space
                ]
            )

            self.root_values.append(root.value())
        else:
            self.root_values.append(None)

class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
