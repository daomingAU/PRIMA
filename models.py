import math
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import copy
from OpsAsAct_net.nn.neural_logic import LogicLayer, LogicInference, LogitsInference
from prediction_net import Planner, Reasoner_Val

class NLM_MBRL_Network:
    def __new__(cls, config):
        if config.network == "fullyconnected":
            return NLM_MBRL_FullyConnectedNetwork(config)
        else:
            raise NotImplementedError('The network parameter should be "fullyconnected" or "resnet".')


class AbstractNetwork(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, layer_i, encoded_state, action):
        pass

    def get_weights(self):
        return {key: value.cpu() for key, value in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)


class NLM_MBRL_FullyConnectedNetwork(AbstractNetwork):
    """
    Network Architecture for PRIMA/PRISA
    """
    def __init__(self, config):
        super().__init__()

        self.depth = config.depth
        self.breadth = config.breadth
        self.recursion = config.NLM_recursion
        self.io_residual = config.NLM_io_residual
        self.full_support_size = 2 * config.support_size + 1

        self.LogMac_layers = nn.ModuleList()
        LM_current_dims = config.NLM_input_dims.copy()

        for i in range(self.depth):
            # Not support output_dims as list or list[list] yet.
            LM_layer = LogicLayer(self.breadth, LM_current_dims, config.NLM_output_dims, config.NLM_logic_hidden_dim,
                                  config.NLM_exclude_self, config.NLM_residual)
            LM_current_dims = LM_layer.output_dims
            self.LogMac_layers.append(LM_layer)

        LM_output_dims = LM_current_dims

        output_unary = LM_output_dims[1]
        self.pred_adjacent = LogitsInference(output_unary, config.adjacent_pred_colors, [])
        self.pred_outdegree = LogitsInference(output_unary, 1, [])
        self.pred_hfather = LogitsInference(output_unary, 1, [])
        self.pred_hsister = LogitsInference(output_unary, 1, [])

        output_binary = LM_output_dims[2]
        self.pred_connectivity = LogitsInference(output_binary, 1, [])
        self.pred_grandparents = LogitsInference(output_binary, 1, [])
        self.pred_uncle = LogitsInference(output_binary, 1, [])
        self.pred_MGuncle = LogitsInference(output_binary, 1, [])

        self.loss = nn.BCEWithLogitsLoss()

        ### policy network ###
        self.prediction_policy_network = Planner(config.breadth, config.nlm_attributes, config.NLM_logic_hidden_dim, config.matrix)
        ### value network ###
        self.prediction_value_network = Reasoner_Val(config.breadth, config.nlm_attributes, config.NLM_logic_hidden_dim, self.full_support_size)

    def prediction(self, encoded_state):
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logits, value

    def dynamics(self, layer_i, input_state, action):
        Reason_layer = self.LogMac_layers[layer_i]
        out_layer, reward_layer = Reason_layer(input_state, action)
        return out_layer, reward_layer, None

    def initial_inference(self, observation):
        device = observation[1].device
        batchsize = observation[1].size(0)
        policy_logits, value = self.prediction(observation)

        reward = torch.log((torch.zeros(batchsize, self.full_support_size).scatter(1, torch.tensor([[self.full_support_size // 2] for _ in range(batchsize)]).long(), 1))).to(device)
        return value, reward, policy_logits, observation

    def recurrent_inference(self, layer_i, encoded_state, action):
        next_encoded_state, reward_layer, reward_MLP = self.dynamics(layer_i, encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward_layer, reward_MLP, policy_logits, next_encoded_state


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=nn.Identity,
    activation=nn.ReLU):

    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)

def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    probabilities = torch.softmax(logits, dim=1)
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=1, keepdim=True)
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x


def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits


