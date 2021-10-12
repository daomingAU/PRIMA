import datetime
import os
import numpy
import torch


class NLM_MBRL_Config:
    """
    Hyper-parameter settings
    """
    def __init__(self,
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
                 batch_size_train,
                 c_init,
                 ):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 123  # Seed for numpy, torch and the game

        self.TASKs = ['outdegree', 'adjacent', 'connectivity', 'has-father', 'has-sister', 'grandparents', 'uncle', 'maternal-great-uncle']
        self.len_tasks = len(self.TASKs)
        self.prob_task_train = [0.1, 0.12, 0.12, 0.1, 0.1, 0.13, 0.13, 0.20]
        self.prob_task_uniform = [1. / self.len_tasks for _ in range(self.len_tasks)]

        self.balance_sample = False
        self.num_warm_ups = 200   ## warm-ups before starting training

        ### Game
        self.players = [i for i in range(1)]  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_actors = num_actors  # Number of simultaneous threads self-playing to feed the replay buffer
        self.num_simulations = [1800, 1200, 1000, 800] # Number of simulations
        self.discount = 1.  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = .3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = c_init

        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 10

        ### Training
        self.results_path = os.path.join(os.path.dirname(__file__), "./results", datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.training_steps = training_steps  # Total number of training steps
        self.batch_size_opt = batch_size_train
        self.value_loss_weight = 1.
        self.reward_loss_weight = 1.
        self.policy_loss_weight = 1.
        self.muzero_loss_weight = 1.
        self.nlm_loss_weight = 1000
        # self.training_device = "cuda" if torch.cuda.is_available() else "cpu"  # Train on GPU if available
        self.training_device = "cpu"

        self.optimizer = "AdamW"  # "AdamW",
        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_nlm = lr_nlm
        self.lr_RL = lr_RL
        self.lr_pol = lr_pol
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000

        ### Replay Buffer
        self.window_size = 400  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = depth  # Number of game moves to keep for every batch element
        self.td_steps = depth  # Number of steps in the future to take into account for calculating the target value
        self.use_last_model_value = False  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)

        # Prioritized Replay
        self.PER = False  # Select in priority the elements in the replay buffer which are unexpected for the network
        self.use_max_priority = False  # If False, use the n-step TD error as initial priority. Better for large replay buffer
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1
        self.PER_beta = 1.0

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired self played games per training step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        self.rwd_temp = rwd_temp

        ### config for graph
        self.breadth = breadth                ## breadth of NLM
        self.depth = depth                  ## depth of NLM
        self.gen_graph_pmax = gen_graph_pmax       ## max prob. used to generate graph
        self.gen_graph_pmin = gen_graph_pmin       ## min prob. used to generate graph
        self.gen_graph_colors = 4       ## number of colors in the generated graph
        self.gen_directed = gen_directed       ## True if the graph is directed
        self.gen_graph_method = gen_graph_method
        self.outdegree_n = outdegree_n
        self.connectivity_dist_limit = connectivity_dist_limit
        self.adjacent_pred_colors = 4

        self.nlm_attributes = 8
        self.epoch_size = 1
        self.batchsize_NLM = 1 # batch size for self-play
        self.NumObj_graph_train = 10    ## (training) problem size for graph tasks
        self.NumObj_FTree_train = 20    ## (training) problem size for graph tasks

        ### config for model
        self.NLM_input_dims = [self.nlm_attributes for _ in range(self.breadth + 1)]
        self.NLM_output_dims = self.nlm_attributes
        self.NLM_logic_hidden_dim = []
        self.NLM_exclude_self = True
        self.NLM_residual = residual
        self.NLM_io_residual = False
        self.NLM_recursion = False

        self.final_reward_weight = 100
        self.action_num = 4*8**(self.breadth-1)*4
        self.action_space = [i for i in range(self.action_num)]

        self.matrix = self.calc_matrix(breadth)

        self.NumObj_graph_test = 10     ## (testing) problem size for graph tasks
        self.NumObj_FTree_test = 20     ## (testing) problem size for family tree tasks
        self.test_batchsize = 1
        self.checkpoints_dir = 'saved_model'

    def calc_matrix(self, breadth):
        num_indicators = sum([2 if r == 0 or r == breadth else 3 for r in range(breadth + 1)])
        len_bin = '{0:0' + '{}'.format((breadth - 1) * 3 + 2 * 2) + 'b}'
        prob1_idx = torch.tensor([list(map(int, len_bin.format(i))) for i in range(2 ** num_indicators)]).float()
        prob0_idx = 1. - prob1_idx
        Mat = torch.cat([prob0_idx.transpose(0, 1), prob1_idx.transpose(0, 1)], dim=0)
        return Mat

    def sleep_time(self, task):
        if task == 0 or task == 2:
            return 1.4
        elif task == 1:
            return 1.8
        else:
            return .04

    def visit_softmax_temperature_fn(self, trained_steps, prob_task):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.4 * self.training_steps*prob_task:
            return 1.0
        elif trained_steps < 0.6 * self.training_steps*prob_task:
            return 0.5
        elif trained_steps < 0.7 * self.training_steps*prob_task:
            return 0.2
        else:
            return 0.1


