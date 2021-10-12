import collections
import copy
import numpy
import ray
import torch
import models


@ray.remote
class ReplayBuffer:
    """
    Class which run in a dedicated thread to store played games and generate batch.
    """

    def __init__(self, config):
        self.config = config
        self.buffer = {}
        self.self_play_count = 0
        self.total_samples = 0

    def save_game(self, game_history):
        if self.config.PER:
            if game_history.priorities is not None:
                # Avoid read only array when loading replay buffer from disk
                game_history.priorities = numpy.copy(game_history.priorities)
            else:
                # Initial priorities for the prioritized replay (See paper appendix Training)
                priorities = []
                for i, root_value in enumerate(game_history.root_values):
                    priority = (
                            numpy.abs(
                                root_value - self.compute_target_value(game_history, i)
                            )
                            ** self.config.PER_alpha
                    )
                    priorities.append(priority)

                game_history.priorities = numpy.array(priorities, dtype="float32")
                game_history.game_priority = numpy.max(game_history.priorities)

        self.buffer[self.self_play_count] = game_history
        self.self_play_count += 1
        self.total_samples += len(game_history.root_values)

        if self.config.window_size < len(self.buffer):
            del_id = self.self_play_count - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].root_values)
            del self.buffer[del_id]

    def get_self_play_count(self):
        return self.self_play_count

    def get_buffer(self):
        return self.buffer

    def get_batch(self, model_weights):
        (
            feature_axis,
            task_id,
            index_batch,
            observation_batch,
            action_batch,
            reward_batch,
            value_batch,
            policy_batch,
            gradient_scale_batch,
            target_batch,
        ) = (None, None, [], [], [], [], [], [], [], [])
        weight_batch = [] if self.config.PER else None

        if self.config.use_last_model_value:
            self.model.set_weights(model_weights)

        for _ in range(self.config.batch_size_opt):
            game_id, game_history, game_prob = self.sample_game()
            game_pos, pos_prob = self.sample_position(game_history)

            values, rewards, policies, actions = self.make_target(
                game_history, game_pos
            )

            index_batch.append([game_id, game_pos])
            feature_axis = game_history.feature_axis
            task_id = game_history.task_id
            observation_batch.append(game_history.observation_history[0].copy())
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)

            gradient_scale_batch.append(
                [
                    min(
                        self.config.num_unroll_steps,
                        len(game_history.action_history) - game_pos,
                    )
                ]
                * len(actions)
            )

            target_batch.append(game_history.target)

            if self.config.PER:
                weight_batch.append(1 / (self.total_samples * game_prob * pos_prob))

        if self.config.PER:
            weight_batch = numpy.array(weight_batch, dtype="float32") / max(
                weight_batch
            )

        return (
            index_batch,
            (
                feature_axis,
                task_id,
                observation_batch,
                action_batch,
                value_batch,
                reward_batch,
                policy_batch,
                weight_batch,
                gradient_scale_batch,
                target_batch,
            ),
        )

    def sample_game(self, force_uniform=False):
        """
        Sample game from buffer either uniformly or according to some priority.
        See paper appendix Training.
        """
        game_prob = None
        if self.config.PER and not force_uniform:
            game_probs = numpy.array(
                [game_history.game_priority for game_history in self.buffer.values()],
                dtype="float32",
            )
            game_probs /= numpy.sum(game_probs)
            game_index = numpy.random.choice(len(self.buffer), p=game_probs)
            game_prob = game_probs[game_index]
        else:
            game_index = numpy.random.choice(len(self.buffer))
        game_id = self.self_play_count - len(self.buffer) + game_index

        return game_id, self.buffer[game_id], game_prob

    def sample_position(self, game_history, force_uniform=False):
        """
        Sample position from game either uniformly or according to some priority.
        See paper appendix Training.
        """
        position_prob = None
        if self.config.PER and not force_uniform:
            position_probs = game_history.priorities / sum(game_history.priorities)
            position_index = numpy.random.choice(len(position_probs), p=position_probs)
            position_prob = position_probs[position_index]
        else:
            position_index = 0
        return position_index, position_prob

    def update_priorities(self, priorities, index_info):
        """
        Update game and position priorities with priorities calculated during the training.
        See Distributed Prioritized Experience Replay https://arxiv.org/abs/1803.00933
        """
        for i in range(len(index_info)):
            game_id, game_pos = index_info[i]

            # The element could have been removed since its selection and training
            if next(iter(self.buffer)) <= game_id:
                # Update position priorities
                priority = priorities[i, :]
                start_index = game_pos
                end_index = min(
                    game_pos + len(priority), len(self.buffer[game_id].priorities)
                )
                self.buffer[game_id].priorities[start_index:end_index] = priority[
                                                                         : end_index - start_index
                                                                         ]

                # Update game priorities
                self.buffer[game_id].game_priority = numpy.max(
                    self.buffer[game_id].priorities
                )

    def compute_target_value(self, game_history, index):
        bootstrap_index = index + self.config.td_steps
        if bootstrap_index < len(game_history.root_values):
            root_values = (
                game_history.root_values
                if game_history.reanalysed_predicted_root_values is None
                else game_history.reanalysed_predicted_root_values
            )
            last_step_value = (
                root_values[bootstrap_index]
                if game_history.to_play_history[bootstrap_index]
                   == game_history.to_play_history[index]
                else -root_values[bootstrap_index]
            )

            value = last_step_value * self.config.discount ** self.config.td_steps
        else:
            value = 0
        for i, reward in enumerate(
                game_history.reward_history[index + 1 : bootstrap_index + 1]
        ):
            value += (
                         reward
                         if game_history.to_play_history[index]
                            == game_history.to_play_history[index + 1 + i]
                         else -reward
                     ) * self.config.discount ** i

        return value

    def make_target(self, game_history, state_index):
        """
        Generate targets for every unroll steps.
        """
        target_values, target_rewards, target_policies, actions = [], [], [], []
        for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
            value = self.compute_target_value(game_history, current_index)
            if current_index < len(game_history.root_values):
                target_values.append(value)
                target_rewards.append(game_history.reward_history[current_index])
                target_policies.append(game_history.child_visits[current_index])
                actions.append(game_history.action_history[current_index+1])
            elif current_index == len(game_history.root_values):
                target_values.append(0)
                target_rewards.append(game_history.reward_history[current_index])
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(numpy.random.choice(self.config.action_space))
            else:
                assert current_index <= self.config.depth, "Make target: the horizon should be no more than the depth of NLM!"
                raise NotImplementedError
        return target_values, target_rewards, target_policies, actions
