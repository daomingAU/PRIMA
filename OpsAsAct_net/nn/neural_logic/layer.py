#! /usr/bin/env python3
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implement Neural Logic Layers and Machines."""

import torch
import torch.nn as nn
import numpy
from jacinle.logging import get_logger

from .modules.dimension import Expander, Reducer, Permutation
from .modules.neural_logic import LogicInference, LogitsInference


__all__ = ['LogicLayer', 'LogicMachine']

logger = get_logger(__file__)


def _get_tuple_n(x, n, tp):
  """Get a length-n list of type tp."""
  assert tp is not list
  if isinstance(x, tp):
    x = [x,] * n
  assert len(x) == n, 'Parameters should be {} or list of N elements.'.format(
      tp)
  for i in x:
    assert isinstance(i, tp), 'Elements of list should be {}.'.format(tp)
  return x


class LogicLayer(nn.Module):
  """Logic Layers do one-step differentiable logic deduction.

  The predicates grouped by their number of variables. The inter group deduction
  is done by expansion/reduction, the intra group deduction is done by logic
  model.

  Args:
    breadth: The breadth of the logic layer.
    input_dims: the number of input channels of each input group, should consist
                with the inputs. use dims=0 and input=None to indicate no input
                of that group.
    output_dims: the number of output channels of each group, could
                 use a single value.
    logic_hidden_dim: The hidden dim of the logic model.
    exclude_self: Not allow multiple occurrence of same variable when
                  being True.
    residual: Use residual connections when being True.
  """

  def __init__(
      self,
      breadth,
      input_dims,
      output_dims,
      logic_hidden_dim,
      exclude_self=True,
      residual=False,
  ):
    super().__init__()
    self.max_order = breadth
    self.residual = residual

    input_dims = _get_tuple_n(input_dims, self.max_order + 1, int)
    output_dims = _get_tuple_n(output_dims, self.max_order + 1, int)

    #### logic: MLP; dim_perms: Permutation; dim_expanders: expand; dim_reducers: reduce
    self.Op_mlp, self.red_mlp, self.exp_mlp, self.Op_perms, self.red_perms, self.exp_perms, self.Op_expanders, self.Op_reducers = [nn.ModuleList() for _ in range(8)]

    for i in range(self.max_order + 1):
        DirInp_dim = input_dims[i]
        if DirInp_dim == 0:
            self.Op_perms.append(None)
            self.Op_mlp.append(None)
        else:
            perm = Permutation(i)
            self.Op_perms.append(perm)
            DirInp_dim = perm.get_output_dim(DirInp_dim)
            self.Op_mlp.append(LogitsInference(DirInp_dim, output_dims[i], logic_hidden_dim))
        if i > 0:
            expander = Expander(i - 1)
            self.Op_expanders.append(expander)
            exp_dim = expander.get_output_dim(input_dims[i - 1])
            perm = Permutation(i)
            self.exp_perms.append(perm)
            exp_dim = perm.get_output_dim(exp_dim)
            self.exp_mlp.append(LogitsInference(exp_dim, output_dims[i], logic_hidden_dim))
        else:
            self.Op_expanders.append(None)
            self.exp_perms.append(None)
            self.exp_mlp.append(None)
        if i + 1 < self.max_order + 1:
            reducer = Reducer(i + 1, exclude_self)
            self.Op_reducers.append(reducer)
            red_dim = reducer.get_output_dim(input_dims[i + 1])
            perm = Permutation(i)
            self.red_perms.append(perm)
            red_dim = perm.get_output_dim(red_dim)
            self.red_mlp.append(LogitsInference(red_dim, output_dims[i], logic_hidden_dim))
        else:
            self.Op_reducers.append(None)
            self.red_perms.append(None)
            self.red_mlp.append(None)

    self.agg_sig = torch.nn.Sigmoid()
    self.input_dims = input_dims
    self.output_dims = output_dims
    self.rwd_scale = 1.

    if self.residual:
        for i in range(len(input_dims)):
            self.output_dims[i] += input_dims[i]

  def forward(self, inputs, action):
    device = inputs[1].device
    # assert len(inputs) == self.max_order + 1

    len_bin = '{0:0' + '{}'.format((self.max_order - 1) * 3 + 2 * 2) + 'b}'

    if type(action) is int:
        act_bin_all = len_bin.format(action)
        reward_layer = -self.rwd_scale * act_bin_all.count('1')
        act = list(map(int, act_bin_all))
        act_list = torch.tensor(act).unsqueeze(0)
    elif type(action) is list:
        act_list = [list(map(int, len_bin.format(act))) for act in action]
        act_list = torch.tensor(act_list)
        reward_layer = 0.
    else:
        raise NotImplementedError

    outputs = []
    for i in range(self.max_order + 1):
        ### collect input f from group i-1, i and i+1.
        f = []
        input_size = inputs[i].shape[:-1] + (self.output_dims[i],)
        if i == 0:
            idx_dir = torch.where(act_list[:, i] == 1)
            if len(idx_dir[0]) > 0:
                out_dir = torch.zeros(input_size).to(device)
                inp = torch.index_select(inputs[i], 0, idx_dir[0])
                out_mlp = self.Op_mlp[i](inp)
                out_dir.index_copy_(0, idx_dir[0], out_mlp)
                f.append(out_dir)
            idx_red = torch.where(act_list[:, i + 1] == 1)
            if len(idx_red[0]) > 0:
                out_red = torch.zeros(input_size).to(device)
                inp = torch.index_select(inputs[i + 1], 0, idx_red[0])
                op_red = self.Op_reducers[i](inp)
                out_mlp = self.red_mlp[i](op_red)
                out_red.index_copy_(0, idx_red[0], out_mlp)
                f.append(out_red)
        elif i == self.max_order:
            idx_exp = torch.where(act_list[:, 3 * (i - 1) + 2] == 1)
            if len(idx_exp[0]) > 0:
                out_exp = torch.zeros(input_size).to(device)
                inp = torch.index_select(inputs[i - 1], 0, idx_exp[0])
                n = inputs[i].size(1) if i == 1 else None
                op_exp = self.Op_expanders[i](inp, n)
                op_perm = self.exp_perms[i](op_exp)
                out_mlp = self.exp_mlp[i](op_perm)
                out_exp.index_copy_(0, idx_exp[0], out_mlp)
                f.append(out_exp)
            idx_dir = torch.where(act_list[:, 3 * (i - 1) + 3] == 1)
            if len(idx_dir[0]) > 0:
                out_dir = torch.zeros(input_size).to(device)
                inp = torch.index_select(inputs[i], 0, idx_dir[0])
                op_perm = self.Op_perms[i](inp)
                out_mlp = self.Op_mlp[i](op_perm)
                out_dir.index_copy_(0, idx_dir[0], out_mlp)
                f.append(out_dir)
        else:
            idx_exp = torch.where(act_list[:, 3 * (i - 1) + 2] == 1)
            if len(idx_exp[0]) > 0:
                out_exp = torch.zeros(input_size).to(device)
                inp = torch.index_select(inputs[i - 1], 0, idx_exp[0])
                n = inputs[i].size(1) if i == 1 else None
                op_exp = self.Op_expanders[i](inp, n)
                op_perm = self.exp_perms[i](op_exp)
                out_mlp = self.exp_mlp[i](op_perm)
                out_exp.index_copy_(0, idx_exp[0], out_mlp)
                f.append(out_exp)
            idx_dir = torch.where(act_list[:, 3 * (i - 1) + 3] == 1)
            if len(idx_dir[0]) > 0:
                out_dir = torch.zeros(input_size).to(device)
                inp = torch.index_select(inputs[i], 0, idx_dir[0])
                op_perm = self.Op_perms[i](inp)
                out_mlp = self.Op_mlp[i](op_perm)
                out_dir.index_copy_(0, idx_dir[0], out_mlp)
                f.append(out_dir)
            idx_red = torch.where(act_list[:, 3 * (i - 1) + 4] == 1)
            if len(idx_red[0]) > 0:
                out_red = torch.zeros(input_size).to(device)
                inp = torch.index_select(inputs[i + 1], 0, idx_red[0])
                op_red = self.Op_reducers[i](inp)
                op_perm = self.red_perms[i](op_red)
                out_mlp = self.red_mlp[i](op_perm)
                out_red.index_copy_(0, idx_red[0], out_mlp)
                f.append(out_red)
        if len(f) > 0:
            output = self.agg_sig(sum(f))
        else:
            output = torch.zeros(input_size).to(device)

        if self.residual and self.input_dims[i] > 0:
            output = torch.cat([inputs[i], output], dim=-1)
        outputs.append(output)
    return outputs, reward_layer


  __hyperparams__ = (
      'breadth',
      'input_dims',
      'output_dims',
      'logic_hidden_dim',
      'exclude_self',
      'residual',
  )

  __hyperparam_defaults__ = {
      'exclude_self': True,
      'residual': False,
  }

  @classmethod
  def make_nlm_parser(cls, parser, defaults, prefix=None):
    for k, v in cls.__hyperparam_defaults__.items():
      defaults.setdefault(k, v)

    if prefix is None:
      prefix = '--'
    else:
      prefix = '--' + str(prefix) + '-'

    parser.add_argument(
        prefix + 'breadth',
        type='int',
        default=defaults['breadth'],
        metavar='N',
        help='breadth of the logic layer')
    parser.add_argument(
        prefix + 'logic-hidden-dim',
        type=int,
        nargs='+',
        default=defaults['logic_hidden_dim'],
        metavar='N',
        help='hidden dim of the logic model')
    parser.add_argument(
        prefix + 'exclude-self',
        type='bool',
        default=defaults['exclude_self'],
        metavar='B',
        help='not allow multiple occurrence of same variable')
    parser.add_argument(
        prefix + 'residual',
        type='bool',
        default=defaults['residual'],
        metavar='B',
        help='use residual connections')

  @classmethod
  def from_args(cls, input_dims, output_dims, args, prefix=None, **kwargs):
    if prefix is None:
      prefix = ''
    else:
      prefix = str(prefix) + '_'

    setattr(args, prefix + 'input_dims', input_dims)
    setattr(args, prefix + 'output_dims', output_dims)
    init_params = {k: getattr(args, prefix + k) for k in cls.__hyperparams__}
    init_params.update(kwargs)

    return cls(**init_params)


class LogicMachine(nn.Module):
  """Neural Logic Machine consists of multiple logic layers."""

  def __init__(
      self,
      depth,
      breadth,
      input_dims,
      output_dims,
      logic_hidden_dim,
      exclude_self=True,
      residual=False,
      io_residual=False,
      recursion=False,
      connections=None,
  ):
    super().__init__()
    self.depth = depth
    self.breadth = breadth
    self.residual = residual
    self.io_residual = io_residual
    self.recursion = recursion
    self.connections = connections

    # print("LogicMachine depth:", depth)
    # print("LogicMachine breadth:", breadth)

    assert not (self.residual and self.io_residual), \
        'Only one type of residual connection is allowed at the same time.'

    self.layers = nn.ModuleList()
    current_dims = input_dims

    for i in range(depth):
      # print("NLM init: depth", i)
      # print("current_dims", current_dims)
      # Not support output_dims as list or list[list] yet.
      layer = LogicLayer(breadth, current_dims, output_dims, logic_hidden_dim,
                         exclude_self, residual)
      current_dims = layer.output_dims
      # print("output_dims", current_dims)
      self.layers.append(layer)

    #####################
    self.output_dims = current_dims


  def forward(self, inputs, depth=None):
    outputs = [None for _ in range(self.breadth + 1)]
    f = inputs
    # print("------ LogicMachine Forward ------")
    # print('f-states:', inputs[1][0, :, :])
    # print('f-relations:', inputs[2][0, :, :, 0])

    # depth: the actual depth used for inference
    if depth is None:
      depth = self.depth
    if not self.recursion:
      depth = min(depth, self.depth)

    layer = None
    last_layer = None
    for i in range(depth):
      # print("------LM Forward ------depth: ", i)
      # To enable recursion, use scroll variables layer/last_layer
      # For weight sharing of period 2, i.e. 0,1,2,1,2,1,2,...
      if self.recursion and i >= 3:
        assert not self.residual
        layer, last_layer = last_layer, layer
      else:
        last_layer = layer
        layer = self.layers[i]

      f = layer(f)

    if not self.io_residual:
      outputs = f
    return outputs

  __hyperparams__ = (
      'depth',
      'breadth',
      'input_dims',
      'output_dims',
      'logic_hidden_dim',
      'exclude_self',
      'io_residual',
      'residual',
      'recursion',
  )

  __hyperparam_defaults__ = {
      'exclude_self': True,
      'io_residual': False,
      'residual': False,
      'recursion': False,
  }

  @classmethod
  def make_nlm_parser(cls, parser, defaults, prefix=None):
    for k, v in cls.__hyperparam_defaults__.items():
      defaults.setdefault(k, v)

    if prefix is None:
      prefix = '--'
    else:
      prefix = '--' + str(prefix) + '-'

    parser.add_argument(
        prefix + 'depth',
        type=int,
        default=defaults['depth'],
        metavar='N',
        help='depth of the logic machine')
    parser.add_argument(
        prefix + 'breadth',
        type=int,
        default=defaults['breadth'],
        metavar='N',
        help='breadth of the logic machine')
    parser.add_argument(
        prefix + 'logic-hidden-dim',
        type=int,
        nargs='+',
        default=defaults['logic_hidden_dim'],
        metavar='N',
        help='hidden dim of the logic model')
    parser.add_argument(
        prefix + 'exclude-self',
        type='bool',
        default=defaults['exclude_self'],
        metavar='B',
        help='not allow multiple occurrence of same variable')
    parser.add_argument(
        prefix + 'io-residual',
        type='bool',
        default=defaults['io_residual'],
        metavar='B',
        help='use input/output-only residual connections')
    parser.add_argument(
        prefix + 'residual',
        type='bool',
        default=defaults['residual'],
        metavar='B',
        help='use residual connections')
    parser.add_argument(
        prefix + 'recursion',
        type='bool',
        default=defaults['recursion'],
        metavar='B',
        help='use recursion weight sharing')

  @classmethod
  def from_args(cls, input_dims, output_dims, prefix=None, **kwargs):
    # if prefix is None:
    #   prefix = ''
    # else:
    #   prefix = str(prefix) + '_'
    #
    # setattr(args, prefix + 'input_dims', input_dims)
    # setattr(args, prefix + 'output_dims', output_dims)
    # init_params = {k: getattr(args, prefix + k) for k in cls.__hyperparams__}
    # init_params.update(kwargs)
    #
    # ###################################
    # for k, v in init_params.items():
    #     print("LogicMachine key: {}, val:{}".format(k, v))
    # ###################################
    init_params = {}
    init_params['depth'] = 2 #4
    init_params['breadth'] = 2 #3
    init_params['input_dims'] = [0, 4, 1] #[0, 4, 1, 0]
    init_params['output_dims'] = 8 #8
    init_params['logic_hidden_dim'] = []
    init_params['exclude_self'] = True
    init_params['io_residual'] = False
    init_params['residual'] = False
    init_params['recursion'] = False

    return cls(**init_params)
