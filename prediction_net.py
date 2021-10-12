import torch
import torch.nn as nn
import torch.nn.functional as F

from OpsAsAct_net.nn.neural_logic.modules.dimension import Expander, Reducer
from OpsAsAct_net.nn.neural_logic.modules.neural_logic import LogitsInference, LogicInference

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

CONST = 1e+5

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
    # assert breadth > 0, 'Does not support breadth <= 0.'

    self.max_order = breadth
    self.residual = residual

    input_dims = _get_tuple_n(input_dims, self.max_order + 1, int)
    output_dims = _get_tuple_n(output_dims, self.max_order + 1, int)

    #### logic: MLP; dim_perms: Permutation; dim_expanders: expand; dim_reducers: reduce
    self.logic, self.dim_expanders, self.dim_reducers = [nn.ModuleList() for _ in range(3)]

    for i in range(self.max_order + 1):
      # collect current_dim from group i-1, i and i+1.
      current_dim = input_dims[i]
      if i > 0:
        expander = Expander(i - 1)
        self.dim_expanders.append(expander)
        current_dim += expander.get_output_dim(input_dims[i - 1])
      else:
        self.dim_expanders.append(None)

      if i + 1 < self.max_order + 1:
        reducer = Reducer(i + 1, exclude_self)
        self.dim_reducers.append(reducer)
        current_dim += reducer.get_output_dim(input_dims[i + 1])
      else:
        self.dim_reducers.append(None)

      if current_dim == 0:
        self.logic.append(None)
        output_dims[i] = 0
      else:
        self.logic.append(
            LogitsInference(current_dim, output_dims[i], logic_hidden_dim))

    self.input_dims = input_dims
    self.output_dims = output_dims

  def forward(self, inputs):
      outputs = []
      mask_idx = []
      for i in range(self.max_order + 1):
          # collect input f from group i-1, i and i+1.
          f = []
          if i > 0 and self.input_dims[i - 1] > 0:
              n = inputs[i].size(1) if i == 1 else None
              f.append(self.dim_expanders[i](inputs[i - 1], n))

              if len(inputs[i - 1].nonzero()) == 0:
                  mask_idx.append(-CONST)
              else:
                  mask_idx.append(0.)

          if i < len(inputs) and self.input_dims[i] > 0:
              f.append(inputs[i])

              if len(inputs[i].nonzero()) == 0:
                  mask_idx.append(-CONST)
              else:
                  mask_idx.append(0.)

          if i + 1 < len(inputs) and self.input_dims[i + 1] > 0:
              f.append(self.dim_reducers[i](inputs[i + 1]))

              if len(inputs[i + 1].nonzero()) == 0:
                  mask_idx.append(-CONST)
              else:
                  mask_idx.append(0.)

          if len(f) == 0:
              output = None
          else:
              f = torch.cat(f, dim=-1)
              output = self.logic[i](f)

          outputs.append(output)
      return outputs, mask_idx


class Reasoner(nn.Module):
    """one layer of Reasoner"""

    def __init__(self, breadth, input_dims, logic_hidden_dim, mask, residual=False, exclude_self=True):
        super().__init__()

        self.breadth = breadth
        LogMac_input_dims = [input_dims for _ in range(breadth + 1)]
        LogMac_output_dims = [2 if r == 0 or r == breadth else 3 for r in range(breadth + 1)]
        LogMac_logic_hidden_dim = logic_hidden_dim ##[]
        LogMac_exclude_self = exclude_self
        LogMac_residual = residual  # residual connection

        LM_current_dims = LogMac_input_dims.copy()
        self.Reason_layers_pi = LogicLayer(self.breadth, LM_current_dims, LogMac_output_dims, LogMac_logic_hidden_dim,
                                  LogMac_exclude_self, LogMac_residual)

        self.act_mask = mask

    def forward(self, inp):
        out, mask = self.Reason_layers_pi(inp)
        output = []
        batchsize = inp[0].size(0)
        ## max pooling
        for r in range(self.breadth + 1):
            if r == 0:
                output.append(out[r])
            else:
                idx_tup = tuple(range(1, r + 1))
                for dim in idx_tup:
                    out[r] = torch.max(out[r], dim=dim, keepdim=True)[0]
                if batchsize > 1:
                    output.append(out[r].squeeze())
                else:
                    output.append(out[r].squeeze().unsqueeze(0))

        if self.act_mask:
            output_tensor = torch.cat(output, dim=1)
            mask_tensor = torch.tensor(mask).unsqueeze(0).repeat(batchsize, 1)
            return output_tensor.add(mask_tensor)
        else:
            return torch.cat(output, dim=1)


class Planner(nn.Module):
    """
    main Planner module or policy network
    """
    def __init__(self, breadth, input_dims, logic_hidden_dim, Mat):
        super().__init__()
        self.act_mask = False
        self.reasoner_pi = Reasoner(breadth, input_dims, logic_hidden_dim, self.act_mask)
        self.matrix = Mat
        self.ops = nn.LogSigmoid()

    def forward(self, inputs):
        pol_logits_1 = self.reasoner_pi(inputs)
        pol_logits_0 = - pol_logits_1
        pol_prob = torch.cat([self.ops(pol_logits_0), self.ops(pol_logits_1)], dim=1)
        log_prob = torch.matmul(pol_prob, self.matrix)
        return log_prob

########################################################################################

class ReasonLayer(nn.Module):
  """
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
    # assert breadth > 0, 'Does not support breadth <= 0.'

    self.max_order = breadth
    self.residual = residual

    input_dims = _get_tuple_n(input_dims, self.max_order + 1, int)
    output_dims = _get_tuple_n(output_dims, self.max_order + 1, int)

    #### logic: MLP; dim_perms: Permutation; dim_expanders: expand; dim_reducers: reduce
    self.logic, self.dim_expanders, self.dim_reducers = [nn.ModuleList() for _ in range(3)]

    for i in range(self.max_order + 1):
      # collect current_dim from group i-1, i and i+1.
      current_dim = input_dims[i]
      if i > 0:
        expander = Expander(i - 1)
        self.dim_expanders.append(expander)
        current_dim += expander.get_output_dim(input_dims[i - 1])
      else:
        self.dim_expanders.append(None)

      if i + 1 < self.max_order + 1:
        reducer = Reducer(i + 1, exclude_self)
        self.dim_reducers.append(reducer)
        current_dim += reducer.get_output_dim(input_dims[i + 1])
      else:
        self.dim_reducers.append(None)

      if current_dim == 0:
        self.logic.append(None)
        output_dims[i] = 0
      else:
        self.logic.append(
            LogitsInference(current_dim, output_dims[i], logic_hidden_dim))

    self.input_dims = input_dims
    self.output_dims = output_dims

  def forward(self, inputs):
      outputs = []
      for i in range(self.max_order + 1):
          # collect input f from group i-1, i and i+1.
          f = []
          if i > 0 and self.input_dims[i - 1] > 0:
              n = inputs[i].size(1) if i == 1 else None
              f.append(self.dim_expanders[i](inputs[i - 1], n))

          if i < len(inputs) and self.input_dims[i] > 0:
              f.append(inputs[i])

          if i + 1 < len(inputs) and self.input_dims[i + 1] > 0:
              f.append(self.dim_reducers[i](inputs[i + 1]))

          if len(f) == 0:
              output = None
          else:
              f = torch.cat(f, dim=-1)
              output = self.logic[i](f)

          outputs.append(output)
      return outputs

class Reasoner_Val(nn.Module):
    """
    Value Network in MCTS
    """

    def __init__(self, breadth, input_dims, logic_hidden_dim, output_size, residual=False, exclude_self=True):
        super().__init__()

        self.breadth = breadth
        LogMac_input_dims = [input_dims for _ in range(breadth + 1)]
        LogMac_output_dims = [2 if r == 0 or r == breadth else 3 for r in range(breadth + 1)]
        LogMac_logic_hidden_dim = logic_hidden_dim
        LogMac_exclude_self = exclude_self
        LogMac_residual = residual  # residual connection

        LM_current_dims = LogMac_input_dims.copy()
        self.Reason_layers_val = ReasonLayer(self.breadth, LM_current_dims, LogMac_output_dims, LogMac_logic_hidden_dim,
                                  LogMac_exclude_self, LogMac_residual)
        inp_dim = sum(LogMac_output_dims)
        self.ops = nn.Sigmoid()
        self.val_pred = LogitsInference(inp_dim, output_size, inp_dim*2)

    def forward(self, inp):
        out = self.Reason_layers_val(inp)
        output = []
        batchsize = inp[0].size(0)
        for r in range(self.breadth + 1):
            if r == 0:
                output.append(out[r])
            else:
                idx_tup = tuple(range(1, r+1))
                for dim in idx_tup:
                    out[r] = torch.max(out[r], dim=dim, keepdim=True)[0]
                if batchsize > 1:
                    output.append(out[r].squeeze())
                else:
                    output.append(out[r].squeeze().unsqueeze(0))
        ops_indicator = self.ops(torch.cat(output, dim=1))
        value = self.val_pred(ops_indicator)
        return value
