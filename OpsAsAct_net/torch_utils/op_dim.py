import collections
import torch

def flatten(tensor):
    return tensor.view(-1)


def flatten2(tensor):
    return tensor.view(tensor.size(0), -1)


def concat_shape(*shapes):
    output = []
    for s in shapes:
        if isinstance(s, collections.Sequence):
            output.extend(s)
        else:
            output.append(int(s))
    return tuple(output)


def broadcast(tensor, dim, size):
    if dim < 0:
        dim += tensor.dim()
    assert tensor.size(dim) == 1
    shape = tensor.size()
    return tensor.expand(concat_shape(shape[:dim], size, shape[dim+1:]))


def add_dim(tensor, dim, size):
    return broadcast(tensor.unsqueeze(dim), dim, size)


def add_dim_as_except(tensor, target, *excepts):
    assert len(excepts) == tensor.dim()
    tensor = tensor.clone()
    excepts = [e + target.dim() if e < 0 else e for e in excepts]
    for i in range(target.dim()):
        if i not in excepts:
            tensor.unsqueeze_(i)
    return tensor


def move_dim(tensor, dim, dest):
    dims = list(range(tensor.dim()))
    dims.pop(dim)
    dims.insert(dest, dim)
    return tensor.permute(dims)




