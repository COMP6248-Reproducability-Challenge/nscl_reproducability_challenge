import torch
import collections


def normalize(x):
    return x / x.norm(2, dim=-1, keepdim=True)


def __last(arr, x):
    return arr.narrow(-1, x, 1).squeeze(-1)


def generate_roi_pool_bins(box, bin_size, c2l=0):
    linspace = torch.linspace(0, 1, bin_size + 1, dtype=box.dtype).to(device=box.device)
    for i in range(box.dim() - 1):
        linspace.unsqueeze_(0)
    x_space = linspace * (__last(box, 2) - __last(box, 0) + c2l).unsqueeze(-1) + __last(box, 0).unsqueeze(-1)
    y_space = linspace * (__last(box, 3) - __last(box, 1) + c2l).unsqueeze(-1) + __last(box, 1).unsqueeze(-1)
    x1, x2 = x_space[:, :-1], x_space[:, 1:] - c2l
    y1, y2 = y_space[:, :-1], y_space[:, 1:] - c2l

    y1, x1 = meshgrid(y1, x1, dim=-1)
    y2, x2 = meshgrid(y2, x2, dim=-1)

    # shape: nr_boxes, bin_size^2, 4
    bins = torch.stack([x1, y1, x2, y2], dim=-1).view(box.size(0), -1, 4)

    return bins.float()


def concat_shape(*shapes):
    output = []
    for s in shapes:
        if isinstance(s, collections.abc.Sequence):
            output.extend(s)
        else:
            output.append(int(s))
    return tuple(output)


def broadcast(tensor, dim, size):
    if dim < 0:
        dim += tensor.dim()
    assert tensor.size(dim) == 1
    shape = tensor.size()
    return tensor.expand(concat_shape(shape[:dim], size, shape[dim + 1:]))


def meshgrid(input1, input2=None, dim=1):
    """Perform np.meshgrid along given axis. It will generate a new dimension after dim."""
    if input2 is None:
        input2 = input1
    if dim < 0:
        dim += input1.dim()
    n, m = input1.size(dim), input2.size(dim)
    x = broadcast(input1.unsqueeze(dim + 1), dim + 1, m)
    y = broadcast(input2.unsqueeze(dim + 0), dim + 0, n)
    return x, y


def box_size(box, c2l=0):
    return (__last(box, 2) - __last(box, 0) + c2l) * (__last(box, 3) - __last(box, 1) + c2l)


def box_intersection(box1, box2, ratio=False, c2l=0):
    xmin, ymin = [torch.max(__last(box1, i), __last(box2, i)) for i in range(2)]
    xmax, ymax = [torch.min(__last(box1, i), __last(box2, i)) for i in range(2, 4)]
    iw = torch.max(xmax - xmin + c2l, torch.zeros_like(xmax))
    ih = torch.max(ymax - ymin + c2l, torch.zeros_like(ymax))
    inter = iw * ih
    if ratio:
        return inter / box_size(box2)
    return inter


def generate_intersection_map(box1, box2, bin_size, c2l=0):
    # box: nr_boxes, 4
    # bins: nr_boxes, bin_size^2, 4
    bins = generate_roi_pool_bins(box2, bin_size, c2l)
    box1 = box1.unsqueeze(1).expand_as(bins)
    return box_intersection(box1, bins, ratio=True, c2l=c2l).view(box1.size(0), 1, bin_size, bin_size).float()
