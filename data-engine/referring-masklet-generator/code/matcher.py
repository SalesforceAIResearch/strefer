import torch
from torch import nn
from torchvision.ops.boxes import box_area

import numpy as np
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """
    Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


class HungarianMatcher(nn.Module):
    
    def __init__(self):
        """Initializes the matcher. Only GIoU is used for cost calculation."""
        super().__init__()

    @torch.no_grad()
    def forward(self, pred_boxes_xyxy, tgt_boxes_xyxy):
        """
        Performs the matching based solely on the GIoU cost.

        Params:
            pred_boxes_xyxy: Tensor of shape [num_pred_boxes, 4] in xyxy format
            tgt_boxes_xyxy: Tensor of shape [num_target_boxes, 4] in xyxy format

        Returns:
            Tuple of (index_i, index_j) where:
                - index_i are the indices of selected predictions
                - index_j are the indices of corresponding selected targets
        """
        if not isinstance(pred_boxes_xyxy, torch.Tensor):
            pred_boxes_xyxy = torch.Tensor(pred_boxes_xyxy)
        if not isinstance(tgt_boxes_xyxy, torch.Tensor):
            tgt_boxes_xyxy = torch.Tensor(tgt_boxes_xyxy)
        
        # Compute the negative GIoU as cost matrix
        cost_giou = -generalized_box_iou(pred_boxes_xyxy, tgt_boxes_xyxy)  # [num_pred_boxes, num_target_boxes]

        # Solve the linear sum assignment problem
        pred_boxes_indices, tgt_boxes_indices = linear_sum_assignment(cost_giou.cpu())

        matching = dict()
        for i in range(len(pred_boxes_indices)):
            matching[int(pred_boxes_indices[i])] = int(tgt_boxes_indices[i])
        
        return pred_boxes_indices, tgt_boxes_indices, matching


if __name__ == "__main__":
    matcher = HungarianMatcher()
    box1 = torch.Tensor(np.array([[159, 170, 273, 351],
           [398,   0, 453, 176],
           [248, 185, 308, 313],
           [514, 278, 639, 359]]))
    box2 = torch.Tensor(np.array([[158, 170, 273, 352],
           [248, 185, 308, 313],
           [398,   0, 453, 176],
           [514, 278, 639, 359]]))
    box1_indices, box2_indices, matching = matcher(box1, box2)
    print("box1_indices:", box1_indices)
    print("box2_indices:", box2_indices)
    # box1_indices: array([0, 1, 2, 3])
    # box2_indices: array([0, 2, 1, 3])

    print(matching)
    # {0: 0, 1: 2, 2: 1, 3: 3}
    

    
