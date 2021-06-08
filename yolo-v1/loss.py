import torch
import torch.nn as nn
from utils import intersection_over_union


# defines class to calculate the loss of the model
class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()

        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.4
        self.lambda_coord = 5.0

    def forward(self, predictions, target):
        n_classes = self.C
        predictions = predictions.reshape(-1,
                                          self.S, self.S, self.C + self.B*5)

        iou_b1 = intersection_over_union(
            predictions[..., (n_classes+1):(n_classes+5)], target[..., (n_classes+1):(n_classes+5)])

        iou_b2 = intersection_over_union(
            predictions[..., (n_classes+6):(n_classes+10)], target[..., (n_classes+1):(n_classes+5)])

        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # [max iou, argmax iou]
        iou_maxes, bestbox = torch.max(ious, dim=0)

        # identity function Iobj_i
        exists_box = target[..., n_classes].unsqueeze(3)

        #---------BOX COORDINATE LOSS----------#

        box_predictions = exists_box * (
            bestbox * predictions[..., (n_classes+6):(n_classes+10)] +
            (1-bestbox) * predictions[..., (n_classes+1):(n_classes+5)]
        )

        box_targets = exists_box * target[..., (n_classes+1):(n_classes+5)]

        # sqrt of width and heights
        box_predictions[..., 2:4] = torch.sign(
            box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4]+1e-6))

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        #-------------OBJECT LOSS--------------#
        pred_box = (
            bestbox * predictions[..., (n_classes+5):(n_classes+6)] +
            (1-bestbox) * predictions[..., n_classes:(n_classes+1)]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., n_classes:(n_classes+1)])
        )

        #-----------NO OBJECT LOSS-------------#
        no_object_loss = self.mse(
            torch.flatten((1-exists_box) *
                          predictions[..., n_classes:(n_classes+1)], start_dim=1),
            torch.flatten((1-exists_box) *
                          target[..., n_classes:(n_classes+1)], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1-exists_box) *
                          predictions[..., (n_classes+5):(n_classes+6)], start_dim=1),
            torch.flatten((1-exists_box) *
                          target[..., (n_classes):(n_classes+1)], start_dim=1)
        )

        #----------------CLASS-----------------#
        class_loss = self.mse(
            torch.flatten(
                exists_box * predictions[..., :n_classes], end_dim=2),
            torch.flatten(exists_box * target[..., :n_classes], end_dim=2)
        )

        loss = (
            self.lambda_coord * box_loss +
            object_loss +
            self.lambda_noobj * no_object_loss +
            class_loss
        )

        return loss
