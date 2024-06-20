import torch

class AbstractStrategy():
    def __init__(self, ssl_model):
        self.strategy_name = "AbstractStrategy"

    def before_experience(self):
        pass

    def before_mb_passes(self, stream_mbatch, stream_task_labels):
        return stream_mbatch, stream_task_labels

    def before_forward(self, batch, batch_task_labels):
        return batch, batch_task_labels

    def before_forward(self, stream_mbatch) -> torch.Tensor:
        return stream_mbatch

    def after_transforms(self, x_views_list) -> list[torch.Tensor]:
        return x_views_list

    def after_forward(self, x_views_list, loss, z_list, e_list, all_features_list, task_labels):
        return loss
    
    def after_backward(self):
        pass

    def after_mb_passes(self):
        pass

    def after_epoch(self, exp_idx, epoch):
        pass

    def get_params(self) -> list:
        return []

    def get_name(self):
        return self.strategy_name