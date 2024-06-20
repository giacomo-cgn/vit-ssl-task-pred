from torchvision import models
from torch import nn
from typing import Tuple

from .vit import ViT
from .task_pred_vit import TaskPredViT

def get_encoder(encoder_name, image_size, vit_avg_pooling, kwargs, save_pth=None) -> Tuple[nn.Module, int]:
    """Returns an initialized encoder without the last clf layer and the encoder feature dimensions."""

    if encoder_name == 'vit_tiny' and kwargs["strategy"] != 'replay_task_pred':
        if image_size == 32:
            encoder = ViT(image_size=image_size, patch_size=2, return_avg_pooling=vit_avg_pooling, save_pth=save_pth)
        elif image_size == 64:
            encoder = ViT(image_size=image_size, patch_size=4, return_avg_pooling=vit_avg_pooling, save_pth=save_pth)
        elif image_size == 224:
            encoder = ViT(image_size=image_size, patch_size=16, return_avg_pooling=vit_avg_pooling, save_pth=save_pth)
        elif image_size == 256:
            encoder = ViT(image_size=image_size, patch_size=16, return_avg_pooling=vit_avg_pooling, save_pth=save_pth)
        else:
            raise Exception(f'Invalid image size for ViT backbone: {image_size}')
        dim_encoder_features = encoder.emb_dim

        # Wrap to return only 1 feature tensor
        encoder = encoder.return_features_wrapper()

    elif encoder_name == 'vit_tiny' and kwargs["strategy"] == 'replay_task_pred':
        if image_size == 32:
            encoder = TaskPredViT(image_size=image_size, patch_size=2, return_avg_pooling=vit_avg_pooling, save_pth=save_pth,
                                  num_tasks=kwargs["num_exps"], detach_task_head=kwargs["detach_task_head"], task_criterion=kwargs["task_criterion"])
        elif image_size == 64:
            encoder = TaskPredViT(image_size=image_size, patch_size=4, return_avg_pooling=vit_avg_pooling, save_pth=save_pth,
                                  num_tasks=kwargs["num_exps"], detach_task_head=kwargs["detach_task_head"], task_criterion=kwargs["task_criterion"])
        elif image_size == 224:
            encoder = TaskPredViT(image_size=image_size, patch_size=16, return_avg_pooling=vit_avg_pooling, save_pth=save_pth,
                                  num_tasks=kwargs["num_exps"], detach_task_head=kwargs["detach_task_head"], task_criterion=kwargs["task_criterion"])
        elif image_size == 256:
            encoder = TaskPredViT(image_size=image_size, patch_size=16, return_avg_pooling=vit_avg_pooling, save_pth=save_pth,
                                  num_tasks=kwargs["num_exps"], detach_task_head=kwargs["detach_task_head"], task_criterion=kwargs["task_criterion"])
        else:
            raise Exception(f'Invalid image size for ViT backbone: {image_size}')
        dim_encoder_features = encoder.emb_dim

        # Wrap to return only 1 feature tensor
        encoder = encoder.return_features_wrapper()   
    else:
        raise Exception(f'Invalid encoder: {encoder_name}')
    
    
    # Print number of parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    total_params_in_millions = total_params / 1e6
    print(f'NUM PARAMS for {encoder_name}: {total_params_in_millions:.1f}M')
    
    return encoder, dim_encoder_features