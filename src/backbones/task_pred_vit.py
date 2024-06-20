import numpy as np
import torch
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from einops import repeat, rearrange

from .vit import PatchShuffle, random_indexes, take_indexes

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


class TaskPredViT(torch.nn.Module):
    # Default ViT encoder is ViT-Tiny
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 return_avg_pooling=False,
                 save_pth: str = None,
                 num_tasks: int = 20,
                 detach_task_head: bool = False,
                 task_criterion: str = 'cross_entropy',
                 ) -> None:
        super(TaskPredViT, self).__init__()

        self.emb_dim = emb_dim
        self.return_average_pooling = return_avg_pooling
        self.save_pth = save_pth
        self.num_tasks = num_tasks
        self.detach_task_head = detach_task_head

        # if task_criterion == 'cross_entropy':
        #     self.task_criterion = torch.nn.CrossEntropyLoss()
        # else:
        #     raise Exception(f"Invalid alignment criterion: {self.task_criterion}")


        self.backbone_name = 'Task_prediction_ViT'

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.task_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(ratio=0)

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        # self.task_pred_head = torch.nn.Linear(emb_dim, num_tasks)

        self.init_weight()

        if self.save_pth is not None:
            # Save backbone configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- BACKBONE MODEL CONFIG ----\n')
                f.write(f'BACKBONE: {self.backbone_name}\n')
                f.write(f'image_size: {image_size}\n')
                f.write(f'patch_size: {patch_size}\n')
                f.write(f'emb_dim: {emb_dim}\n')
                f.write(f'num_layer: {num_layer}\n')
                f.write(f'num_head: {num_head}\n')
                f.write(f'ViT Average Pooling: {self.return_average_pooling}\n')
                f.write(f'Detach task head: {self.detach_task_head}\n')
                f.write(f'Task criterion: {task_criterion}\n')


    def init_mask_ratio(self, mask_ratio=0.75):
        self.shuffle = PatchShuffle(mask_ratio)

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.task_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches, self.task_token.expand(-1, patches.shape[1], -1)], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes
    
    def return_features_wrapper(self):
        return TaskViTFeaturesWrapper(self, self.return_average_pooling)
    

class TaskViTFeaturesWrapper(torch.nn.Module):
    def __init__(self, encoder:TaskPredViT, return_avg_pooling=False):
        super().__init__()
        self.encoder = encoder
        self.return_avg_pooling = return_avg_pooling

    def forward(self, x):
        features, _ = self.encoder(x)
        features = rearrange(features, 't b c -> b t c')
        # Return avg pooling of transformer token outputs, otherwise only return clf token
        if self.return_avg_pooling:
            return_features = features.mean(dim=1)
        else:
            return_features = features[:,0]

        
        # # Predict the task
        # if self.encoder.detach_task_head:
        #     task_head_logits = self.encoder.task_pred_head(features[:,-1].detach())
        # else:
        #     task_head_logits = self.encoder.task_pred_head(features[:, -1])

        # task_pred_loss = self.encoder.task_criterion(task_head_logits, task_labels).mean()
        # task_pred = torch.argmax(task_head_logits, dim=1)

        return return_features, features

