import os
import torch

from ..ssl_models import AbstractSSLModel
from .abstract_strategy import AbstractStrategy

class ReplayTaskPred(AbstractStrategy):

    def __init__(self,
                 ssl_model: AbstractSSLModel = None,
                 buffer = None,
                 device = 'cpu',
                 save_pth: str  = None,
                 replay_mb_size: int = 32,
                 omega_task: float = 0.1,
                 detach_task_head: bool = False,
                 task_criterion: str = 'cross_entropy',
                 num_tasks: int = 20,
                 emb_dim: int = 192
                ):
            
        self.ssl_model = ssl_model
        self.buffer = buffer
        self.device = device
        self.save_pth = save_pth
        self.replay_mb_size = replay_mb_size
        self.omega_task = omega_task
        self.detach_task_head = detach_task_head
        self.num_tasks = num_tasks

        self.strategy_name = 'replay_task_pred'

        self.task_pred_head = torch.nn.Linear(emb_dim, num_tasks).to(self.device)

        if task_criterion == 'cross_entropy':
            self.task_criterion = torch.nn.CrossEntropyLoss()
        else:
            raise Exception(f"Invalid alignment criterion: {self.task_criterion}")
        
        self.epoch_task_preds = []
        self.epoch_task_labels = []


        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- STRATEGY CONFIG ----\n')
                f.write(f'STRATEGY: {self.strategy_name}\n')
                f.write(f'Omega task pred: {self.omega_task}\n')
                f.write(f'Detach task head: {self.detach_task_head}\n')
                f.write(f'Task criterion: {self.task_criterion}\n')

         # Write tr task prediction file column names
        with open(os.path.join(self.save_pth, 'tr_task_pred_acc.csv'), 'a') as f:
            header = 'exp_idx,epoch,total_acc'
            for t in range(num_tasks):
                header += f',t{t}_acc'
            f.write(header + '\n')

    def get_params(self):
        """Get trainable parameters of the strategy.
        
        Returns:
            alignment_projector (nn.Module): The alignment projector module.
        """
        return list(self.task_pred_head.parameters())

    def before_forward(self, stream_mbatch, stream_task_labels):
        """Sample from buffer and concat with stream batch."""

        self.stream_mbatch = stream_mbatch
        self.stream_task_labels = stream_task_labels

        if len(self.buffer.buffer) > self.replay_mb_size:
            self.use_replay = True
            # Sample from buffer and concat
            replay_batch, _, replay_task_labels, replay_indices = self.buffer.sample(self.replay_mb_size)
            replay_batch, replay_task_labels = replay_batch.to(self.device), replay_task_labels.to(self.device)
            
            combined_batch = torch.cat((replay_batch, stream_mbatch), dim=0)
            combined_task_labels = torch.cat((replay_task_labels, stream_task_labels), dim=0)
            # Save buffer indices of replayed samples
            self.replay_indices = replay_indices
        else:
            self.use_replay = False
            # Do not sample buffer if not enough elements in it
            combined_batch = stream_mbatch
            combined_task_labels = stream_task_labels

        return combined_batch, combined_task_labels
    
    def after_forward(self, x_views_list, loss, z_list, e_list, all_features_list, task_labels):
        """ Only update buffer features for replayed samples"""
        self.task_labels = task_labels
        
        task_loss_list = []
        for all_features in all_features_list:
            # Predict the task
            if self.detach_task_head:
                task_head_logits = self.task_pred_head(all_features[:,-1].detach())
            else:
                task_head_logits = self.task_pred_head(all_features[:, -1])
            task_loss_list.append(self.task_criterion(task_head_logits, task_labels).mean())
            self.task_pred = torch.argmax(task_head_logits, dim=1)
        
        task_loss = sum(task_loss_list)/len(task_loss_list)
        loss = loss + self.omega_task * task_loss 

        self.z_list = z_list
        if self.use_replay:
            # Take only the features from the replay batch (for each view minibatch in z_list,
            #  take only the first replay_mb_size elements)
            z_list_replay = [z[:self.replay_mb_size] for z in z_list]
            # Update replayed samples with avg of last extracted features
            avg_replayed_z = sum(z_list_replay)/len(z_list_replay)
            self.buffer.update_features(avg_replayed_z.detach(), self.replay_indices)
        
        return loss
    

    def after_mb_passes(self):
        """Update buffer with new samples after all mb_passes with streaming mbatch."""

        # Get features only of the streaming mbatch and their avg across views
        z_list_stream = [z[-len(self.stream_mbatch):] for z in self.z_list]
        z_stream_avg = sum(z_list_stream)/len(z_list_stream)

        # Update buffer with new stream samples and avg features
        self.buffer.add(self.stream_mbatch.detach(), z_stream_avg.detach(), self.stream_task_labels)

        # Save latest task prediction and labels
        self.epoch_task_preds += self.task_pred.detach().cpu().tolist()
        self.epoch_task_labels += self.task_labels.detach().cpu().tolist()

    def after_epoch(self, exp_idx, epoch):
        print ('---- Calculating training task prediction accuracy ----')
        # given self.epoch_task_preds and self.epoch_task_labels calculate total accuracy and task-specific accuracy
        total_acc = 100 * sum([1 for pred, label in zip(self.epoch_task_preds, self.epoch_task_labels) if pred == label]) / len(self.epoch_task_preds)
        task_acc = {} # {label: (count, count_correct_preds)}
        for task in range(self.num_tasks):
            task_acc[task] = [0, 0]

        for pred, label in zip(self.epoch_task_preds, self.epoch_task_labels):
            task_acc[label][0] += 1
            if pred == label:
                task_acc[label][1] += 1
        for k in task_acc:
            # multiply by 100 and limit 2 decimal
            if task_acc[k][0] == 0:
                task_acc[k] = 0
            else:
                task_acc[k] = 100 * task_acc[k][1] / task_acc[k][0]

        with open(os.path.join(self.save_pth, 'tr_task_pred_acc.csv'), 'a') as f:
            f.write(f'{exp_idx},{epoch},{total_acc:.2f}')
            for t in range(self.num_tasks):
                f.write(f',{task_acc[t]:.2f}')
            f.write('\n')

        self.task_pred, self.task_labels = [], []

