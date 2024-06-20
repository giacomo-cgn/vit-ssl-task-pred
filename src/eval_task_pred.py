import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from .benchmark import Benchmark


def exec_eval_task_pred(kwargs: dict,
                        model: torch.nn,
                        task_pred_head: torch.nn,
                        benchmark: Benchmark,
                        exp_idx: int,
                        save_pth: str,
                        epoch: int = None,
                        device: str = 'cpu',
                            ):
    # Generate joint tasks datasets
    task_pred_dataset_test = ConcatDataset([benchmark.test_stream[i] for i in range(kwargs["num_exps"])])
    if kwargs['probing_val_ratio'] > 0:
        task_pred_dataset_val = ConcatDataset([benchmark.test_stream[i] for i in range(kwargs["num_exps"])])
    else:
        task_pred_dataset_val = None

    print(f'-- Evaluating Task Prediction accuracy--')
    eval_task_pred(model, task_pred_head, exp_idx, save_pth, task_pred_dataset_test, task_pred_dataset_val,
                   epoch=epoch, device=device, num_tasks=kwargs["num_exps"], mb_size=kwargs["eval_mb_size"])


def eval_task_pred(model: torch.nn,
                   task_pred_head: torch.nn,
                   exp_idx: int,
                   save_pth: str,
                   test_dataset: Dataset,
                   val_dataset: Dataset = None,
                   epoch = None,
                   device: str = 'cpu',
                   num_tasks: int = 20,
                   mb_size: int = 256):
    
    # Write separate task prediction accuracy header
    save_file_separate = os.path.join(save_pth, 'eval_task_pred_separate.csv')
    with open(save_file_separate, 'a') as f:
        if not os.path.exists(save_file_separate) or os.path.getsize(save_file_separate) == 0:
            # Write header if file has been created
            if epoch is None:
                header = 'exp_idx'
            else:
                header = 'exp_idx,epoch'
            for t in range(num_tasks):
                header += f',t{t}_acc'
            f.write(header + '\n')
    # Write joint task prediction accuracy header
    save_file_joint = os.path.join(save_pth, 'eval_task_pred_joint.csv')
    with open(save_file_joint, 'a') as f:
        if not os.path.exists(save_file_joint) or os.path.getsize(save_file_joint) == 0:
            # Write header if file has been created
            if epoch is None:
                header = 'exp_idx,val_acc,test_acc\n'
            else:
                header = 'exp_idx,epoch,val_acc,test_acc\n'
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=mb_size, shuffle=False)
    if val_dataset is not None:
        val_loader = DataLoader(dataset=val_dataset, batch_size=mb_size, shuffle=False)

    with torch.no_grad():
        model.eval()
        task_pred_head.eval()

        if val_dataset is not None:
            # Get task predictions for val dataloader
            val_task_pred_list = []
            val_task_labels_list = []
            for inputs, _, task_labels in val_loader:
                inputs, task_labels = inputs.to(device), task_labels.to(device)
                _, all_features = model(inputs)
                task_head_logits = task_pred_head(all_features[:, -1])
                task_pred = torch.argmax(task_head_logits, dim=1)

                val_task_pred_list.append(task_pred.detach().cpu())
                val_task_labels_list.append(task_labels.detach().cpu())
  
            val_task_pred_list = torch.cat(val_task_pred_list, dim=0).tolist()
            val_task_labels_list = torch.cat(val_task_labels_list, dim=0).tolist()

            total_acc_val, task_acc_val = get_accuracies(val_task_pred_list, val_task_labels_list, num_tasks)

        # Get task predictions for test dataloader
        test_task_pred_list = []
        test_task_labels_list = []
        for inputs, _, task_labels in test_loader:
            inputs, task_labels = inputs.to(device), task_labels.to(device)
            _, all_features = model(inputs)
            task_head_logits = task_pred_head(all_features[:, -1])
            task_pred = torch.argmax(task_head_logits, dim=1)

            test_task_pred_list.append(task_pred.detach().cpu())
            test_task_labels_list.append(task_labels.detach().cpu())

        test_task_pred_list = torch.cat(test_task_pred_list, dim=0).tolist()
        test_task_labels_list = torch.cat(test_task_labels_list, dim=0).tolist()

        total_acc_test, task_acc_test = get_accuracies(test_task_pred_list, test_task_labels_list, num_tasks)


        with open(save_file_separate, 'a') as f:
            if epoch is None:
                line = f'{exp_idx}'
            else:
                line = f'{exp_idx},{epoch}'
            for t in range(num_tasks):
                if val_dataset is not None:
                    line += f',{task_acc_val[t]:.2f}/{task_acc_test[t]:.2f}'
                else:
                    line += f',{task_acc_test[t]:.2f}'
            f.write(line + '\n')

        with open(save_file_joint, 'a') as f:
            if epoch is None:
                line = f'{exp_idx}'
            else:
                line = f'{exp_idx},{epoch}'
            if val_dataset is None:
                line += f',{total_acc_test:.2f}\n'
            else:
                line += f',{total_acc_val:.2f},{total_acc_test:.2f}\n'
            f.write(line)


def get_accuracies(pred_list, label_list, num_tasks):
    total_acc = sum([1 for p, l in zip(pred_list, label_list) if p == l]) / len(label_list)
    task_acc_list = [sum([1 for p, l in zip(pred_list, label_list) if p == l and l == t]) / label_list.count(t) for t in range(num_tasks)]
    return total_acc, task_acc_list