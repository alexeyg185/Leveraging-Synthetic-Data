from math import sqrt
from typing import Dict, List

import torch
from matplotlib import pyplot as plt
from torch import nn


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        total = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = (pred == target)

        top_1_5_pred_results = []
        top_1_5_results = []

        for k in topk:
            topk_correct = correct[:k]
            correct_k = topk_correct.reshape(-1).float().sum(0, keepdim=True)
            acc = 100 * correct_k / total
            top_1_5_results.append(acc)
            top_1_5_pred_results.append(topk_correct.tolist())

        return top_1_5_results, top_1_5_pred_results, list(zip(target.tolist(), pred[:1][0].tolist()))


def accuracy_per_class(output, target, correct_preds: List[Dict], total_pred: Dict, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = (pred == target)
        correct_list = correct.T.tolist()
        for i, top_k_correct in enumerate(correct_list):
            klass = target[i].item()
            total_pred[klass] += 1
            for i, k in enumerate(topk):
                correct_preds[i][klass] += sum(top_k_correct[:k])
        return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def convert_to_sorted_dict(d):
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=True))


def get_model_weights(model_children):
    counter = 0
    model_weights = []  # we will save the conv layer weights in this list
    conv_layers = []  # we will save the 49 conv layers in this list

    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolutional layers: {counter}")
    return model_weights, conv_layers


def plot_kernels(model_weights, figsize=(20, 17), save_path='/tmp/',show=True, save=False):
    suplt_size = model_weights.shape[0]
    suplt_size = int(sqrt(suplt_size))
    plt.figure(figsize=figsize)
    for i, filter in enumerate(model_weights):
        plt.subplot(suplt_size, suplt_size,
                    i + 1)  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        # plt.imshow(filter[0, :, :].detach(), cmap='gray')
        filter_s = filter.shape[0]
        plt.imshow(filter[:, :, :].detach())
        plt.axis('off')
    if show:
        plt.show()

    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # plt.show()
    if save:
        plt.savefig(save_path)



def vis_model(model):
    model_weights, conv_layers = get_model_weights(list(model.children()))
    for i, mw in enumerate(model_weights):
        try:
            plot_kernels(mw, figsize=(20, 17), show=True, save=False,
                         save_path=f"/home/alexey.gruzdev/Documents/bench_project/visual_poc/weights_vis/trained_imagenet/{i}.png")
        except:
            pass