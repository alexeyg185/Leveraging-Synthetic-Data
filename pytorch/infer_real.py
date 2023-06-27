"""Sample PyTorch Inference script
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
from collections import OrderedDict
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.autograd as autograd
import torch.utils.data as data

import model_factory

import matplotlib.image as mpimg
from dataset import Dataset

import matplotlib

from pytorch.infer_gt_labels import load_named_labels_real, load_named_labels_synth
from pytorch.tools import accuracy, AverageMeter

# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


"""
-m resnet50 --restore-checkpoint /home/alexey.gruzdev/Documents/bench_project/code/nns/pytorch/old_saves/checkpoint_synth_31_aug.pth.tar /home/alexey.gruzdev/Documents/bench_project/synth_data/renders/validation
"""

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--output_dir', metavar='DIR', default='./',
                    help='path to output files')
parser.add_argument('--model', '-m', metavar='MODEL', default='resnet18',
                    help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=224, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--restore-checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--multi-gpu', dest='multi_gpu', action='store_true',
                    help='use multiple-gpus')
parser.add_argument('--no-test-pool', dest='test_time_pool', action='store_false',
                    help='use pre-trained model')


def main():
    args = parser.parse_args()

    # create model
    num_classes = 17
    model = model_factory.create_model(
        args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        test_time_pool=args.test_time_pool)

    # resume from a checkpoint
    if args.restore_checkpoint and os.path.isfile(args.restore_checkpoint):
        print("=> loading checkpoint '{}'".format(args.restore_checkpoint))
        checkpoint = torch.load(args.restore_checkpoint)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state = checkpoint['state_dict']
            new_state = OrderedDict()
            for k, v in state.items():
                new_k = k.replace('module.', '')
                new_state[new_k] = v
            model.load_state_dict(new_state)
        else:
            model.load_state_dict(checkpoint)
        print("=> loaded checkpoint '{}'".format(args.restore_checkpoint))
    elif not args.pretrained:
        print("=> no checkpoint found at '{}'".format(args.restore_checkpoint))
        exit(1)

    if args.multi_gpu:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    transforms = model_factory.get_transforms_eval(
        args.model,
        args.img_size)

    dataset = Dataset(
        root=args.data,
        transform=transforms,
        dir_structure_mode='real')

    # imgs = dataset.imgs
    # random.shuffle(imgs)
    # imgs = randomly_leave(data=imgs, percent=0.01)
    # dataset.imgs = imgs
    # # leave_only_validation(imgs)
    names_labels = load_named_labels_synth(validation_path=args.data)

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model.eval()
    top5_ids = predict(args.print_freq, loader, model)

    pred = build_image_gt_predict_map(imgs=dataset.imgs, predicts=top5_ids)
    save_file(args.output_dir, dataset, top5_ids)
    visualize_pictures(pred=pred, names_labels=names_labels)

def build_image_gt_predict_map(imgs: List[Tuple], predicts: np.ndarray):
    ret = []
    for item in zip(imgs,predicts):
        path = item[0][0]
        gt_label = item[0][1]
        preds = list(item[1])
        ret.append((path,gt_label,preds))
    return ret


def visualize_pictures(pred, names_labels):
    def label_idx_to_name(idx_label):
        name_label = names_labels[idx_label]
        return ' '.join(name_label)

    def gt_label_to_name(label):
        return label_idx_to_name(label)

    def pred_label_to_names(labels: List):
        name_labels = [label_idx_to_name(l) for l in labels]
        return '; '.join(name_labels)

    for img in pred:
        path = img[0]

        gt_label = gt_label_to_name(img[1])
        gt_label = f"Ground truth label: {gt_label}"

        pred_labels = pred_label_to_names(img[2])
        pred_labels = f"Pred labels: {pred_labels}"

        # plt.imshow(mpimg.imread(path))
        # plt.text(x=0, y=0, s=gt_label)
        # plt.text(x=0, y=30, s=pred_labels)
        # plt.show()
        # plt.close()


def leave_only_validation(imgs):
    to_remove = []
    for val in imgs:
        if '/validation/' not in val[0]:
            to_remove.append(val)
    for r in to_remove:
        imgs.remove(r)

def randomly_leave(data:List, percent: float):
    K = int(len(data)*percent)
    return random.sample(data, k=K)


def predict(print_freq, loader, model):
    batch_time = AverageMeter(name='batch_time')
    top1 = AverageMeter('Acc@1', ':6.2f')
    topk = AverageMeter(f'Acc@{3}', ':6.2f')
    top5 = AverageMeter(f'Acc@{5}', ':6.2f')
    end = time.time()
    top5_ids = []
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            input = input.cuda()
            labels = model(input)
            top5_ids.append(labels.topk(5)[1].cpu().numpy())

            accuracies, all_pred_results, top1_target_vs_pred = accuracy(labels.cpu(), target, topk=(1, 3, 5))
            acc1, acck, acc5 = accuracies
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            top1.update(acc1[0], input.size(0))
            topk.update(acck[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            if batch_idx % print_freq == 0:
                print('Predict: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    batch_idx, len(loader), batch_time=batch_time))
                print(f"Accuracy top1 = {top1.avg} top3 = {topk.avg} top5 = {top5.avg}")
    print(f"FINAL: Accuracy top1 = {top1.avg} top3 = {topk.avg} top5 = {top5.avg}")
    top5_ids = np.concatenate(top5_ids, axis=0).squeeze()
    return top5_ids


def save_file(output_dir, dataset, top5_ids):
    with open(os.path.join(output_dir, './top5_ids.csv'), 'w') as out_file:
        filenames = dataset.filenames()
        for filename, label in zip(filenames, top5_ids):
            filename = os.path.basename(filename)
            out_file.write('{0},{1},{2},{3},{4},{5}\n'.format(
                filename, label[0], label[1], label[2], label[3], label[4]))


if __name__ == '__main__':
    main()
