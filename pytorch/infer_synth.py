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
import torch
import torchmetrics

import numpy as np
import torch
import torch.utils.data as data
from torchmetrics import Precision, Recall, Accuracy
from torchvision.transforms import Resize, ToTensor, Compose

import model_factory

import matplotlib.image as mpimg

import matplotlib

from pytorch.data import SyntheticDataset
from pytorch.infer_gt_labels import load_named_labels_synth
from pytorch.tools import accuracy, AverageMeter, accuracy_per_class, convert_to_sorted_dict, plot_kernels, \
    get_model_weights, vis_model
from pytorch.visual.cnn_vis import CNNLayerVisualization

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

num_classes = 15

"""
-m resnet50 --restore-checkpoint /home/alexey.gruzdev/Documents/bench_project/code/nns/pytorch/old_saves/checkpoint_synth_31_aug.pth.tar /home/alexey.gruzdev/Documents/bench_project/synth_data/renders/validation
"""

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--data', metavar='DIR', nargs='+',
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
parser.add_argument('--incl-background', default=-1, type=int,
                    metavar='N', help='True/False(default: -1 (auto))')

def main():
    args = parser.parse_args()

    # create model
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


    for valid_path in args.data:
        print(f"Handling {valid_path}")
        sd=model.state_dict()
        # visualize kernels
        # CNNLayerVisualization(model, num_classes, 5).visualise_layer_with_hooks()
        # vis_model(model)

        if args.multi_gpu:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

        transforms = Compose([Resize(size=[224, 224]), ToTensor()])

        dataset = SyntheticDataset(
            root=valid_path,
            transform=transforms,
            include_background=args.incl_background,
            test=True
        )

        # dataset = Dataset(
        #     root=valid_path,
        #     transform=transforms,
        #     dir_structure_mode='synth')

        # imgs = dataset.imgs
        # random.shuffle(imgs)
        # imgs = randomly_leave(data=imgs, percent=0.01)
        # dataset.imgs = imgs
        # leave_only_validation(imgs)
        named_labels = load_named_labels_synth(validation_path=valid_path)

        loader = data.DataLoader(
            dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        model.eval()
        top5_ids = predict(args.print_freq, loader, model, named_labels=named_labels)

        pred = build_image_gt_predict_map(imgs=dataset.imgs, predicts=top5_ids)
        save_file(args.output_dir, dataset, top5_ids)
        random.shuffle(pred)
        # visualize_pictures(pred=pred, names_labels=named_labels)


def build_image_gt_predict_map(imgs: List[Tuple], predicts: np.ndarray):
    ret = []
    for item in zip(imgs, predicts):
        path = item[0][0]
        gt_label = item[0][1]
        preds = list(item[1])
        ret.append((path, gt_label, preds))
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

        plt.imshow(mpimg.imread(path))
        plt.text(x=0, y=0, s=gt_label)
        plt.text(x=0, y=30, s=pred_labels)
        plt.show()
        plt.close()


def leave_only_validation(imgs):
    to_remove = []
    for val in imgs:
        if '/validation/' not in val[0]:
            to_remove.append(val)
    for r in to_remove:
        imgs.remove(r)


def randomly_leave(data: List, percent: float):
    K = int(len(data) * percent)
    return random.sample(data, k=K)


def predict(print_freq, loader, model, named_labels):
    batch_time = AverageMeter(name='batch_time')
    top1 = AverageMeter('Acc@1', ':6.2f')
    topk = AverageMeter(f'Acc@{3}', ':6.2f')
    top5 = AverageMeter(f'Acc@{5}', ':6.2f')
    precision = Precision(average='macro', num_classes=num_classes)
    recall = Recall(average='macro', num_classes=num_classes)
    accuracy_tm = Accuracy(num_classes=num_classes, top_k=1)
    end = time.time()
    top5_ids = []
    top1_pred_results = []
    infered_file_names = []
    top1_target_vs_pred_all=[]
    topk_values = (1, 3, 5)

    keys = {i for i in range(0, num_classes)}
    correct_preds = []
    correct_percents = []
    for t in topk_values:
        correct_preds.append(dict())
        correct_percents.append(dict())
    total_pred = dict()

    for k in keys:
        for correct_pred in correct_preds:
            correct_pred[k] = 0
        total_pred[k] = 0

    with torch.no_grad():
        for batch_idx, (input, labels_gt, fpaths) in enumerate(loader):
            input = input.cuda()
            labels_predict = model(input)
            top5_ids.append(labels_predict.topk(max(topk_values))[1].cpu().numpy())

            labels_predict_cpu = labels_predict.cpu()
            accuracies, all_pred_results, top1_target_vs_pred = accuracy(output=labels_predict_cpu, target=labels_gt, topk=topk_values)
            acc1, acck, acc5 = accuracies
            short_fpaths = list(map(file_name_to_short, fpaths))
            infered_file_names.extend(short_fpaths)
            top1_pred_results.extend(all_pred_results[0][0])
            top1_target_vs_pred_all.extend(top1_target_vs_pred)
            accuracy_per_class(output=labels_predict_cpu, target=labels_gt, correct_preds=correct_preds, total_pred=total_pred,
                               topk=topk_values)

            precision.update(preds=labels_predict_cpu, target=labels_gt)
            recall.update(preds=labels_predict_cpu, target=labels_gt)
            accuracy_tm.update(preds=labels_predict_cpu, target=labels_gt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            top1.update(acc1[0], input.size(0))
            topk.update(acck[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # if batch_idx % print_freq == 0:
            #     print('Predict: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
            #         batch_idx, len(loader), batch_time=batch_time))
                # print(f"Accuracy top1 = {top1.avg} top3 = {topk.avg} top5 = {top5.avg}")
    for v in zip(correct_preds, correct_percents):
        for class_idx, correct_count in v[0].items():
            v[1][named_labels[class_idx][0]] = round(100 * float(correct_count) / total_pred[class_idx])

    filenames_correct = dict(zip(infered_file_names, top1_pred_results))
    filenames_target_vs_pred = dict(zip(infered_file_names, top1_target_vs_pred_all))

    # FINAL
    sep_str = "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    print(sep_str)
    print("RESULTS")
    print(f"Accuracy: top1 = {top1.avg} top3 = {topk.avg} top5 = {top5.avg}")
    dict(sorted(correct_percents[0].items(), key=lambda x: x[1], reverse=True))
    print(
        f"Accuracy: per class:\n"
        f"top1 = {convert_to_sorted_dict(correct_percents[0])}\n"
        f"top3 = {convert_to_sorted_dict(correct_percents[1])}\n"
        f"top5 = {convert_to_sorted_dict(correct_percents[2])}")

    print(f"Precision = {precision.compute() * 100}")
    print(f"Recall = {recall.compute()* 100}")
    # print(f"Accuracy = {accuracy_tm.compute()* 100}")

    print(filenames_correct)
    print(filenames_target_vs_pred)

    print(sep_str)
    top5_ids = np.concatenate(top5_ids, axis=0).squeeze()
    return top5_ids

def file_name_to_short(filename):
    sp = filename.split('/')
    return '/'.join(sp[-2:])

def save_file(output_dir, dataset, top5_ids):
    with open(os.path.join(output_dir, './top5_ids.csv'), 'w') as out_file:
        filenames = dataset.filenames()
        for filename, label in zip(filenames, top5_ids):
            filename = os.path.basename(filename)
            out_file.write('{0},{1},{2},{3},{4},{5}\n'.format(
                filename, label[0], label[1], label[2], label[3], label[4]))


if __name__ == '__main__':
    main()

