import json
import os
from typing import List


def retrieve_path_names(root_path: str):
    return os.listdir(root_path)


def load_synset_mapping_file(file_path: str):
    with open(file_path) as f:
        lines = f.readlines()
        return lines


def parse_synset_mapping_file(lines: List[str]):
    syn_to_class_name = {}
    for line in lines:
        strings = line.split()
        syn_to_class_name[strings[0]] = strings[1:]
    return syn_to_class_name


def store_dict_as_json(data, file_path):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)


def load_json_as_dict(file_path):
    with open(file_path) as json_file:
        return json.load(json_file)


def build_index_to_class_names_mapping(synset_labels: List[str], synset_mapping):
    ret = []
    synset_labels = sorted(synset_labels)
    for l in synset_labels:
        ret.append(synset_mapping[l])
    return ret


def filter_only_by_existing_labels(existing_labels, synset_mapping):
    all_keys = set(synset_mapping.keys())
    leave_keys = set(existing_labels)
    to_remove_keys = all_keys - leave_keys
    for k in to_remove_keys:
        del synset_mapping[k]
    return synset_mapping


def store():
    path_names = sorted(
        retrieve_path_names('/home/alexey.gruzdev/Documents/bench_project/imagenet_dataset/reduced/train'))
    mapping = parse_synset_mapping_file(load_synset_mapping_file(
        '/home/alexey.gruzdev/Documents/bench_project/imagenet_dataset/LOC_synset_mapping.txt'))
    mapping = filter_only_by_existing_labels(existing_labels=path_names, synset_mapping=mapping)
    store_dict_as_json(data=mapping,
                       file_path='/home/alexey.gruzdev/Documents/bench_project/imagenet_dataset/reduced/synset_to_class_names.json')


def load_named_labels_real(validation_path: str):
    path_names = retrieve_path_names(
        '/home/alexey.gruzdev/Documents/bench_project/imagenet_dataset/reduced/validation')
    mapping = load_json_as_dict(
        '/home/alexey.gruzdev/Documents/bench_project/imagenet_dataset/reduced/synset_to_class_names.json')
    return build_index_to_class_names_mapping(path_names, mapping)


def load_named_labels_synth(validation_path: str):
    path_names = retrieve_path_names(
        validation_path)
    path_names = sorted(path_names)
    ret = []
    for path_name in path_names:
        ret.append([path_name])
    return ret


if __name__ == '__main__':
    load_named_labels_real()
    # store()
