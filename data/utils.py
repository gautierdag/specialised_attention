import torch
import torchtext

from machine.tasks import get_task
from machine.dataset import SourceField, TargetField

from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_iters(task_name="long_lookup",
              batch_size=1,
              eval_batch_size=512,
              max_len=50):

    T = get_task(task_name)
    train_path = T.train_path
    valid_path = T.valid_path
    test_paths = T.test_paths

    # generate training and testing data
    train_dataset, valid_dataset, test_datasets, src, tgt = get_train_valid_tests(
        train_path, valid_path, test_paths, max_len)

    train_iter = get_standard_batch_iterator(train_dataset, batch_size)
    valid_iter = get_standard_batch_iterator(valid_dataset, eval_batch_size)

    tests_iters = OrderedDict()
    for i, dataset in enumerate(test_paths):
        tests_iters[dataset] = get_standard_batch_iterator(
            test_datasets[i], eval_batch_size)

    return train_iter, valid_iter, tests_iters, src, tgt


def get_train_valid_tests(train_path, valid_path, test_paths, max_len,
                          src_vocab=50000, tgt_vocab=50000):
    """Gets the formatted train, valid, and test data."""

    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len

    src = SourceField()
    tgt = TargetField(include_eos=True)
    fields = [('src', src), ('tgt', tgt)]
    train = torchtext.data.TabularDataset(
        path=train_path, format='tsv', fields=fields, filter_pred=len_filter)
    valid = torchtext.data.TabularDataset(
        path=valid_path, format='tsv', fields=fields, filter_pred=len_filter)

    tests = []
    for t in test_paths:
        tests.append(torchtext.data.TabularDataset(path=t, format='tsv',
                                                   fields=fields,
                                                   filter_pred=len_filter))

    src.build_vocab(train, max_size=src_vocab)
    tgt.build_vocab(train, max_size=tgt_vocab)

    return train, valid, tests, src, tgt


def get_standard_batch_iterator(data, batch_size):
    return torchtext.data.BucketIterator(
        dataset=data, batch_size=batch_size,
        sort=False, sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device, repeat=False)
