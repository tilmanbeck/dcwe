import os
import argparse
import logging
import random
import time
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, Trainer, TrainingArguments, IntervalStrategy, EarlyStoppingCallback
import datasets
from collections import Counter, defaultdict
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))

from data_helpers import convert_times, compute_metrics
from model_temporal import TemporalClassificationModelNew


def convert_timediffs_to_timebins(time_diffs):
    time_diffs = sorted(time_diffs)
    start = 1 # we start at 1 because we need an initial bin (without data) for the offset computation in model.forward
    conversion = {}
    for i in time_diffs:
        if i not in conversion.keys():
            conversion[i]  = start
            start += 1
    return conversion


label_maps = {
    'debate': {'claim': 1, 'noclaim': 0},
    'sandy': {'y': 1, 'n': 0},
    'rumours':  {'comment': 0, 'deny': 1, 'support': 2, 'query': 3},
    'clex': {'Related - but not informative': 0, 'Not related': 1,
             'Related and informative': 2, 'Not applicable': 3}
}

label_maps_inverse = {
    'debate': {1: 'claim', 0: 'noclaim'},
    'sandy': {1: 'y', 0:'n'},
    'rumours':  {0: 'comment', 1: 'deny', 2:'support', 3:'query'},
    'clex': {0: 'Related - but not informative', 1: 'Not related',
             2: 'Related and informative', 3:'Not applicable'}
}

def main():

    logging.disable(logging.WARNING)

    seed = 666
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default=None, type=str, required=True, help='Name of the dataset.',
                        choices=['debate', 'sandy', 'clex', 'rumours'])
    parser.add_argument('--data_dir', default=None, type=str, required=True, help='Data directory.')
    parser.add_argument('--partition', default='time_stratified_partition', type=str, help='The data partition.')
    parser.add_argument('--results_dir', default='../results_dir', type=str, help='Results directory.')
    parser.add_argument('--trained_dir', default='../trained_dir', type=str, help='Trained model directory.')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size.')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate.')
    parser.add_argument('--warmup_ratio', default=0.1, type=float, help='Warmup ratio.')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay.')
    parser.add_argument('--n_epochs', default=1, type=int, help='Number of epochs.')
    parser.add_argument('--lambda_a', default=0.1, type=float, help='Regularization constant a.')
    parser.add_argument('--device', default=0, type=int, help='Selected CUDA device.')
    parser.add_argument("--lm_model", default='bert-base-cased', type=str, help='Identifier for pretrained language model.')
    parser.add_argument("--top_n_frequent_words", default=1000, type=int, help="")
    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    output_dir = args.results_dir

    lm_model = args.lm_model
    label_map = label_maps[args.data_name]
    inverse_label_map = label_maps_inverse[args.data_name]

    print('Loading data...')
    time_field = 'date'
    label_field = 'tag'
    dataframe = pd.read_csv(args.data_dir)
    nr_classes = len(set(dataframe[label_field].values))

    ######## FORMAT DATA ############
    # rename data columns to common format
    dataframe.rename(columns={label_field: 'label', time_field: 'time'}, inplace=True)
    # convert string labels to numeric
    dataframe['label'] = dataframe['label'].replace(label_map)
    dataframe.dropna(subset=[args.partition, 'label', 'time'], inplace=True)
    dataframe.time = pd.to_datetime(dataframe.time)
    dataframe.reset_index(inplace=True, drop=True)
    begin_date = dataframe['time'].min().to_pydatetime().date()
    # compute the difference between begin_date and current data for all datapoints
    time_diffs = [convert_times(i, name=args.data_name, begin_date=begin_date) for i in dataframe.time]
    n_times = len(set(time_diffs))
    # there might be missing dates in the data therefore we need to map the actual dates to time bins
    # we do take into account the missing dates; for t, the preceeding date t-1 is always the preceding data point in time (not the actual date before t)
    # therefore we create less parameters for the DCWE model and have more datapoints for a specific date
    timediffs_to_timebins = convert_timediffs_to_timebins(list(set(time_diffs)))
    dataframe['timediff'] = dataframe['time'].map(lambda ex: convert_times(ex, name=args.data_name, begin_date=begin_date))
    dataframe['timediff'] = dataframe['timediff'].replace(timediffs_to_timebins)
    #################################


    tokenizer = AutoTokenizer.from_pretrained(lm_model, use_fast=False)

    ############ PREPARE OFFSET FILTER TENSORS #################
    vocab = defaultdict(Counter)
    for text, time in zip(dataframe.text, time_diffs):
        vocab[time].update(text.strip().split())
    for time in vocab:
        total = sum(vocab[time].values())
        vocab[time] = {w: count / total for w, count in vocab[time].items()}
    w_counts = dict()
    for time in vocab:
        for w in vocab[time]:
            w_counts[w] = w_counts.get(w, 0) + vocab[time][w]
    w_top = sorted(w_counts.keys(), key=lambda x: w_counts[x], reverse=True)[:args.top_n_frequent_words]
    filter_list = [w for w in w_top if w not in stops and w in tokenizer.vocab and w.isalpha()]
    #what does the 2100 actually say? frequency?
    filter_tensor = torch.tensor([t for t in tokenizer.encode(filter_list) if t >= 2100])
    ###############################################################

    # load split data
    features = datasets.Features({
        'label': datasets.Value("int64"),
        'text': datasets.Value("string"),
        'timediff': datasets.Value("int64")
    })
    train_data = dataframe[dataframe[args.partition] == "train"]
    train_data = pd.DataFrame(dataframe.iloc[train_data.index])
    train_dataset = datasets.Dataset.from_pandas(train_data, features=features).map(
        lambda ex: tokenizer(ex['text'], truncation=True, padding='max_length'), batched=True)

    validation_data = dataframe[dataframe[args.partition] == "dev"]
    validation_data = pd.DataFrame(dataframe.iloc[validation_data.index])
    validation_dataset = datasets.Dataset.from_pandas(validation_data, features=features).map(
        lambda ex: tokenizer(ex['text'], truncation=True, padding='max_length'), batched=True)

    test_data = dataframe[dataframe[args.partition] == "test"]
    test_data = pd.DataFrame(dataframe.iloc[test_data.index])
    test_dataset = datasets.Dataset.from_pandas(test_data, features=features).map(
        lambda ex: tokenizer(ex['text'], truncation=True, padding='max_length'), batched=True)

    lambda_a = args.lambda_a
    lambda_w = lambda_a / 0.001 # see paper by Hofmann et al. ACL 2021
    print('Lambda a: {:.0e}'.format(lambda_a))
    print('Lambda w: {:.0e}'.format(lambda_w))
    print('Number of time units: {}'.format(n_times))
    print('Number of vocabulary items: {}'.format(len(filter_tensor)))

    filename = 'dcwe_{}_{}'.format(args.data_name, args.partition)

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    vocab_filter = filter_tensor.to(device)

    # model preparation
    def model_init():
        model = TemporalClassificationModelNew(
            n_times=n_times + 1,
            # we have to use the test_dataset here because we do a temporal split and the oldest dates are in the test split
            vocab_filter=vocab_filter,
            nr_classes=nr_classes,
            lm_model=lm_model
        )
        return model

    training_args = TrainingArguments(
        output_dir=output_dir,          # output directory
        num_train_epochs=args.n_epochs,              # total number of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
        warmup_ratio=args.warmup_ratio,                # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,               # strength of weight decay
        evaluation_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        save_total_limit=1,
        seed=seed
    )

    trainer = Trainer(
        model=model_init(),                   # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=validation_dataset,      # evaluation dataset
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # evaluate before training
    eval_results_before = trainer.evaluate()
    print(eval_results_before)

    # training
    print('Train model..')
    trainer.train()

    # evaluation
    print('Evaluate model..')
    eval_results = trainer.evaluate()
    print(eval_results)

    # test
    print('Test model..')
    test_results = trainer.predict(test_dataset=test_dataset)
    print(test_results)
    preds = test_results.predictions[0] if isinstance(test_results.predictions, tuple) else test_results.predictions
    preds =  [inverse_label_map[i] for i in list(np.argmax(preds, axis=1))]
    truth =  [inverse_label_map[i] for i in list(test_results.label_ids)]


if __name__ == '__main__':

    start_time = time.time()

    main()

    print('---------- {:.1f} minutes ----------'.format((time.time() - start_time) / 60))
    print()
