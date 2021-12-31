import os
import argparse
import logging
import random
import time

from torch import optim, nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments, IntervalStrategy, \
    EarlyStoppingCallback, EvalPrediction, AutoTokenizer
from data_helpers import *
from datasets import Dataset



label_maps = {
    'debate': {'claim': 1, 'noclaim': 0},
    'sandy': {'y': 1, 'n': 0},
    'rumours':  {'comment': 2, 'deny': 1, 'support': 3, 'query': 0},
    'clex': {'Related - but not informative': 2, 'Not related': 1,
             'Related and informative': 3, 'Not applicable': 0}
}

label_maps_inverse = {
    'debate': {1: 'claim', 0: 'noclaim'},
    'sandy': {1: 'y', 0:'n'},
    'rumours':  {0: 'query', 1: 'deny', 2:'comment', 3:'support'},
    'clex': {2: 'Related - but not informative', 1: 'Not related',
             3: 'Related and informative', 0:'Not applicable'}
}


def main():

    logging.disable(logging.WARNING)

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
    parser.add_argument('--n_epochs', default=2, type=int, help='Number of epochs.')
    parser.add_argument("--lm_model", default='bert-base-cased', type=str, help='Identifier for pretrained language model.')
    args = parser.parse_args()

    seed = 666

    lm_model = args.lm_model
    output_dir = args.results_dir
    label_map = label_maps[args.data_name]
    inverse_label_map = label_maps_inverse[args.data_name]

    print('Loading data...')
    time_field = 'date'
    label_field = 'tag'
    dataframe = pd.read_csv(args.data_dir)
    n_labels = len(set(dataframe[label_field].values))
    # take the corresponding split (if it exists otherwise take random split)

    # rename data columns to common format
    dataframe.rename(columns={label_field: 'label', time_field: 'time'}, inplace=True)
    # convert string labels to numeric
    dataframe['label'] = dataframe['label'].replace(label_map)

    dataframe.dropna(subset=[args.partition, 'label', 'time'], inplace=True)
    dataframe.time = pd.to_datetime(dataframe.time)
    dataframe.reset_index(inplace=True, drop=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = output_dir

    tokenizer = AutoTokenizer.from_pretrained(lm_model)
    # load split data
    train_data = dataframe[dataframe[args.partition] == "train"]
    train_data = pd.DataFrame(dataframe.iloc[train_data.index])
    train_dataset = Dataset.from_pandas(train_data).map(
        lambda ex: tokenizer(ex['text'], truncation=True, padding='max_length'), batched=True)

    validation_data = dataframe[dataframe[args.partition] == "dev"]
    validation_data = pd.DataFrame(dataframe.iloc[validation_data.index])
    validation_dataset = Dataset.from_pandas(validation_data).map(
        lambda ex: tokenizer(ex['text'], truncation=True, padding='max_length'), batched=True)

    test_data = dataframe[dataframe[args.partition] == "test"]
    test_data = pd.DataFrame(dataframe.iloc[test_data.index])
    test_dataset = Dataset.from_pandas(test_data).map(
        lambda ex: tokenizer(ex['text'], truncation=True, padding='max_length'), batched=True)

    filename = 'dcwe_{}_{}'.format(args.data_name, args.partition)

    # model preparation
    def model_init():
        config = AutoConfig.from_pretrained(
            lm_model,
            num_labels=n_labels
        )
        model = AutoModelForSequenceClassification.from_pretrained(lm_model, config=config)
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    # evaluate before training
    eval_results_before = trainer.evaluate()
    print(eval_results_before)

    # training
    trainer.train()

    # evaluation
    eval_results = trainer.evaluate()
    print(eval_results)

    # test
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
