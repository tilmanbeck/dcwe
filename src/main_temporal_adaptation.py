# BERT with domain adaptation
import logging
from transformers import AutoTokenizer, Trainer, TrainingArguments, IntervalStrategy, EarlyStoppingCallback
import datasets
from collections import Counter, defaultdict
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))

from data_helpers import convert_times, compute_metrics
from utils import convert_timediffs_to_timebins
from CONST import label_maps, label_maps_inverse, id_field_map, metrics_for_datasets
import json
import os

import pandas as pd
from transformers import AutoTokenizer, TrainingArguments, \
    IntervalStrategy, EarlyStoppingCallback
import numpy as np
import argparse
from utils import DomainAdaptationTrainer, compute_metrics_da
from models import BertForSequenceClassificationAndDomainAdaptation, BertForSequenceClassificationAndDomainAdaptationConfig


logging.disable(logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", default=None, type=str, required=True, help='Name of the dataset.',
                    choices=['debate', 'sandy', 'clex', 'rumours'])
parser.add_argument('--data_dir', default=None, type=str, required=True, help='Data directory.')
parser.add_argument('--partition', default='time_stratified_partition', type=str, help='The data partition.')
parser.add_argument('--results_dir', default='../results_dir', type=str, help='Results directory.')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size.')
parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate.')
parser.add_argument('--warmup_ratio', default=0.1, type=float, help='Warmup ratio.')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay.')
parser.add_argument('--n_epochs', default=4, type=int, help='Number of epochs.')
parser.add_argument('--alpha', default=0.1, type=float, help='alpha value for secondary loss.')
parser.add_argument('--device', default=0, type=int, help='Selected CUDA device.')
parser.add_argument("--lm_model", default='bert-base-cased', type=str, help='Identifier for pretrained language model.')
parser.add_argument("--seed", default=666, type=int)
parser.add_argument("--max_length", default=64, type=int, help="Maximum length for tokenizer.")
parser.add_argument("--early_stopping_patience", default=3, type=int, help="Early stopping trials before stopping.")
args = parser.parse_args()

if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)
output_dir = args.results_dir
max_length = args.max_length

seed = args.seed
lm_model = args.lm_model
label_map = label_maps[args.data_name]
inverse_label_map = label_maps_inverse[args.data_name]

print('Loading data...')
time_field = 'date'
label_field = 'tag'
id_field = id_field_map[args.data_name]
dataframe = pd.read_csv(args.data_dir)
nr_classes = len(set(dataframe[label_field].values))

######## FORMAT DATA ############
# rename data columns to common format
dataframe['id'] = dataframe[id_field].astype(str)
dataframe.rename(columns={label_field: 'label', time_field: 'time'}, inplace=True)
# convert string labels to numeric
dataframe['label'] = dataframe['label'].replace(label_map)
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
dataframe['time_labels'] = dataframe['time'].map(lambda ex: convert_times(ex, name=args.data_name, begin_date=begin_date))
dataframe['time_labels'] = dataframe['time_labels'].replace(timediffs_to_timebins)
#################################

tokenizer = AutoTokenizer.from_pretrained(lm_model)
# load split data
features = datasets.Features({
    'label': datasets.Value("int64"),
    'text': datasets.Value("string"),
    'time_labels': datasets.Value("int64"),
    'id': datasets.Value("string")
})

train_data = dataframe[dataframe[args.partition] == "train"]
train_data = pd.DataFrame(dataframe.iloc[train_data.index])
train_dataset = datasets.Dataset.from_pandas(train_data, features=features).map(
    lambda ex: tokenizer(ex['text'], max_length=max_length, truncation=True, padding='max_length'), batched=True)

validation_data = dataframe[dataframe[args.partition] == "dev"]
validation_data = pd.DataFrame(dataframe.iloc[validation_data.index])
validation_dataset = datasets.Dataset.from_pandas(validation_data, features=features).map(
    lambda ex: tokenizer(ex['text'], max_length=max_length, truncation=True, padding='max_length'), batched=True)

test_data = dataframe[dataframe[args.partition] == "test"]
test_data = pd.DataFrame(dataframe.iloc[test_data.index])
test_dataset = datasets.Dataset.from_pandas(test_data, features=features).map(
    lambda ex: tokenizer(ex['text'], max_length=max_length, truncation=True, padding='max_length'), batched=True)

# model preparation
def model_init():
    config = BertForSequenceClassificationAndDomainAdaptationConfig(
        num_labels=nr_classes,
        num_temporal_classes=n_times,
        alpha=args.alpha
    )
    model = BertForSequenceClassificationAndDomainAdaptation(config)
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
    metric_for_best_model=metrics_for_datasets[args.data_name],
    save_total_limit=1,
    seed=seed,
    label_names=['labels', 'time_labels']
)

trainer = DomainAdaptationTrainer(
    model=model_init(),                   # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=validation_dataset,      # evaluation dataset
    compute_metrics=compute_metrics_da,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
)

trainer.train()

eval_results = trainer.evaluate()

test_results = trainer.predict(test_dataset=test_dataset)
preds = test_results.predictions[0] if isinstance(test_results.predictions, tuple) else test_results.predictions
# preds[0] contains class predictions, preds[0] contains topic predictions
class_preds = [inverse_label_map[i] for i in list(np.argmax(preds[0], axis=1))]
time_preds = [i for i in list(np.argmax(preds[1], axis=1))]
# test_results.label_ids[0] contains class labels, test_results.label_ids[1] contains topic labels
class_truth =  [inverse_label_map[i] for i in list(test_results.label_ids[0])]
time_truth =  [i for i in list(test_results.label_ids[1])]

with open(os.path.join(output_dir, 'training_args.json'), 'w') as fp:
    json.dump(training_args.to_json_string(), fp)
with open(os.path.join(output_dir, 'results.json'), 'w') as fp:
    json.dump({**test_results.metrics, **eval_results, 'best_model_checkpoint': trainer.state.best_model_checkpoint}, fp)
with open(os.path.join(output_dir, 'class_predictions.csv'), 'w') as fp:
    fp.write('tweet_id,time_label,time_prediction,label,prediction\n')
    for idd,time_label,time_pred,label,pred in zip(list(test_data.id), time_truth, time_preds, class_truth, class_preds):
        fp.write(str(idd) + ',' + str(time_label) + ',' + str(time_pred) + ',' + str(label) + ',' + str(pred) + '\n')
