import json
import os
import argparse
import logging
import time
from transformers import AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments, IntervalStrategy, \
    EarlyStoppingCallback, AutoTokenizer
import pandas as pd
import numpy as np
from data_helpers import compute_metrics
import datasets
from CONST import label_maps, label_maps_inverse, id_field_map, metrics_for_datasets


def main():

    logging.disable(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default=None, type=str, required=True, help='Name of the dataset.',
                        choices=['debate', 'sandy', 'clex', 'rumours'])
    parser.add_argument('--data_dir', default=None, type=str, required=True, help='Data directory.')
    parser.add_argument('--partition', default='time_stratified_partition', type=str, help='The data partition.')
    parser.add_argument('--results_dir', default='../results_dir', type=str, help='Results directory.')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size.')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate.')
    parser.add_argument('--warmup_ratio', default=0.0, type=float, help='Warmup ratio.')
    parser.add_argument('--weight_decay', default=0.001, type=float, help='Weight decay.')
    parser.add_argument('--n_epochs', default=3, type=int, help='Number of epochs.')
    parser.add_argument("--lm_model", default='bert-base-cased', type=str, help='Identifier for pretrained language model.')
    parser.add_argument("--seed", default=666, type=int)
    parser.add_argument("--max_length", default=64, type=int, help="Maximum length for tokenizer.")
    args = parser.parse_args()

    output_dir = args.results_dir
    seed = args.seed
    max_length = args.max_length

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.join(output_dir, str(seed))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
    dataframe.reset_index(inplace=True, drop=True)
    #################################

        # load split data
    features = datasets.Features({
        'label': datasets.Value("int64"),
        'text': datasets.Value("string"),
        'id': datasets.Value("string")
    })

    tokenizer = AutoTokenizer.from_pretrained(lm_model)

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
        config = AutoConfig.from_pretrained(
            lm_model,
            num_labels=nr_classes
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
        metric_for_best_model=metrics_for_datasets[args.data_name],
        save_total_limit=1,
        seed=seed
    )

    trainer = Trainer(
        model=model_init(),                   # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=validation_dataset,      # evaluation dataset
        compute_metrics=compute_metrics,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # training
    trainer.train()

    # evaluation
    eval_results = trainer.evaluate()
    with open(os.path.join(output_dir, 'eval_results.json'), 'w') as fp:
        json.dump(eval_results, fp)

    # test
    print('Test model..')
    test_results = trainer.predict(test_dataset=test_dataset, )
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as fp:
        json.dump(test_results.metrics, fp)
    preds = test_results.predictions[0] if isinstance(test_results.predictions, tuple) else test_results.predictions
    preds =  [inverse_label_map[i] for i in list(np.argmax(preds, axis=1))]
    truth =  [inverse_label_map[i] for i in list(test_results.label_ids)]
    with open(os.path.join(output_dir, 'test_predictions.csv'), 'w') as fp:
        fp.write('tweet_id,truth,prediction\n')
        for idd,t,p in zip(list(test_data.id),truth, preds):
            fp.write(str(idd) + ',' + t + ',' + p + '\n')
    with open(os.path.join(output_dir, 'training_args.json'), 'w') as fp:
        json.dump(training_args.to_json_string(), fp)

if __name__ == '__main__':

    start_time = time.time()

    main()

    print('---------- {:.1f} minutes ----------'.format((time.time() - start_time) / 60))
    print()
