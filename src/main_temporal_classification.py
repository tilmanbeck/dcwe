import argparse
import logging
import random
import time
import datetime

from sklearn.metrics import f1_score
from torch import optim, nn
from torch.utils.data import DataLoader

from data_helpers import *
from model_temporal import TemporalClassificationModel

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

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default=None, type=str, required=True, help='Name of the dataset.',
                        choices=['debate', 'sandy', 'clex', 'rumours'])
    parser.add_argument('--data_dir', default=None, type=str, required=True, help='Data directory.')
    parser.add_argument('--partition', default='time_stratified_partition', type=str, help='The data partition.')
    parser.add_argument('--results_dir', default='../results_dir', type=str, help='Results directory.')
    parser.add_argument('--trained_dir', default='../trained_dir', type=str, help='Trained model directory.')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size.')
    parser.add_argument('--lr', default=0.0003, type=float, help='Learning rate.')
    parser.add_argument('--n_epochs', default=2, type=int, help='Number of epochs.')
    parser.add_argument('--lambda_a', default=0.1, type=float, help='Regularization constant a.')
    parser.add_argument('--device', default=0, type=int, help='Selected CUDA device.')
    parser.add_argument("--lm_model", default='bert-base-cased', type=str, help='Identifier for pretrained language model.')
    args = parser.parse_args()

    lm_model = args.lm_model


    print('Loading data...')
    time_field = 'date'
    label_field = 'tag'
    dataframe = pd.read_csv(args.data_dir, parse_dates=[time_field], encoding='utf-8')
    nr_classes = len(set(dataframe[label_field].values))
    begin_date = dataframe[time_field].min().to_pydatetime().date()
    task_label_map = label_maps[args.data_name]
    inverse_task_label_map = label_maps_inverse[args.data_name]

    train_dataset = TemporalClassificationDataset(args.data_name, dataframe, 'train', begin_date=begin_date,
                                                  partition=args.partition, label_mapping=task_label_map,
                                                  label_field=label_field, time_field=time_field, lm_model=lm_model)
    dev_dataset = TemporalClassificationDataset(args.data_name, dataframe, 'dev', begin_date, partition=args.partition,
                                                label_mapping=task_label_map,
                                                label_field=label_field, time_field=time_field, lm_model=lm_model)
    test_dataset = TemporalClassificationDataset(args.data_name, dataframe, 'test', begin_date,
                                                 partition=args.partition, label_mapping=task_label_map,
                                                 label_field=label_field, time_field=time_field, lm_model=lm_model)

    lambda_a = args.lambda_a
    lambda_w = lambda_a / 0.001 # see paper by Hofmann et al. ACL 2021
    print('Lambda a: {:.0e}'.format(lambda_a))
    print('Lambda w: {:.0e}'.format(lambda_w))
    print('Number of time units: {}'.format(train_dataset.n_times))
    print('Number of vocabulary items: {}'.format(len(train_dataset.filter_tensor)))

    collator = TemporalClassificationCollator()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collator)#, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collator)

    filename = 'dcwe_{}_{}'.format(args.data_name, args.partition)

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    model = TemporalClassificationModel(
        #n_times=train_dataset.n_times,
        n_times=max(train_dataset.times + dev_dataset.times + test_dataset.times) + 1,
        # we have to use the test_dataset here because we do a temporal split and the oldest dates are in the test split
        nr_classes=nr_classes,
        lm_model=lm_model
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    vocab_filter = train_dataset.filter_tensor.to(device)

    # best_result = get_best('{}/{}.txt'.format(args.results_dir, filename), metric='f1')
    # if best_result:
    #     best_f1 = best_result[0]
    # else:
    #     best_f1 = None
    # print('Best F1 so far: {}'.format(best_f1))

    print('Train model...')
    for epoch in range(1, args.n_epochs + 1):

        model.train()

        for i, batch in enumerate(train_loader):

            if i % 1000 == 0:
                print('Processed {} examples...'.format(i * args.batch_size))

            labels, times, years, months, days, reviews, masks, segs = batch

            labels = labels.to(device)
            reviews = reviews.to(device)
            masks = masks.to(device)
            segs = segs.to(device)

            optimizer.zero_grad()

            offset_t0, offset_t1, output = model(reviews, masks, segs, times, vocab_filter)

            loss = criterion(output, labels.long().view(-1))
            # print('Loss before offsetting: {:.5f}'.format(loss))
            loss += lambda_a * torch.norm(offset_t1, dim=-1).pow(2).mean()
            # print('Loss after offsetting: {:.5f}'.format(loss))
            loss += lambda_w * torch.norm(offset_t1 - offset_t0, dim=-1).pow(2).mean()
            # print('Loss after offsetting diff: {:.5f}'.format(loss))

            loss.backward()

            optimizer.step()

        model.eval()
        print('Evaluate model...')

        y_true = list()
        y_pred = list()

        with torch.no_grad():

            for batch in dev_loader:

                labels, times, years, months, days, reviews, masks, segs = batch

                labels = labels.to(device)
                reviews = reviews.to(device)
                masks = masks.to(device)
                segs = segs.to(device)

                offset_t0, offset_t1, output = model(reviews, masks, segs, times, vocab_filter)

                y_true.extend(labels.tolist())
                # old code for binary classification
                # y_pred.extend(torch.round(output).tolist())
                y_pred.extend(torch.argmax(output, axis=-1).tolist())

        f1_dev_binary = -1.0
        if nr_classes == 2:
            f1_dev_binary = f1_score(y_true, y_pred, average='binary')
        f1_dev_macro = f1_score(y_true, y_pred, average='macro')
        f1_dev_micro = f1_score(y_true, y_pred, average='micro')
        print('Epoch: {}, F1-binary: {:.4f}, F1-macro: {:.4f}, F1-micro: {:.4f}'.format(epoch,
                                                                                        f1_dev_binary,
                                                                                        f1_dev_macro,
                                                                                        f1_dev_micro))
        with open('{}/{}.txt'.format(args.results_dir, "dev_epoch_" + str(epoch)), 'w') as f:
            f.write('binary-F1 dev\tmacro-F1 dev\tmicro-F1 dev\tlr\tlambda_a\tlambda_w\n')
            f.write('{}\t{}\t{}\t{:.0e}\t{:.0e}\t{:.0e}\n'.format(f1_dev_binary,f1_dev_macro, f1_dev_micro,
                                                              args.lr, lambda_a, lambda_w))
        with open('{}/{}_{}.txt'.format(args.results_dir, filename, "dev_epoch_" + str(epoch)), 'w') as f:
            f.write('gold,pred' + '\n')
            for y_t, y_p in zip(y_true, y_pred):
                f.write(inverse_task_label_map[int(y_t)] + ',' + inverse_task_label_map[int(y_p)] + '\n')

    print('Test model...')

    y_true = list()
    y_pred = list()

    with torch.no_grad():

        for batch in test_loader:

            labels, times, years, months, days, reviews, masks, segs = batch

            labels = labels.to(device)
            reviews = reviews.to(device)
            masks = masks.to(device)
            segs = segs.to(device)

            offset_t0, offset_t1, output = model(reviews, masks, segs, times, vocab_filter)

            y_true.extend(labels.tolist())
            # y_pred.extend(torch.round(output).tolist())
            y_pred.extend(torch.argmax(output, axis=-1).tolist())

    f1_test_binary = -1.0
    if nr_classes == 2:
        f1_test_binary = f1_score(y_true, y_pred, average='binary')
    f1_test_macro = f1_score(y_true, y_pred, average='macro')
    f1_test_micro = f1_score(y_true, y_pred, average='micro')


    with open('{}/{}.txt'.format(args.results_dir, filename), 'w') as f:
        f.write('binary-F1 dev\tmacro-F1 dev\tmicro-F1 dev\tbinary-F1 test\tmacro-F1 test\tmicro-F1 test\tlr\tlambda_a\tlambda_w\n')
        f.write('{}\t{}\t{}\t{}\t{}\t{}\t{:.0e}\t{:.0e}\t{:.0e}\n'.format(f1_dev_binary,f1_dev_macro, f1_dev_micro,
                                                                  f1_test_binary, f1_test_macro, f1_test_micro,
                                                          args.lr, lambda_a, lambda_w))

        # if best_f1 is None or f1_dev > best_f1:
        #
        #     best_f1 = f1_dev
        #     torch.save(model.state_dict(), '{}/{}.torch'.format(args.trained_dir, filename))


if __name__ == '__main__':

    start_time = time.time()

    main()

    print('---------- {:.1f} minutes ----------'.format((time.time() - start_time) / 60))
    print()
