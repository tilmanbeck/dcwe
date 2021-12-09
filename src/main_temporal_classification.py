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


def main():

    logging.disable(logging.WARNING)

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default=None, type=str, required=True, help='Name of the dataset.')
    parser.add_argument('--data_dir', default=None, type=str, required=True, help='Data directory.')
    parser.add_argument('--results_dir', default='../results_dir', type=str, help='Results directory.')
    parser.add_argument('--trained_dir', default='../trained_dir', type=str, help='Trained model directory.')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size.')
    parser.add_argument('--lr', default=0.0003, type=float, help='Learning rate.')
    parser.add_argument('--n_epochs', default=2, type=int, help='Number of epochs.')
    parser.add_argument('--lambda_a', default=0.1, type=float, help='Regularization constant a.')
    parser.add_argument('--lambda_w', default=0, type=float, help='Regularization constant w.')
    parser.add_argument('--device', default=0, type=int, help='Selected CUDA device.')
    args = parser.parse_args()

    begin_date = datetime.date(2015, 1, 1) # debatenet begin date
    print('Load training data...')
    train_dataset = TemporalClassificationDataset(args.data_name, args.data_dir, 'train', begin_date, label_field='tag',
                                                 time_field='date', lm_model='bert-base-german-cased')
    # eval_dataset = TemporalClassificatonDataset()
    print('Load test data...')
    test_dataset = TemporalClassificationDataset(args.data_name, args.data_dir, 'test', begin_date, label_field='tag',
                                                 time_field='date', lm_model='bert-base-german-cased')

    print('Lambda a: {:.0e}'.format(args.lambda_a))
    print('Lambda w: {:.0e}'.format(args.lambda_w))
    print('Number of time units: {}'.format(train_dataset.n_times))
    print('Number of vocabulary items: {}'.format(len(train_dataset.filter_tensor)))

    collator = TemporalClassificationCollator()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collator)#, shuffle=True)
    # dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collator)

    nr_classes = 2 #TODO

    filename = 'dcwe_{}'.format(args.data_name)

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    model = TemporalClassificationModel(
        n_times=train_dataset.n_times,
        nr_classes=nr_classes
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    model = model.to(device)
    vocab_filter = train_dataset.filter_tensor.to(device)

    best_result = get_best('{}/{}.txt'.format(args.results_dir, filename), metric='f1')
    if best_result:
        best_f1 = best_result[0]
    else:
        best_f1 = None
    print('Best F1 so far: {}'.format(best_f1))

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

            loss = criterion(output, labels)
            loss += args.lambda_a * torch.norm(offset_t1, dim=-1).pow(2).mean()
            loss += args.lambda_w * torch.norm(offset_t1 - offset_t0, dim=-1).pow(2).mean()

            loss.backward()

            optimizer.step()

        model.eval()
        # NO DEV DATA SO FAR
        # print('Evaluate model...')

        # y_true = list()
        # y_pred = list()
        #
        # with torch.no_grad():
        #
        #     for batch in dev_loader:
        #
        #         labels, times, years, months, days, reviews, masks, segs = batch
        #
        #         labels = labels.to(device)
        #         reviews = reviews.to(device)
        #         masks = masks.to(device)
        #         segs = segs.to(device)
        #
        #         offset_t0, offset_t1, output = model(reviews, masks, segs, times, vocab_filter)
        #
        #         y_true.extend(labels.tolist())
        #         y_pred.extend(torch.round(output).tolist())
        #
        # f1_dev = f1_score(y_true, y_pred, average='macro')
        f1_dev = 0.0

        print('Test model...')

        y_true = list()
        y_pred = list()

        with torch.no_grad():

            for batch in test_loader:

                labels, users, times, years, months, days, reviews, masks, segs = batch

                labels = labels.to(device)
                reviews = reviews.to(device)
                masks = masks.to(device)
                segs = segs.to(device)

                offset_t0, offset_t1, output = model(reviews, masks, segs, times, vocab_filter)

                y_true.extend(labels.tolist())
                y_pred.extend(torch.round(output).tolist())

        f1_test = f1_score(y_true, y_pred, average='macro')

        print(f1_dev, f1_test)

        with open('{}/{}.txt'.format(args.results_dir, filename), 'a+') as f:
            f.write('{}\t{}\t{:.0e}\t{:.0e}\t{:.0e}\n'.format(f1_dev, f1_test, args.lr, args.lambda_a, args.lambda_w))

        # if best_f1 is None or f1_dev > best_f1:
        #
        #     best_f1 = f1_dev
        #     torch.save(model.state_dict(), '{}/{}.torch'.format(args.trained_dir, filename))


if __name__ == '__main__':

    start_time = time.time()

    main()

    print('---------- {:.1f} minutes ----------'.format((time.time() - start_time) / 60))
    print()
