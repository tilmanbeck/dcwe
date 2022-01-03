import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment','raise')
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
from transformers import EvalPrediction
from sklearn.metrics import f1_score


def convert_labels(data, label_map):
    data['label'] = [label_map[i] for i in data['label']]
    return data

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    f1_bin = f1_score(p.label_ids, preds, average='binary', pos_label=1) if len(set(p.label_ids))==2 else None
    result = {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item(),
              'f1_binary': f1_bin,
              'f1_macro': f1_score(p.label_ids, preds, average='macro'),
              'f1_micro': f1_score(p.label_ids, preds, average='micro')}
    return result


def convert_times(time, name, begin_date=None, temporal_granularity='day'):

    if name == 'arxiv':
        return time - 2001

    elif name == 'ciao':
        return time - 2000

    elif name == 'yelp':
        return time - 2010

    elif name == 'reddit':
        return time - 9
    else:
        if temporal_granularity == 'day':
            return abs(time.date() - begin_date).days
        elif temporal_granularity == 'week':
            return abs(time.date() - begin_date).weeks
        elif temporal_granularity == 'month':
            return abs(time.date() - begin_date).months
        else: # default
            return abs(time.date() - begin_date).days

def truncate(reviews):

    truncated = list()

    for r in reviews:
        if len(r) > 512:
            r = r[:256] + r[-256:]
        truncated.append(r)

    return truncated


def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)

def data_stats(path, split='train', partition='time_stratified_partition'):
    time_field = 'date'
    label_field = 'tag'
    dataframe = pd.read_csv(path, parse_dates=[time_field], encoding='utf-8')
    # take the corresponding split (if it exists otherwise take random split)
    split_data = dataframe[dataframe[partition] == split]
    filtered_data = pd.DataFrame(dataframe.iloc[split_data.index])
    # rename data columns to common format
    filtered_data.rename(columns={label_field: 'label', time_field: 'time'}, inplace=True)
    # convert string labels to numeric

    filtered_data.dropna(subset=[partition, 'label', 'time'], inplace=True)
    filtered_data.time = pd.to_datetime(filtered_data.time)
    filtered_data.reset_index(inplace=True, drop=True)

    from collections import Counter
    print('Split:', split)
    print(Counter(filtered_data['label'].values.tolist()))
