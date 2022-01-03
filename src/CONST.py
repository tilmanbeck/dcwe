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

id_field_map = {
    'debate': 'tweet_id',
    'sandy': 'tweet_id',
    'rumours':  'id',
    'clex': 'tweet_id'
}

metrics_for_datasets = {
    'debate': 'f1_binary',
    'sandy': 'f1_binary',
    'rumours':  'f1_micro',
    'clex': 'f1_micro'
}