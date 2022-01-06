import numpy as np
import json
import os
import argparse
from CONST import metrics_for_datasets


def evaluate_progressive():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_name", type=str, required=True)
	parser.add_argument("--dir", type=str, required=True)
	args = parser.parse_args()
	result_dir = os.listdir(args.dir)
	results = {

	}

	for seed in result_dir:
		for dev_bin in os.listdir(os.path.join(args.dir, seed)):
			if dev_bin not in results:
				results[dev_bin] = {}
			for test_prediction in os.listdir(os.path.join(args.dir, seed, dev_bin)):
				# select only the test prediction files which are json and start with numbers 0-9
				if not (test_prediction.endswith('.json') and test_prediction.split('_')[0] in list([str(i) for i in range(0,10)])):
					continue
				if test_prediction not in results[dev_bin]:
					results[dev_bin][test_prediction] = []
				path = os.path.join(args.dir, seed, dev_bin, test_prediction)
				with open(path) as fp:
					test_result = json.load(fp)
					results[dev_bin][test_prediction].append(test_result['test_' + metrics_for_datasets[args.data_name]])

	print('Results for: ', args.data_name)
	print('Directory: ', args.dir)
	print('Metric: ', metrics_for_datasets[args.data_name])
	for dev_bin, test_preds in sorted(results.items()):
		train_bins = [i for i in range(0, int(dev_bin.split('_')[-1]))]
		print('Training bins:', train_bins)
		for k,test_results in sorted(test_preds.items()):
			print('Test bin: ', k.split('_')[0])
			print('test mean (std): {:.4f} ({:.4f})'.format(np.mean(test_results), np.std(test_results)))


def evaluate_seed_results():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_name", type=str, required=True)
	parser.add_argument("--dir", type=str, required=True)
	args = parser.parse_args()

	eval_results = []
	test_results = []

	for seed in os.listdir(args.dir):
		seed_dir = os.path.join(args.dir, seed)
		with open(os.path.join(seed_dir, 'eval_results.json')) as fp:
			eval_result = json.load(fp)
			eval_results.append(eval_result['eval_' + metrics_for_datasets[args.data_name]])
		with open(os.path.join(seed_dir, 'test_results.json')) as fp:
			test_result = json.load(fp)
			test_results.append(test_result['test_' + metrics_for_datasets[args.data_name]])

	print('Results for: ', args.data_name)
	print('Directory: ', args.dir)
	print('Seeds: ', os.listdir(args.dir))
	print('Metric: ', metrics_for_datasets[args.data_name])
	print('eval mean (std): {:.4f} ({:.4f})'.format(np.mean(eval_results), np.std(eval_results)))
	print('test mean (std): {:.4f} ({:.4f})'.format(np.mean(test_results), np.std(test_results)))


if __name__ == '__main__':
	#evaluate_seed_results()
	evaluate_progressive()
