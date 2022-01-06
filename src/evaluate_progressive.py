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
	result_dirs = os.listdir(args.dir)
	eval_results = []
	test_results = []

	for d in result_dirs:
		with open(os.path.join(args.dir, d, 'eval_results.json')) as fp:
			eval_result = json.load(fp)
			eval_results.append(eval_result['eval_' + metrics_for_datasets[args.data_name]])
		with open(os.path.join(args.dir, d, 'test_results.json')) as fp:
			test_result = json.load(fp)
			test_results.append(test_result['test_' + metrics_for_datasets[args.data_name]])
	
	print('Results for: ', args.data_name)
	print('Directory: ', args.dir)
	print('Metric: ', metrics_for_datasets[args.data_name])
	print('eval mean (std): {:.4f} ({:.4f})'.format(np.mean(eval_results), np.std(eval_results)))
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
	evaluate_seed_results()
