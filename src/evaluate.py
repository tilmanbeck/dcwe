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
	teb = [] # test bins
	trb = [] # train bins
	tmp = sorted(results.items())[0][1]
	plot_data = np.zeros((len(tmp), len(results.keys())))
	for idx_i, (dev_bin, test_preds) in enumerate(sorted(results.items())):
		train_bins = [i for i in range(0, int(dev_bin.split('_')[-1]))]
		if str(train_bins) not in trb:
			trb.append(str(train_bins))
		print('Training bins:', train_bins)
		for idx_j, (k,test_results) in enumerate(sorted(test_preds.items())):
			test_bin = k.split('_')[0]
			if test_bin not in teb:
				teb.append(test_bin)
			print('Test bin: ', test_bin)
			print('test mean (std): {:.4f} ({:.4f})'.format(np.mean(test_results), np.std(test_results)))
			plot_data[idx_i, idx_j] = np.mean(test_results)
	assert plot_data.shape[0] == len(teb)
	assert plot_data.shape[1] == len(trb)
	print(teb)
	print('---')
	print(trb)
	print('---')
	print(plot_data)


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
