import numpy as np
import json
import os
import argparse
from CONST import metrics_for_datasets
import matplotlib.pyplot as plt

def extract_progressive_results_by_seed(dir):
	result_dir = [i for i in os.listdir(dir) if os.path.isdir(os.path.join(dir, i))]
	results = {}

	# collecting all the data
	for seed in result_dir:
		results[seed] = {}
		for dev_bin in os.listdir(os.path.join(dir, seed)):
			if dev_bin not in results:
				results[seed][dev_bin] = {}
			for test_prediction in os.listdir(os.path.join(dir, seed, dev_bin)):
				# select only the test prediction files which are json and start with numbers 0-9
				if not (test_prediction.endswith('.json') and test_prediction.split('_')[0] in list([str(i) for i in range(0,10)])):
					continue
				path = os.path.join(dir, seed, dev_bin, test_prediction)
				with open(path) as fp:
					test_result = json.load(fp)
					results[seed][dev_bin][test_prediction] = test_result['test_' + metrics_for_datasets[args.data_name]]
	return results


def extract_progressive_results(dir):
	result_dir = [i for i in os.listdir(dir) if os.path.isdir(os.path.join(dir, i))]
	results = {}

	# collecting all the data
	for seed in result_dir:
		for dev_bin in os.listdir(os.path.join(dir, seed)):
			if dev_bin not in results:
				results[dev_bin] = {}
			for test_prediction in os.listdir(os.path.join(dir, seed, dev_bin)):
				# select only the test prediction files which are json and start with numbers 0-9
				if not (test_prediction.endswith('.json') and test_prediction.split('_')[0] in list([str(i) for i in range(0,10)])):
					continue
				if test_prediction not in results[dev_bin]:
					results[dev_bin][test_prediction] = []
				path = os.path.join(dir, seed, dev_bin, test_prediction)
				with open(path) as fp:
					test_result = json.load(fp)
					results[dev_bin][test_prediction].append(test_result['test_' + metrics_for_datasets[args.data_name]])
	return results


def evaluate_progressive(data_name, dir,):
	results = extract_progressive_results_by_seed(dir)

	# average over test bins
	for seed in results.keys():
		for dev_bin, res in results[seed].items():
			results[seed][dev_bin] = np.mean(list(res.values()))
	# average over dev bins
	for seed in results.keys():
		test_res = list(results[seed].values())
		results[seed] = np.mean(test_res)

	print('Results for: ', data_name)
	print('Directory: ', dir)
	print('Metric: ', metrics_for_datasets[data_name])
	print('test mean (std): {:.4f} ({:.4f})'.format(np.mean(list(results.values())), np.std(list(results.values()))))
	# for idx_i, (dev_bin, test_preds) in enumerate(sorted(results.items())):
	# 	train_bins = [i for i in range(0, int(dev_bin.split('_')[-1]))]
	# 	print('Training bins:', train_bins)
	# 	for idx_j, (k,test_results) in enumerate(sorted(test_preds.items())):
	# 		test_bin = k.split('_')[0]
	# 		print('Test bin: ', test_bin)
	# 		print('test mean (std): {:.4f} ({:.4f})'.format(np.mean(test_results), np.std(test_results)))

def prepare_data_progressive_heatmap(data_name, dir):
	results = extract_progressive_results(dir)

	print('Results for: ', data_name)
	print('Directory: ', dir)
	print('Metric: ', metrics_for_datasets[data_name])
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
			# print(idx_i, idx_j, np.mean(test_results))
			# offsetting here to account for the decreasing length of test bins (e.g. when trained with [0,1,2,3] we only
			# have [5,6,7,8,9] as test bins whereas in the beginning we have [2,3,4,5,6,7,8,9] as test bins
			plot_data[idx_i, idx_j + (plot_data.shape[1] - len(test_preds))] = np.mean(test_results)
	assert plot_data.shape[0] == len(teb)
	assert plot_data.shape[1] == len(trb)
	# print(teb)
	# print('---')
	# print(trb)
	# print('---')
	print(plot_data)
	plot_progressive_heatmap(teb, trb, plot_data, data_name, dir)

def plot_progressive_heatmap(test_bins, train_bins, data, data_name, output_dir):
	# test_bins = [2,3,4,5,6,7,8,9]
	# train_bins = [[0], [0,1], [0,1,2] ..., [0, 1, 2, 3, 4, 5, 6, 7]]
	# data is supposed to be in shape (|test_bins|, |train_bins|)
	# with data[0,0] representing data for test_bin=2 and train_bin [0]
	# and data[2,3] representing data for test_bin=4 and train_bin [0,1,2]
	fig, ax = plt.subplots()
	im = ax.imshow(data)

	# Show all ticks and label them with the respective list entries
	ax.set_yticks(np.arange(len(train_bins)))
	ax.set_yticklabels(train_bins)
	ax.set_xticks(np.arange(len(test_bins)))
	ax.set_xticklabels(test_bins)

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	for i in range(len(test_bins)):
		for j in range(len(train_bins)):
			# switch the coordinates here to account for the reverse layout of plt's heatmap
			text = ax.text(j, i, '{:.1f}'.format(data[i, j]* 100),
						   ha="center", va="center", color="w")

	ax.set_title("DCWE Progressive Performance (" + data_name + ')')
	fig.tight_layout()
	plt.savefig(os.path.join(output_dir, 'progressive_' + data_name + '.png'))
	plt.show()

def evaluate_seed_results(data_name, dir):

	eval_results = []
	test_results = []

	for seed in os.listdir(dir):
		seed_dir = os.path.join(dir, seed)
		if os.path.exists(os.path.join(seed_dir, 'eval_results.json')):
			eval_dir = os.path.join(seed_dir, 'eval_results.json')
		else:
			eval_dir = os.path.join(seed_dir, 'results.json')
		with open(eval_dir) as fp:
			eval_result = json.load(fp)
			eval_results.append(eval_result['eval_' + metrics_for_datasets[data_name]])
		with open(os.path.join(seed_dir, 'test_results.json')) as fp:
			test_result = json.load(fp)
			test_results.append(test_result['test_' + metrics_for_datasets[data_name]])

	print('Results for: ', data_name)
	print('Directory: ', dir)
	print('Seeds: ', os.listdir(dir))
	print('Metric: ', metrics_for_datasets[data_name])
	print('eval mean (std): {:.4f} ({:.4f})'.format(np.mean(eval_results), np.std(eval_results)))
	print('test mean (std): {:.4f} ({:.4f})'.format(np.mean(test_results), np.std(test_results)))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_name", type=str, required=True)
	parser.add_argument("--dir", type=str, required=True)
	args = parser.parse_args()
	data_name = args.data_name
	dir = args.dir
	evaluate_seed_results(data_name, dir)
	# prepare_data_progressive_heatmap(data_name, dir)
	# evaluate_progressive(data_name, dir)
	# extract_progressive_results_by_seed(dir)
