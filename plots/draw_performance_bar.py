import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Script for drawing pareto fig")

parser.add_argument("--input", type=str, required=True, help="Path to input data")
parser.add_argument("--output", type=str, required=True, help="Path to output fig")

def calculate_metric(application, filepath):
	if application == "vllm":
		total_tokens = 0
		fileobj = open(filepath)
		data = json.load(fileobj)
		fileobj.close()

		return {
			"Median TTFT (ms)" : data["median_ttft_ms"],
			"p99 TTFT (ms)" : data["p99_ttft_ms"],
			"Median ITL (ms)" : data["median_itl_ms"],
			"p99 ITL (ms)" : data["p99_itl_ms"],
		}
	elif application == "diffusion":
		return { "latency" : np.mean(np.loadtxt(filepath)) }
	elif application == "llama_factory":
		return { "latency" : np.loadtxt(filepath) }
	else:
		raise NotImplementedError("Application: {} is not supported".format(application))


if __name__ == "__main__":
	args = parser.parse_args()

	applications = os.path.basename(os.path.normpath(args.input)).split("+")
	results = { application : {} for application in applications }

	for file in os.listdir(args.input):
		application = file.split("-")[0]
		method = "-".join(file.split("-")[1:2])
		results[application][method] = calculate_metric(application, os.path.join(args.input, file))

	print(results)

	# Set plotting order
	methods_preferred_order = ['TGS', 'GPreempt', 'xsched', "GVM", 'exclusive']
	methods_colors = ["blue", "blue", "blue", "blue", "red"]
	vllm_metrics = ['Median TTFT (ms)', 'p99 TTFT (ms)', 'Median ITL (ms)', 'p99 ITL (ms)']
	
	# Create a figure with 5 subplots (5 rows, 1 column)
	plt.figure(figsize=(40, 8))
	
	# 1–4: vllm metrics
	for i, metric in enumerate(vllm_metrics, start=1):
	    plt.subplot(1, 5, i)
	    values = [results['vllm'][m][metric] for m in methods_preferred_order]
	    plt.bar(methods_preferred_order, values, color=methods_colors)
	    plt.title("{} — {}".format(applications[0], metric))
	    plt.ylabel(metric)
	    plt.tight_layout()
	
	# 5: Low priority latency
	plt.subplot(1, 5, 5)
	diff_vals = [results[applications[1]][m]['latency'] for m in methods_preferred_order]
	plt.bar(methods_preferred_order, diff_vals, color=methods_colors)
	plt.title("{} — Latency (s)".format(applications[1]))
	plt.ylabel("Latency (s)")
	plt.tight_layout()

	plt.savefig(os.path.join(args.output, "{}_performance_bar.png".format("+".join(applications))))
