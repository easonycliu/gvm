import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Script for drawing pareto fig")

parser.add_argument("--input", type=str, required=True, help="Path to input data")
parser.add_argument("--output", type=str, required=True, help="Path to output fig")
parser.add_argument("--slo", type=float, required=True, help="SLO for counting SLO attainment")

def calculate_metric(application, filepath, slo):
	if application == "vllm":
		total_tokens = 0
		slo_satisfied_tokens = 0
		fileobj = open(filepath)
		data = json.load(fileobj)
		fileobj.close()

		average_output_tokens = data["total_output_tokens"] // data["completed"]

		for itl in data["itls"]:
			if len(itl) == 0:
				total_tokens += average_output_tokens
				continue

			for latency in itl:
				if latency <= slo:
					slo_satisfied_tokens += 1
				total_tokens += 1

		return slo_satisfied_tokens / np.max([total_tokens, 1])
	elif application == "diffusion":
		return 1 / np.mean(np.loadtxt(filepath))
	else:
		raise NotImplementedError("Application: {} is not supported".format(application))


if __name__ == "__main__":
	args = parser.parse_args()

	applications = os.path.basename(os.path.normpath(args.input)).split("+")
	results = { application : {} for application in applications }

	for file in os.listdir(args.input):
		application = file.split("-")[0]
		method = "-".join(file.split("-")[1:-2])
		results[application][method] = calculate_metric(application, os.path.join(args.input, file), args.slo)

	print(results)
	# Extract common keys (methods)
	keys = sorted(results["vllm"].keys())
	
	x = [results["vllm"][k] for k in keys]         # vllm performance (x-axis)
	y = [results["diffusion"][k] for k in keys]    # diffusion performance (y-axis)
	
	plt.figure(figsize=(16, 16))
	plt.scatter(x, y, color='dodgerblue', s=150)
	
	# Label each point
	for i, k in enumerate(keys):
	    plt.text(x[i] + 0.002, y[i] + 0.002, k, fontsize=20)
	
	plt.xlabel("vLLM SLO attainment (%)")
	plt.ylabel("Diffusion throughput (req/s)")
	plt.title("vLLM + Diffusion")
	plt.grid(True, linestyle='--', alpha=0.5)
	plt.tight_layout()

	plt.savefig(os.path.join(args.output, "{}_pareto.png".format("+".join(applications))))
