import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

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

	# Set plotting order and colors (colorblind-friendly palette)
	methods_preferred_order = ['TGS', 'GPreempt', 'xsched', "GVM", 'exclusive']
	palette = plt.get_cmap('tab10').colors
	methods_color_map = { m: palette[i % len(palette)] for i, m in enumerate(methods_preferred_order) }
	# Override GVM with tab:purple
	methods_color_map['GVM'] = 'tab:purple'
	bar_methods = [m for m in methods_preferred_order if m != 'exclusive']
	# Distinct hatch patterns for bar methods
	hatch_patterns = ['/', '\\', 'x', 'o']
	methods_hatch_map = { m: hatch_patterns[i % len(hatch_patterns)] for i, m in enumerate(bar_methods) }
	# Define mapping for display: (source_key_in_results, display_label, scale_factor)
	vllm_plot_map = [
		('Median TTFT (ms)', 'Median TTFT (s)', 0.001),
		('p99 TTFT (ms)', 'P99 TTFT (s)', 0.001),
		('Median ITL (ms)', 'Median ITL (ms)', 1.0),
		('p99 ITL (ms)', 'P99 ITL (ms)', 1.0),
	]

	# Nicely formatted application labels
	app0_label = 'vLLM' if applications[0].lower() == 'vllm' else applications[0].capitalize()
	app1_label = 'Diffusion' if applications[1].lower() == 'diffusion' else applications[1].capitalize()

	# Global font sizing
	FONT_SIZE = 15
	plt.rcParams.update({
		'font.size': FONT_SIZE,
		'axes.labelsize': FONT_SIZE,
		'xtick.labelsize': FONT_SIZE,
		'ytick.labelsize': FONT_SIZE,
		'legend.fontsize': FONT_SIZE,
	})
	# Create a figure with 5 subplots (1 row, 5 columns)
	plt.figure(figsize=(14, 3.2))

	# 1–4: vLLM metrics (TTFT in seconds, ITL in ms). Hide x-axis labels.
	for i, (src_key, display_label, scale) in enumerate(vllm_plot_map, start=1):
		plt.subplot(1, 5, i)
		if src_key == 'XSched':
			src_key = 'xsched'
		values = [results['vllm'][m][src_key] * scale for m in bar_methods]
		colors = [methods_color_map[m] for m in bar_methods]
		bars = plt.bar(bar_methods, values, color='white', edgecolor='black', linewidth=1.5)
		for rect, m in zip(bars, bar_methods):
			rect.set_hatch(methods_hatch_map[m])
			rect.set_edgecolor(methods_color_map[m])
			rect.set_linewidth(2.5)
		# Exclusive as horizontal red line
		exclusive_val = results['vllm']['exclusive'][src_key] * scale if 'exclusive' in results['vllm'] else None
		if exclusive_val is not None:
			plt.axhline(y=exclusive_val, color='tab:red', linestyle='--', linewidth=3)
		# Log scale for TTFT metrics
		if 'TTFT' in src_key:
			plt.yscale('log')
		plt.ylabel("{} {}".format(app0_label, display_label))
		plt.xticks([])

	# 5: Diffusion normalized throughput vs exclusive (hide x-axis labels)
	plt.subplot(1, 5, 5)
	latencies = [results[applications[1]][m]['latency'] for m in bar_methods]
	# Normalize throughput to exclusive: (1/L) / (1/L_exclusive) = L_exclusive / L
	exclusive_latency = results[applications[1]]['exclusive']['latency'] if 'exclusive' in results[applications[1]] else None
	if exclusive_latency is not None:
		norm_throughput = [exclusive_latency / L for L in latencies]
	else:
		norm_throughput = [1 for _ in latencies]
	colors = [methods_color_map[m] for m in bar_methods]
	bars = plt.bar(bar_methods, norm_throughput, color='white', edgecolor='black', linewidth=1.5)
	for rect, m in zip(bars, bar_methods):
		rect.set_hatch(methods_hatch_map[m])
		rect.set_edgecolor(methods_color_map[m])
		rect.set_linewidth(2.5)
	# Exclusive normalized throughput line at 1.0
	# plt.axhline(y=1.0, color='tab:red', linestyle='--', linewidth=3)
	plt.ylabel("{} Norm. Tput.".format(app1_label))
	# plt.ylim(0, 0.5)
	plt.xticks([])

	# Legend (top center) with bar methods and exclusive line
	legend_handles = [
		mpatches.Patch(facecolor=methods_color_map[m], edgecolor='black', hatch=methods_hatch_map[m], label=m)
		for m in bar_methods
	]
	legend_handles.append(Line2D([], [], color='tab:red', linestyle='--', linewidth=3, label='exclusive'))
	plt.tight_layout(rect=[0, 0.15, 1, 1])
	plt.figlegend(handles=legend_handles, loc=(0.25, .9), ncol=len(legend_handles), frameon=False)

	plt.savefig(os.path.join(args.output, "{}_performance_bar.pdf".format("+".join(applications))), bbox_inches='tight')
