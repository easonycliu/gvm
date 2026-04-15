import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Global Configuration
FONT_SIZE = 15
FIGURE_SIZE = (14, 2.8)
# Canonical ordering - methods not present in data are automatically skipped
ALL_METHODS_ORDER = ['MIG', 'TGS', 'GPreempt', 'xsched', 'GVM', 'GVMDYN', 'GVMCOOP', 'exclusive']
HATCH_PATTERNS = ['/', '\\', 'x', 'o', '+', '.']
STUCK_SENTINEL = 'STUCK'
VLLM_SPACING_REDUCTION = 0.65  # Reduce vLLM subplot gaps to 65% of original
MIN_SUBPLOT_GAP = 0.015  # Minimum gap to avoid text overlap
SHOW_LEGEND = True

# Color Configuration
METHOD_COLORS = {
    'MIG': 'tab:cyan',
    'TGS': 'tab:blue',
    'GPreempt': 'tab:orange',
    'xsched': 'tab:green',
    'GVM': 'tab:purple',
    'GVMDYN': 'tab:pink',
    'GVMCOOP': 'tab:brown',
    'exclusive': 'tab:red'
}
EXCLUSIVE_LINE_COLOR = 'tab:red'
EXCLUSIVE_LINE_STYLE = '--'
EXCLUSIVE_LINE_WIDTH = 3

# vLLM metrics: (source_key, display_label, scale_factor, use_log_scale)
VLLM_METRICS = [
    ('Median TTFT (ms)', 'Median TTFT (ms)', 1.0, True),
    ('P99 TTFT (ms)', 'P99 TTFT (ms)', 1.0, True),
    ('Median ITL (ms)', 'Median ITL (ms)', 1.0, False),
    ('P99 ITL (ms)', 'P99 ITL (ms)', 1.0, False),
]

def _is_stuck(value):
    """Check if a value represents a stuck result."""
    return isinstance(value, str) and value.strip().lower() == 'stuck'

def calculate_metric(application, filepath):
    if application == "vllm":
        with open(filepath) as f:
            data = json.load(f)
        if _is_stuck(data.get("median_ttft_ms", "")):
            return STUCK_SENTINEL
        return {
            "Median TTFT (ms)": data["median_ttft_ms"],
            "P99 TTFT (ms)": data["p99_ttft_ms"],
            "Median ITL (ms)": data["median_itl_ms"],
            "P99 ITL (ms)": data["p99_itl_ms"],
        }
    elif application == "diffusion":
        latency = np.mean(np.loadtxt(filepath))
        tput = 1 / latency
        return {"latency": latency, "tput": tput}
    elif application == "llama_factory":
        with open(filepath) as f:
            content = f.read().strip()
        if content.lower() == 'stuck':
            return STUCK_SENTINEL
        latency = float(content)
        tput = 1 / latency
        return {"latency": latency, "tput": tput}
    else:
        raise NotImplementedError(f"Application: {application} is not supported")

def load_data(input_path):
    dir_name = os.path.basename(os.path.normpath(input_path))
    # Strip suffixes like .vl from application names for matching filenames
    applications = [app.split(".")[0] for app in dir_name.split("+")]
    results = {application: {} for application in applications}
    detected_methods = set()

    for f in os.listdir(input_path):
        application = f.split("-")[0]
        method = "-".join(f.split("-")[1:2])
        results[application][method] = calculate_metric(application, os.path.join(input_path, f))
        detected_methods.add(method)

    # Filter and order methods based on what's actually in the data
    methods_order = [m for m in ALL_METHODS_ORDER if m in detected_methods]
    return results, applications, methods_order, dir_name

def get_color_map():
    """Build color map from explicit color configuration."""
    return {method: METHOD_COLORS[method] for method in ALL_METHODS_ORDER}

def create_bar_plot(ax, values, method_names, color_map, hatch_map):
    """Create bar plot, rendering STUCK placeholders for stuck values."""
    plot_values = [0 if v == STUCK_SENTINEL else v for v in values]
    bars = ax.bar(method_names, plot_values, color='white', edgecolor='black', linewidth=1.5)
    stuck_xs = []
    for bar, method, val in zip(bars, method_names, values):
        if val == STUCK_SENTINEL:
            bar.set_visible(False)
            stuck_xs.append(bar.get_x() + bar.get_width() / 2)
        else:
            bar.set_hatch(hatch_map[method])
            bar.set_edgecolor(color_map[method])
            bar.set_linewidth(2.5)
    return bars, stuck_xs

def add_stuck_labels(ax, stuck_xs):
    """Add STUCK text at the bottom of the axis for stuck data points."""
    if not stuck_xs:
        return
    y_bottom = ax.get_ylim()[0]
    for x in stuck_xs:
        ax.text(x, y_bottom, 'STUCK',
                ha='center', va='bottom', fontsize=FONT_SIZE - 4,
                rotation=90, color='gray', fontweight='bold')

def add_exclusive_line(ax, exclusive_val):
    """Add exclusive baseline as horizontal line."""
    if exclusive_val is not None:
        ax.axhline(y=exclusive_val, color=EXCLUSIVE_LINE_COLOR,
                  linestyle=EXCLUSIVE_LINE_STYLE, linewidth=EXCLUSIVE_LINE_WIDTH)

def reduce_vllm_spacing(fig, vllm_axes, diff_x0):
    """Reduce horizontal spacing between vLLM subplots."""
    first_pos = vllm_axes[0].get_position()
    first_x0 = first_pos.x0
    subplot_width = first_pos.x1 - first_pos.x0
    available_width = diff_x0 - first_x0
    n = len(vllm_axes)
    gap_width = max((available_width - n * subplot_width) / max(n - 1, 1) * VLLM_SPACING_REDUCTION, MIN_SUBPLOT_GAP)

    current_x = first_x0
    for ax in vllm_axes:
        pos = ax.get_position()
        ax.set_position([current_x, pos.y0, subplot_width, pos.y1 - pos.y0])
        current_x += subplot_width + gap_width

def main():
    parser = argparse.ArgumentParser(description="Script for drawing performance bar chart")
    parser.add_argument("--input", type=str, required=True, help="Path to input data")
    parser.add_argument("--output", type=str, required=True, help="Path to output fig")
    args = parser.parse_args()

    # Load data
    results, applications, methods_order, dir_name = load_data(args.input)
    print(results)
    print(f"Detected methods: {methods_order}")

    # Setup colors and patterns
    color_map = get_color_map()
    bar_methods = [m for m in methods_order if m != 'exclusive']
    hatch_map = {method: HATCH_PATTERNS[i % len(HATCH_PATTERNS)] for i, method in enumerate(bar_methods)}

    # Setup matplotlib with Type 1 fonts for OSDI submission
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'legend.fontsize': FONT_SIZE,
        # Type 1 font configuration for OSDI submission
        'pdf.fonttype': 42,  # Embed fonts as Type 1
        'ps.fonttype': 42,   # Embed fonts as Type 1
        'font.family': 'sans-serif',  # Use sans-serif fonts (Type 1 compatible)
    })

    # Create figure
    fig = plt.figure(figsize=FIGURE_SIZE)

    # Application labels
    def format_app_label(app):
        app_lower = app.lower()
        if app_lower == 'vllm':
            return 'vLLM'
        elif app_lower == 'diffusion':
            return 'Diffusion'
        elif app_lower == 'llama_factory':
            return 'LlamaFactory'
        else:
            return app.capitalize()

    app0_label = format_app_label(applications[0])
    app1_label = format_app_label(applications[1])

    # Plot vLLM metrics (subplots 1-N)
    n_vllm = len(VLLM_METRICS)
    n_total = n_vllm + 1
    for i, (src_key, display_label, scale, use_log) in enumerate(VLLM_METRICS, start=1):
        ax = fig.add_subplot(1, n_total, i)
        actual_src_key = 'xsched' if src_key == 'XSched' else src_key

        values = [
            STUCK_SENTINEL if results['vllm'][method] == STUCK_SENTINEL
            else results['vllm'][method][actual_src_key] * scale
            for method in bar_methods
        ]
        bars, stuck_xs = create_bar_plot(ax, values, bar_methods, color_map, hatch_map)

        # Add exclusive line
        exclusive_val = (results['vllm']['exclusive'][actual_src_key] * scale
                        if 'exclusive' in results['vllm'] else None)
        add_exclusive_line(ax, exclusive_val)

        # Collect numeric values for axis scaling
        numeric_values = [v for v in values if v != STUCK_SENTINEL]
        all_numeric = numeric_values + ([exclusive_val] if exclusive_val is not None else [])

        # Configure axis
        if use_log:
            ax.set_yscale('log')
            if all_numeric:
                min_val = min(all_numeric)
                ax.set_ylim(bottom=min_val * 0.1)
        else:
            ax.set_ylim(bottom=0)
            if len(all_numeric) >= 3:
                second_max_val = sorted(all_numeric)[-3]
            #ax.set_ylim(top=second_max_val * 1.5)
            #for bar, value in zip(bars, values):
            #    if value > second_max_val * 1.5:
            #        ax.text(bar.get_x() + bar.get_width() / 2, second_max_val * 1.5,
            #               f'{value:.0f}', ha='center', va='bottom',
            #               fontsize=FONT_SIZE - 5, color='tab:red')
        ax.set_ylabel(display_label)
        ax.set_xticks([])
        add_stuck_labels(ax, stuck_xs)

    # Plot throughput metric - separate group with spacing
    ax = fig.add_subplot(1, n_total, n_total)
    latencies = [
        STUCK_SENTINEL if results[applications[1]][method] == STUCK_SENTINEL
        else results[applications[1]][method]['latency']
        for method in bar_methods
    ]
    exclusive_latency = (results[applications[1]]['exclusive']['latency']
                        if 'exclusive' in results[applications[1]] else None)

    if exclusive_latency is not None:
        norm_throughput = [
            STUCK_SENTINEL if L == STUCK_SENTINEL else exclusive_latency / L
            for L in latencies
        ]
    else:
        norm_throughput = [
            STUCK_SENTINEL if L == STUCK_SENTINEL else 1
            for L in latencies
        ]

    _, stuck_xs = create_bar_plot(ax, norm_throughput, bar_methods, color_map, hatch_map)
    numeric_tput = [v for v in norm_throughput if v != STUCK_SENTINEL]
    ax.set_ylim((0, max(numeric_tput) * 1.5 if numeric_tput else 1))
    ax.set_ylabel("Norm. Tput. (x)")
    add_stuck_labels(ax, stuck_xs)
    ax.set_xticks([])

    # Adjust layout and spacing
    fig.tight_layout(rect=[0, 0.20, 1, 1])

    # Reduce horizontal spacing between vLLM subplots
    vllm_axes = [fig.axes[i] for i in range(n_vllm)]
    ax_diff = fig.axes[n_vllm]
    reduce_vllm_spacing(fig, vllm_axes, ax_diff.get_position().x0)

    # Add visual separator between groups
    ax_vllm_first, ax_vllm_last = fig.axes[0], fig.axes[n_vllm - 1]
    vllm_last_pos = ax_vllm_last.get_position()
    separator_x = (vllm_last_pos.x1 + ax_diff.get_position().x0) / 2 - 0.025
    fig.add_artist(Line2D([separator_x, separator_x],
                         [vllm_last_pos.y0 - 0.15, vllm_last_pos.y1 + 0.03],
                         color='gray', linewidth=2, linestyle='--', alpha=0.6,
                         transform=fig.transFigure, zorder=0))

    # Add subfigure captions
    vllm_center_x = (ax_vllm_first.get_position().x0 + ax_vllm_last.get_position().x1) / 2
    diff_center_x = (ax_diff.get_position().x0 + ax_diff.get_position().x1) / 2 - (0.03 if app1_label == 'LlamaFactory' else 0.025)
    fig.text(vllm_center_x, 0.15, f"{app0_label} (↓ is better)",
             fontsize=FONT_SIZE, ha='center', va='bottom', transform=fig.transFigure)
    fig.text(diff_center_x, 0.15, f"{app1_label} (↑ is better)",
             fontsize=FONT_SIZE, ha='center', va='bottom', transform=fig.transFigure)

    # Add legend - match bar appearance: white fill, colored edge, hatch pattern
    legend_handles = [
        mpatches.Patch(facecolor='white', edgecolor=color_map[method],
                      hatch=hatch_map[method], linewidth=2.5, label=method)
        for method in bar_methods
    ]
    legend_handles.append(Line2D([], [], color=EXCLUSIVE_LINE_COLOR,
                                linestyle=EXCLUSIVE_LINE_STYLE,
                                linewidth=EXCLUSIVE_LINE_WIDTH, label='Exclusive'))

    if SHOW_LEGEND:
        fig.legend(handles=legend_handles, loc=(0.08, 0.88),
                   ncol=len(legend_handles), frameon=False)

    # Save plot
    filename = os.path.join(args.output, f"{dir_name}_performance_bar.pdf")
    fig.savefig(filename, bbox_inches='tight')

if __name__ == "__main__":
    main()
