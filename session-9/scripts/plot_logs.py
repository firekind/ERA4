import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def smooth_values(values, smoothing=0.6):
    """
    Apply exponential moving average smoothing.

    Args:
        values: List of values to smooth
        smoothing: Smoothing factor between 0 and 1 (higher = more smoothing)

    Returns:
        Smoothed values
    """
    smoothed = []
    last = values[0]

    for value in values:
        smoothed_val = last * smoothing + (1 - smoothing) * value
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed


def read_tensorboard_logs(log_dir, metric_name):
    """
    Read a specific metric from TensorBoard log directory.

    Args:
        log_dir: Path to TensorBoard log directory
        metric_name: Name of the metric to extract

    Returns:
        steps: List of step values
        values: List of metric values
    """
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()

    # Check if the metric exists
    if metric_name not in ea.Tags()["scalars"]:
        return [], []

    # Get the scalar data
    events = ea.Scalars(metric_name)
    steps = [event.step for event in events]
    values = [event.value for event in events]

    return steps, values


def get_all_metrics(logs_dir):
    """
    Get all unique metric names from all runs.

    Args:
        logs_dir: Parent directory containing subdirectories for each run

    Returns:
        Set of all metric names (excluding hp_metric)
    """
    logs_dir = Path(logs_dir)
    run_dirs = [d for d in logs_dir.iterdir() if d.is_dir()]

    all_metrics = set()
    for run_dir in run_dirs:
        ea = event_accumulator.EventAccumulator(str(run_dir))
        ea.Reload()
        all_metrics.update(ea.Tags()["scalars"])

    # Remove hp_metric
    all_metrics.discard("hp_metric")

    return sorted(all_metrics)


def plot_single_metric(ax, logs_dir, metric_name, smoothing):
    """
    Plot a single metric on the given axis.

    Args:
        ax: Matplotlib axis to plot on
        logs_dir: Parent directory containing subdirectories for each run
        metric_name: Name of the metric to plot
        smoothing: Smoothing factor
    """
    run_dirs = sorted([d for d in logs_dir.iterdir() if d.is_dir()])

    for run_dir in run_dirs:
        steps, values = read_tensorboard_logs(run_dir, metric_name)

        if steps:
            label = run_dir.name
            if smoothing > 0:
                values = smooth_values(values, smoothing)
            ax.plot(steps, values, label=label, linewidth=2)

    ax.set_xlabel("Step", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)
    ax.set_title(metric_name, fontsize=11)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_tensorboard_runs(logs_dir, metric_name, output_path, smoothing=0.6):
    """
    Plot metrics from multiple TensorBoard runs.

    Args:
        logs_dir: Parent directory containing subdirectories for each run
        metric_name: Name of the metric to plot (None for all metrics)
        output_path: Path to save the output image
        smoothing: Smoothing factor (0 = no smoothing, 0.99 = heavy smoothing)
    """
    logs_dir = Path(logs_dir)
    run_dirs = sorted([d for d in logs_dir.iterdir() if d.is_dir()])

    if not run_dirs:
        print(f"No subdirectories found in {logs_dir}")
        return

    # If no metric specified, plot all metrics
    if metric_name is None:
        metrics = get_all_metrics(logs_dir)

        if not metrics:
            print("No metrics found in the log directories")
            return

        print(f"Found {len(metrics)} metrics: {', '.join(metrics)}")

        # Calculate grid dimensions
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)  # Max 3 columns
        n_rows = math.ceil(n_metrics / n_cols)

        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))

        # Handle single metric case
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_metrics > 1 else [axes]

        # Plot each metric
        for i, metric in enumerate(metrics):
            plot_single_metric(axes[i], logs_dir, metric, smoothing)

        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle("Training Runs Comparison - All Metrics", fontsize=16, y=0.995)
        plt.tight_layout()
    else:
        # Single metric plot
        plt.figure(figsize=(12, 6))

        for run_dir in run_dirs:
            steps, values = read_tensorboard_logs(run_dir, metric_name)

            if steps:
                label = run_dir.name
                if smoothing > 0:
                    values = smooth_values(values, smoothing)
                plt.plot(steps, values, label=label, linewidth=2)
            else:
                print(f"No data found for metric '{metric_name}' in {run_dir.name}")

        plt.xlabel("Step", fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.title(f"{metric_name} - Training Runs Comparison", fontsize=14)
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    # Save the plot
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot TensorBoard metrics from multiple runs"
    )
    parser.add_argument(
        "logs_dir", type=str, help="Directory containing run subdirectories"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Name of the metric to plot (omit to plot all metrics)",
    )
    parser.add_argument("output", type=str, help="Output image path (e.g., plot.png)")
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.6,
        help="Smoothing factor (0-0.99, default: 0.6)",
    )

    args = parser.parse_args()
    plot_tensorboard_runs(args.logs_dir, args.metric, args.output, args.smoothing)
