"""Plots of telemetry, true anomalies and detector scores."""

import matplotlib

matplotlib.use("Agg")  # render to files; works without a display
import matplotlib.pyplot as plt  # noqa: E402

from spacecraft_ad.evaluate import segments  # noqa: E402


def plot_results(telemetry, labels, scores, thresholds, path):
    """Save a figure: telemetry with anomaly segments shaded, then one score panel per detector."""
    n = 1 + len(scores)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    tel = telemetry.loc[next(iter(scores.values())).index]
    lab = labels.loc[tel.index]

    for col in tel.columns:
        axes[0].plot(tel.index, tel[col], lw=0.6, label=col)
    axes[0].set_title("Telemetry (shaded = true anomalies)")
    axes[0].legend(loc="upper right", fontsize=8, ncol=len(tel.columns))

    for ax in axes:
        for start, end in segments(lab.to_numpy()):
            ax.axvspan(tel.index[start], tel.index[end - 1], color="red", alpha=0.15)

    for ax, (name, s) in zip(axes[1:], scores.items()):
        ax.plot(s.index, s, lw=0.6, color="purple")
        ax.axhline(thresholds[name], color="black", ls="--", lw=1, label="threshold")
        ax.set_title(f"{name} anomaly score")
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
