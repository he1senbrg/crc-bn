import pandas as pd
import matplotlib.pyplot as plt


def calibration_quantile_bin(y_true, y_prob, n_bins=10):
    dfc = pd.DataFrame({"y": y_true, "p": y_prob}).dropna()
    dfc["bin"] = pd.qcut(dfc["p"], q=n_bins, duplicates="drop")
    cal = (
        dfc.groupby("bin")
        .agg(mean_p=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
        .reset_index()
    )
    return cal


def plot_calibration(cal):
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(cal["mean_p"], cal["obs"], marker="o")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration (quantile binning)")
    plt.show()
    return cal
