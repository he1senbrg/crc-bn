import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_risk_map(df, infer):
    ages = sorted(df["Age"].unique())
    alc = sorted(df["Alcohol"].unique())
    heat = np.zeros((len(ages), len(alc)))

    for i, a in enumerate(ages):
        for j, al in enumerate(alc):
            q = infer.query(
                ["CRC"], evidence={"Age": a, "Alcohol": al}, show_progress=False
            )
            yes_idx = q.state_names["CRC"].index("Yes")
            heat[i, j] = float(q.values[yes_idx])

    plt.figure(figsize=(6, 4))
    sns.heatmap(heat, annot=True, xticklabels=alc, yticklabels=ages, fmt=".5f")
    plt.title("Risk map: P(CRC=Yes | Age, Alcohol)")
    plt.xlabel("Alcohol")
    plt.ylabel("Age")
    plt.show()

    return heat
