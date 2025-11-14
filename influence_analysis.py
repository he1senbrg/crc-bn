import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_influence(df, infer, model, sample_size=2000):
    model_nodes = set(model.nodes())
    variables_to_test = [
        v
        for v in ["Smoking", "Diabetes", "Alcohol", "BMI", "Hypertension"]
        if v in model_nodes
    ]
    available_vars = [
        v
        for v in ["Age", "Sex", "Smoking", "Alcohol", "Diabetes", "Hypertension", "BMI"]
        if v in model_nodes
    ]

    sample = df.sample(n=sample_size, random_state=0)
    deltas = {}

    for var in variables_to_test:
        diffs = []
        for _, r in sample.iterrows():
            evidence = {
                k: v for k, v in r[available_vars].to_dict().items() if k != var
            }
            base_q = infer.query(["CRC"], evidence=evidence, show_progress=False)
            yes_idx = base_q.state_names["CRC"].index("Yes")
            p_base = float(base_q.values[yes_idx])

            if var in ["Smoking", "Diabetes", "Hypertension"]:
                toggled = "Yes" if evidence.get(var, "No") == "No" else "No"
            elif var == "Alcohol":
                toggled = "High" if evidence.get(var, "None") != "High" else "None"
            else:  # BMI
                toggled = (
                    "Obese" if evidence.get(var, "Normal") != "Obese" else "Normal"
                )

            evidence_t = evidence.copy()
            evidence_t[var] = toggled
            q2 = infer.query(["CRC"], evidence=evidence_t, show_progress=False)
            p_tog = float(q2.values[yes_idx])
            diffs.append(p_tog - p_base)
        deltas[var] = np.mean(diffs)

    return deltas


def plot_influence_ranking(deltas):
    influence = pd.DataFrame(
        {"variable": list(deltas.keys()), "mean_delta": list(deltas.values())}
    )
    influence["abs_delta"] = influence["mean_delta"].abs()
    influence = influence.sort_values("abs_delta", ascending=False)

    plt.figure(figsize=(6, 3))
    sns.barplot(x="abs_delta", y="variable", data=influence)
    plt.xlabel("Average |ΔP(CRC)| when toggled")
    plt.title("Influence ranking")
    plt.show()

    return influence
