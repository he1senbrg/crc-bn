import numpy as np
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork
from config import VARIABLES


def update_temporal_model(df, model_edges, years, alpha0, initial_pseudo_counts):
    years = sorted(years)
    prev_pseudo = initial_pseudo_counts.copy()
    yearly_models = {}

    for y in years:
        df_y = df[df["Year"] == y][VARIABLES]
        m = DiscreteBayesianNetwork(model_edges)
        m.fit(df_y, estimator=MaximumLikelihoodEstimator)
        yearly_models[y] = m

        new_pseudo = {}
        for v in m.nodes():
            cpd = m.get_cpds(v)
            states = list(cpd.state_names[v])
            values = cpd.get_values()
            flat = np.array(values).reshape(len(states), -1)
            marg = flat.mean(axis=1)
            n_year = df_y.shape[0]
            scale = max(1.0, n_year / 1000.0)
            new_pseudo[v] = {
                s: float(marg[i] * alpha0 * scale) for i, s in enumerate(states)
            }

        for v in VARIABLES:
            if v not in new_pseudo:
                new_pseudo[v] = initial_pseudo_counts.get(v, {})

        prev_pseudo = new_pseudo

    return yearly_models, prev_pseudo
