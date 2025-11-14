import numpy as np
from pgmpy.estimators import MaximumLikelihoodEstimator

def compute_marginal_priors(data, variables):
    priors = {}
    for v in variables:
        priors[v] = data[v].value_counts(normalize=True).to_dict()
    return priors

def create_pseudo_counts(prior_means, alpha0, variables):
    pseudo_counts = {}
    for v in variables:
        states = list(prior_means[v].keys())
        pseudo_counts[v] = {s: prior_means[v][s] * alpha0 for s in states}
    return pseudo_counts

def fit_model_with_priors(model, data, pseudo_counts=None):
    if pseudo_counts is None:
        model.fit(data, estimator=MaximumLikelihoodEstimator)
    else:
        model.fit(data, estimator=MaximumLikelihoodEstimator)
    return model