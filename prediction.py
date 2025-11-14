import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from config import VARIABLES

def predict_prob_crc(infer, row, model_nodes):
    evidence = row.to_dict()
    evidence.pop('PatientID', None)
    evidence.pop('Year', None)
    evidence.pop('CRC', None)
    
    evidence = {k: v for k, v in evidence.items() if k in model_nodes and k != 'CRC'}
    
    q = infer.query(['CRC'], evidence=evidence, show_progress=False)
    yes_idx = q.state_names['CRC'].index('Yes')
    return float(q.values[yes_idx])

def create_final_model(train_df, model_edges, final_pseudo_counts):
    final_model = DiscreteBayesianNetwork(model_edges)
    model_vars = [var for var in VARIABLES if var in final_model.nodes()]
    final_model.fit(train_df[model_vars], estimator=MaximumLikelihoodEstimator)
    return final_model

def evaluate_model(test_df, infer, model):
    model_nodes = set(model.nodes())
    test_df['pred_prob'] = test_df.apply(lambda r: predict_prob_crc(infer, r, model_nodes), axis=1)
    test_df['y_true'] = (test_df['CRC']=='Yes').astype(int)
    
    auc = roc_auc_score(test_df['y_true'], test_df['pred_prob'])
    
    best_g = (0, 0., 0., 0.0)
    for thr in np.linspace(0, 0.1, 201):
        pred = (test_df['pred_prob'] >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(test_df['y_true'], pred).ravel()
        sens = tp/(tp+fn) if (tp+fn)>0 else 0.0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
        g = np.sqrt(sens*spec)
        if g > best_g[0]:
            best_g = (g, sens, spec, thr)
    
    return test_df, auc, best_g

def get_confusion_matrix(test_df, threshold):
    pred = (test_df['pred_prob'] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_df['y_true'], pred).ravel()
    return tn, fp, fn, tp