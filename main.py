from pgmpy.inference import VariableElimination

from config import VARIABLES, YEARS
from data_generator import generate_synthetic_crc_data
from structure_learning import learn_structure
from parameter_estimation import compute_marginal_priors, create_pseudo_counts, fit_model_with_priors
from temporal_updating import update_temporal_model
from prediction import create_final_model, evaluate_model, get_confusion_matrix
from calibration import calibration_quantile_bin, plot_calibration
from visualization import create_risk_map
from influence_analysis import calculate_influence, plot_influence_ranking

def main():
    df = generate_synthetic_crc_data()
    print("Rows:", len(df))
    print("CRC prevalence:", (df['CRC']=='Yes').mean())
    
    model = learn_structure(df)
    print("Learned edges:", model.edges())
    
    prior_means = compute_marginal_priors(df[VARIABLES], VARIABLES)
    N = len(df)
    alpha0 = N / 10000.0
    print("alpha0:", alpha0)
    
    pseudo_counts = create_pseudo_counts(prior_means, alpha0, VARIABLES)
    model = fit_model_with_priors(model, df[VARIABLES], pseudo_counts)
    print("Example CPT (CRC) - shape and states:")
    print(model.get_cpds('CRC'))
    
    years = sorted(df['Year'].unique())
    yearly_models, final_pseudo_counts = update_temporal_model(df, model.edges(), years, alpha0, pseudo_counts)
    print("Final pseudo-counts sample for CRC:", final_pseudo_counts['CRC'])
    
    test_year = max(years)
    train_df = df[df['Year'] < test_year]
    test_df = df[df['Year'] == test_year].reset_index(drop=True)
    
    final_model = create_final_model(train_df, model.edges(), final_pseudo_counts)
    infer = VariableElimination(final_model)
    
    test_df, auc, best_g = evaluate_model(test_df, infer, final_model)
    print("Test AUC:", auc)
    print("Best G-mean, Sens, Spec, Threshold:", best_g)
    
    tn, fp, fn, tp = get_confusion_matrix(test_df, best_g[3])
    print("Confusion matrix (tn, fp, fn, tp):", (tn, fp, fn, tp))
    
    cal = calibration_quantile_bin(test_df['y_true'], test_df['pred_prob'], n_bins=10)
    plot_calibration(cal)
    
    heat = create_risk_map(df, infer)
    
    deltas = calculate_influence(df, infer, final_model)
    influence = plot_influence_ranking(deltas)
    print("Influence ranking:\n", influence)
    
    summary = {
        'n_total': len(df),
        'crc_prevalence': (df['CRC']=='Yes').mean(),
        'test_auc': auc,
        'best_gmean': best_g[0],
        'best_sensitivity': best_g[1],
        'best_specificity': best_g[2],
        'best_threshold': best_g[3]
    }
    print("Summary:", summary)

if __name__ == "__main__":
    main()