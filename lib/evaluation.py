import utils, prepare, validation, visualization

from scipy import interp
from sklearn.metrics import roc_curve, auc, brier_score_loss
from sklearn.preprocessing import minmax_scale, RobustScaler
from sklearn.utils import resample
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import logging, warnings, gc, time

warnings.filterwarnings('ignore')

mpl.rcParams['savefig.dpi'] = 300

logger = logging.getLogger()
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
print = logger.info


def geographic_AUC_scoring(model):
    """
    AUC scores mapped for each cluster of each alpha-fold
    """
    pass

def geographic_brier_scoring(model):
    """
    Brier scoring mapped for each cluster of each alpha-fold
    """
    pass

def prepare_x_test_arr(model):
    # data load
    x_test = utils.open_pickle(model.x_test_path); x_test = utils.remove_expver(x_test)
    y_test = utils.open_pickle(model.y_test_path); y_test = utils.remove_expver(y_test)
    y_test = y_test.drop_vars({ 
        'precipitationCal_cnt',
        'precipitationCal_cnt_cond',
        'HQprecipitation',
        'HQprecipitation_cnt',
        'HQprecipitation_cnt_cond',
        'randomError',
        'randomError_cnt'})
    gaps1 = [time for time in x_test.time.data if time not in y_test.time.data]
    gaps2 = [time for time in y_test.time.data if time not in x_test.time.data]
    invalid_dates = gaps1+gaps2
    x_valid_dates = [dt for dt in x_test.time.data if dt not in invalid_dates]
    y_valid_dates = [dt for dt in y_test.time.data if dt not in invalid_dates]
    x_test = x_test.sel(time = x_valid_dates)
    y_test = y_test.sel(time = y_valid_dates)
    print(x_test)

    ## reshaping
    print(f"\n{utils.time_now()} - reshaping rhum dataarrays now, total levels to loop: {model.rhum_pressure_levels}.")

    reshaped_unnorma_darrays = {}
    reshaped_unnorma_darrays['rhum'], reshaped_unnorma_darrays['uwnd'], reshaped_unnorma_darrays['vwnd'] = {}, {}, {}

    for level in model.rhum_pressure_levels:
        print(f'@{level}... ')
        reshaped_unnorma_darrays['rhum'][level] = np.reshape(
            x_test.rhum.sel(level=level).values, (x_test.time.shape[0], x_test.lat.shape[0]*x_test.lon.shape[0]))

    print(f"\n{utils.time_now()} - reshaping uwnd/vwnd dataarrays now, total levels to loop: {model.uwnd_vwnd_pressure_lvls}.")

    for level in model.uwnd_vwnd_pressure_lvls:
        print(f'@{level}... ')
        reshaped_unnorma_darrays['uwnd'][level] = np.reshape(
            x_test.uwnd.sel(level=level).values, (x_test.time.shape[0], x_test.lat.shape[0]*x_test.lon.shape[0]))
        reshaped_unnorma_darrays['vwnd'][level] = np.reshape(
            x_test.vwnd.sel(level=level).values, (x_test.time.shape[0], x_test.lat.shape[0]*x_test.lon.shape[0]))

    ## stacking unstandardized dataarrays
    print(f"{utils.time_now()} - Stacking unstandardized dataarrays now...")
    stacked_unstandardized_x_test = np.hstack([reshaped_unnorma_darrays[var][lvl] for var in reshaped_unnorma_darrays for lvl in reshaped_unnorma_darrays[var]])

    ## standardizing the stacked dataarrays
    print(f"{utils.time_now()} - Standardizing stacked dataarrays now..., \"stacked_unstandardized_x_test.shape\" is {stacked_unstandardized_x_test.shape}")
    transformer = RobustScaler(quantile_range=(25, 75))
    standardized_stacked_x_test_arr = transformer.fit_transform(stacked_unstandardized_x_test) # som & kmeans training
    transformer.get_params()

    alpha_y_test_path = utils.to_pickle('alpha_y_test', y_test, model.alpha_prepared_dir)
    standardized_stacked_x_test_arr_path = utils.to_pickle(f'standardized_stacked_x_test_arr', standardized_stacked_x_test_arr, model.alpha_prepared_dir)
    return alpha_y_test_path, standardized_stacked_x_test_arr_path

def generate_y_test_with_cluster_memberships(model, alpha, dest):

    if utils.find('standardized_stacked_x_test_arr.pkl', model.alpha_prepared_dir) and \
        utils.find('*alpha_y_test*.pkl', model.alpha_prepared_dir):
        standardized_stacked_x_test_arr_path = utils.find('standardized_stacked_x_test_arr.pkl', model.alpha_prepared_dir)[0]
        alpha_y_test_path = utils.find('*alpha_y_test*', model.alpha_prepared_dir)[0]
    else:
        alpha_y_test_path, standardized_stacked_x_test_arr_path = prepare_x_test_arr(model)

    standardized_stacked_x_test_arr = utils.open_pickle(standardized_stacked_x_test_arr_path)
    y_test = utils.open_pickle(alpha_y_test_path)

    print(f'<alpha-{alpha}> {utils.time_now()} - Acquiring RF predictions for alpha-{alpha}...')
    som = utils.open_pickle(model.som_model_path)
    km = utils.open_pickle(model.kmeans_model_path)

    # get predicted rainfalls for x_test (note: "with_predictions" means cluster memberships attached)
    som_nodes_pred = [som.winner(i) for i in standardized_stacked_x_test_arr]
    som_weights_to_nodes = [som.get_weights()[i[0], i[1]] for i in som_nodes_pred]
    pred_clusters = km.predict(som_weights_to_nodes)
    y_test_with_cluster_memberships = y_test.assign_coords(cluster=("time", pred_clusters))

    return utils.to_pickle(f'y_test_with_cluster_memberships', y_test_with_cluster_memberships, dest)

def generate_brier_scores(model, alpha, dest, clus, clus_brier_scores_dict):
    """
    Generating Brier score for ONE cluster, instead of one whole alpha-fold due to memory constraints.
    """

    if utils.find('*y_test_with_cluster_memberships.pkl', dest):
        y_test_with_cluster_memberships_path = utils.find('y_test_with_cluster_memberships.pkl', dest)[0]
    else:
        y_test_with_cluster_memberships_path = generate_y_test_with_cluster_memberships(model, alpha, dest)

    y_test_with_cluster_memberships = utils.open_pickle(y_test_with_cluster_memberships_path)

    # open y_train to get predicted RF for each cluster
    y_train_path = utils.find('*RFprec_to_ClusterLabels_dataset*', model.alpha_cluster_dir)[0]
    y_train = utils.open_pickle(y_train_path); y_train = utils.remove_expver(y_train)
    y_train = y_train.drop_vars({ 
        'precipitationCal_cnt',
        'precipitationCal_cnt_cond',
        'HQprecipitation',
        'HQprecipitation_cnt',
        'HQprecipitation_cnt_cond',
        'randomError',
        'randomError_cnt'})

    # loop clusters to compare with y_test GT rainfall
    print(f'<alpha-{alpha}> {utils.time_now()} - Calculating Brier scores for cluster {clus+1}.')
    clus_gt = y_test_with_cluster_memberships.where(y_test_with_cluster_memberships.cluster==clus, drop=True)
    clus_pred = y_train.where(y_train.cluster==clus, drop=True)
    clus_pred_proba_ls = np.mean([clus_pred.sel(time=t).precipitationCal.T.values > 1 for t in clus_pred.time.data], axis=0)
    clus_brier_scores_dict[clus] = [brier_score_loss(np.ravel(clus_gt.isel(time=i).precipitationCal.T.values > 1), np.ravel(clus_pred_proba_ls)) for i,t in enumerate(clus_gt.time)]

    time.sleep(1); gc.collect()
    return clus_brier_scores_dict, clus_pred_proba_ls

def aspatial_brier_scores(model, alpha, dest):
    """
    Brier scoring for all clusters in one alpha/n-fold.
    """

    if utils.find(f'*clus_brier_scores_dict*', dest) and utils.find(f'*clus_pred_proba_ls*.pkl', dest):
        clus_brier_scores_path = utils.find(f'*clus_brier_scores_dict*', dest)[0]
    else:
        clus_brier_scores_dict = {}
        clus_pred_proba_ls = []
        for clus in range(model.tl_model.optimal_k):
            clus_brier_scores_dict, clus_pred_proba_ls = generate_brier_scores(model, alpha, dest, clus, clus_brier_scores_dict)
        clus_brier_scores_path = utils.to_pickle(f'clus_brier_scores_dict', clus_brier_scores_dict, dest)
        clus_pred_proba_ls_path = utils.to_pickle(f'clus_pred_proba_ls', clus_pred_proba_ls, dest)

    clus_brier_scores = utils.open_pickle(clus_brier_scores_path)

    ## acquiring mean of Brier scores for this alpha-fold
    clus_brier_scores_flat = [j for i in clus_brier_scores.keys() for j in clus_brier_scores[i]]

    # confidence intervals
    n_iterations = 1000
    n_size = len(clus_brier_scores_flat)
    alph = 0.95

    # run bootstrap
    print(f'<alpha-{alpha}> Running bootstrap resampling on Brier scores to gather CI for alpha-{alpha} mean w/ sample size {n_size}, across all {model.tl_model.optimal_k} clusters.')
    bootstrapped_means = []
    for i in range(n_iterations):
        resamp = resample(clus_brier_scores_flat, n_samples=n_size)
        bootstrapped_means.append(np.mean(resamp))
    bootstrapped_mean = np.mean(bootstrapped_means)

    fig = plt.figure(figsize=(10,5))
    height, bins, patches = plt.hist(sorted(clus_brier_scores_flat), alpha=0.5, facecolor='orange')
    plt.suptitle(f'Brier scores for cluster predictions in alpha-{alpha} vs. ground-truth rainfall (y_test)', fontweight='bold')
    plt.title(f'     sample size: {n_size}, ground-truth years: {model.gt_years}.', loc='left')
    plt.xlim(0,1)
    plt.ylabel('Count')
    plt.xlabel('Brier score')
    plt.plot([bootstrapped_mean,bootstrapped_mean],(0,.98*height.max()), color='k')
    plt.annotate(f'Bootstrapped mean: {bootstrapped_mean:.3f}', xy=(bootstrapped_mean+0.03, .97*(height.max())), fontsize='x-small')
    plt.grid(True, alpha=0.3, color='k')
    plt.savefig(f'{dest}/Brier_scores_for_cluster_predictions_in_this_alpha-{alpha}_vs_y_test.png')
    plt.close('all')
    print(f'{utils.time_now()} - Brier scores for cluster predictions in alpha-{alpha} vs. ground-truth rainfall (y_test) -- printed!')
    
    p = ((1.0-alph)/2.0) * 100
    lower = max(0.0, np.percentile(bootstrapped_means, p))
    p = (alph+((1.0-alph)/2.0)) * 100
    upper = min(1.0, np.percentile(bootstrapped_means, p))
    prompt = f'{alph*100:.1f}% confidence interval for Brier score mean is between {lower:.4f} and {upper:.4f}'
    print(prompt)

    fig = plt.figure(figsize=(10,5))
    height, bins, patches = plt.hist(bootstrapped_means, color='g', alpha=0.7)
    plt.fill_betweenx([0,height.max()], lower, upper, color='r', alpha=0.1, linestyle='-', lw=1)
    plt.ylabel('Count')
    plt.xlabel('Brier Score Mean')
    plt.title(prompt, fontsize='small')
    plt.suptitle(f'Brier score mean for alpha-{alpha}, bootstrapped for {n_iterations} iterations (95% confidence): {np.mean(bootstrapped_means):.3f}', fontweight='bold')
    plt.savefig(f'{dest}/Mean_brier_score_({bootstrapped_mean:.3f})_with_CI_bootstrapped_for_{n_iterations}_iterations.png')
    plt.close('all')
    print(f'{utils.time_now()} - Bootstrapped mean plot for Brier scores -- printed!')

    utils.to_pickle(f'alpha_{alpha}_clus_brier_scores_flat', clus_brier_scores_flat, dest)
    utils.to_pickle(f'alpha_{alpha}_bootstrapped_mean', bootstrapped_mean, dest)
    utils.to_pickle(f'alpha_{alpha}_lower', lower, dest)
    utils.to_pickle(f'alpha_{alpha}_upper', upper, dest)


def roc_auc_curves(model, alpha, dest):
    """
    Adapted from : 
    - https://stats.stackexchange.com/questions/186337/average-roc-for-repeated-10-fold-cross-validation-with-probability-estimates
    - https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    """
    print('Preparing to plot ROCs & uncover AUCs')

    if utils.find('y_test_with_cluster_memberships.pkl', dest):
        y_test_with_cluster_memberships_path = utils.find('y_test_with_cluster_memberships.pkl', dest)[0]
    else:
        y_test_with_cluster_memberships_path = generate_y_test_with_cluster_memberships(model, alpha, dest)

    y_test_with_cluster_memberships = utils.open_pickle(y_test_with_cluster_memberships_path)

    if utils.find('*clus_pred_proba_ls*', dest):
        clus_pred_proba_ls_path = utils.find('*clus_pred_proba_ls*', dest)[0]
    else:
        clus_brier_scores_dict = {}
        clus_pred_proba_ls = []
        for clus in range(model.tl_model.optimal_k):
            clus_brier_scores_dict, clus_pred_proba_ls = generate_brier_scores(model, alpha, dest, clus, clus_brier_scores_dict)
        utils.to_pickle(f'clus_brier_scores_dict', clus_brier_scores_dict, dest)
        utils.to_pickle(f'clus_pred_proba_ls', clus_pred_proba_ls, dest)

    # clus predicted avg RF across the grid
    clus_pred_proba_ls = utils.open_pickle(clus_pred_proba_ls_path)

    tpr = {}
    tprs_alpha = []
    fpr = {}
    thresholds = {}
    roc_auc = {}
 
    base_fpr = np.linspace(0, 1, 101)

    for clus in np.unique(y_test_with_cluster_memberships.cluster):
        print(f'<alpha-{alpha}> Plotting for cluster {clus+1}')
        tpr[clus] = {}; fpr[clus] = {}; thresholds[clus] = {}; roc_auc[clus] = {}
        tprs = []
        plt.figure(figsize=(10, 8))

        clus_gt = y_test_with_cluster_memberships.where(y_test_with_cluster_memberships.cluster==clus, drop=True)

        for i, t in enumerate(clus_gt.time):
    #         fpr[clus][i], tpr[clus][i], thresholds[clus][i] = roc_curve(np.ravel(clus_gt.isel(time=i).precipitationCal.T.values > 1), np.ravel(clus_pred_proba_ls), pos_label=True)
            fpr[clus][i], tpr[clus][i], _ = roc_curve(np.ravel(clus_gt.isel(time=i).precipitationCal.T.values > 1), np.ravel(clus_pred_proba_ls), pos_label=True)
            plt.plot(fpr[clus][i], tpr[clus][i], 'b', alpha=0.15)
            roc_auc[clus][i] = auc(fpr[clus][i], tpr[clus][i])
            tpr_ = interp(base_fpr, fpr[clus][i], tpr[clus][i])
            tpr_[0] = 0.0
            tprs.append(tpr_)
            tprs_alpha.append(tpr_)
        roc_auc[clus]['mean'] = np.mean(list(roc_auc[clus].values()))
        
        tprs = np.array(tprs)
        std = tprs.std(axis=0)

        mean_tprs = tprs.mean(axis=0)
        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std

        auc_val = roc_auc[clus]['mean']

        plt.plot(base_fpr, mean_tprs, 'k')
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.title(f'Receiver operating characteristic for alpha_{alpha}\'s cluster {clus+1}, AUC at {auc_val:.3f}')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.axes().set_aspect('equal', 'datalim')
        plt.savefig(f'{dest}/AUC-{auc_val:.3f}_ROCs_for_alpha_{alpha}_cluster-{clus+1}.png')
        plt.close('all')
        print(f'{utils.time_now()} - Receiver operating characteristic for alpha-{alpha}\'s cluster-{clus+1} -- printed')
    
    tprs_alpha = np.array(tprs_alpha)
    std = tprs_alpha.std(axis=0)

    mean_tprs_alpha = tprs_alpha.mean(axis=0)
    tprs_alpha_upper = np.minimum(mean_tprs_alpha + std, 1)
    tprs_alpha_lower = mean_tprs_alpha - std

    auc_val = np.mean([roc_auc[clus]['mean'] for clus in range(len(roc_auc))])

    plt.figure(figsize=(13, 10))
    plt.plot(base_fpr, mean_tprs_alpha, 'b')
    plt.fill_between(base_fpr, tprs_alpha_lower, tprs_alpha_upper, color='grey', alpha=0.3)

    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.title(f'Receiver operating characteristic for alpha_{alpha} - ALL clusters., AUC at {auc_val:.3f}')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.axes().set_aspect('equal', 'datalim')
    plt.savefig(f'{dest}/AUC-{auc_val:.3f}_ROCs_for_alpha_{alpha}_ALL_cluster.png')
    plt.close('all')
    print(f'{utils.time_now()} - Receiver operating characteristic for alpha-{alpha}\ - ALL clusters. -- printed')
    
    time.sleep(1); gc.collect()
    utils.to_pickle(f'alpha_{alpha}_roc_auc', roc_auc, dest)
    utils.to_pickle(f'alpha_{alpha}_tprs_alpha', tprs_alpha, dest)
    utils.to_pickle(f'alpha_{alpha}_mean_tprs_alpha', mean_tprs_alpha, dest)
