import utils, prepare, validation, visualization

from scipy import interp
from sklearn.metrics import roc_curve, auc, brier_score_loss
from sklearn.preprocessing import minmax_scale, RobustScaler
from sklearn.utils import resample
from cartopy import feature as cf
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import logging, warnings, gc, time, itertools
from pathlib import Path

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
    desired_res=0.75
    coarsen_magnitude = int(desired_res/np.ediff1d(x_test.isel(lon=slice(0,2)).lon.data)[0])
    print(f'Coarsen magnitude set at: {coarsen_magnitude} toward desired spatial resolu. of {desired_res}')
    x_test = x_test.coarsen(lat=coarsen_magnitude, lon=coarsen_magnitude, boundary='trim').mean()
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

    if utils.find('*standardized_stacked_x_test_arr.pkl', model.alpha_prepared_dir) and \
        utils.find('*alpha_y_test*.pkl', model.alpha_prepared_dir):
        standardized_stacked_x_test_arr_path = utils.find('*standardized_stacked_x_test_arr.pkl', model.alpha_prepared_dir)[0]
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

    utils.to_pickle(f'y_test_with_cluster_memberships', y_test_with_cluster_memberships, dest)

def generate_y_test_with_cluster_memberships_for_specific_clusters(model, dest, y_test_with_cluster_memberships_path):
    y_test_with_cluster_memberships = utils.open_pickle(y_test_with_cluster_memberships_path[0])
    for i in range(model.tl_model.optimal_k):
        print(f'generating clus_gt_{i}')
        utils.to_pickle(f'clus_gt_{i}', y_test_with_cluster_memberships.where(y_test_with_cluster_memberships.cluster==i, drop=True), dest)

def generate_clus_pred(model, dest, y_train_path):
    y_train = utils.open_pickle(y_train_path); y_train = utils.remove_expver(y_train)
    y_train = y_train.drop_vars({ 
        'precipitationCal_cnt',
        'precipitationCal_cnt_cond',
        'HQprecipitation',
        'HQprecipitation_cnt',
        'HQprecipitation_cnt_cond',
        'randomError',
        'randomError_cnt'})
    for i in range(model.tl_model.optimal_k):
        print(f'generating clus_pred_{i}')
        utils.to_pickle(f'clus_pred_{i}', y_train.where(y_train.cluster==i, drop=True), dest)

def generate_brier_scores(model, alpha, clus):
    """
    Generating Brier score for ONE cluster, instead of one whole alpha-fold due to memory constraints.
    """

    if utils.find('*y_test_with_cluster_memberships.pkl', model.alpha_cluster_scoring_dir):
        pass
    else:
        print(f'y_test_with_cluster_memberships not found...')
        generate_y_test_with_cluster_memberships(model, alpha, model.alpha_cluster_scoring_dir)
    y_test_with_cluster_memberships_path = utils.find('*y_test_with_cluster_memberships.pkl', model.alpha_cluster_scoring_dir)

    # y_test_with_cluster_memberships = utils.open_pickle(y_test_with_cluster_memberships_path)

    # open y_train to get predicted RF for each cluster
    y_train_path = utils.find('*RFprec_to_ClusterLabels_dataset*', model.alpha_cluster_dir)[0]
    
    if len(utils.find(f'*clus_gt_*', model.alpha_cluster_dir)) == model.tl_model.optimal_k: pass
    else:
        generate_y_test_with_cluster_memberships_for_specific_clusters(model, model.alpha_cluster_dir, y_test_with_cluster_memberships_path)
    clus_gt = utils.open_pickle(utils.find(f'*clus_gt_{clus}*', model.alpha_cluster_dir)[0])
    
    if len(utils.find(f'*clus_pred_*', model.alpha_cluster_dir)) == model.tl_model.optimal_k: pass
    else:
        generate_clus_pred(model, model.alpha_cluster_dir, y_train_path)
    clus_pred = utils.open_pickle(utils.find(f'*clus_pred_{clus}*', model.alpha_cluster_dir)[0])
    
    
    # loop clusters to compare with y_test GT rainfall
    print(f'<alpha-{alpha}> Calculating Brier scores for cluster {clus}.')
    
    print(f'{utils.time_now()} - calculating clus_pred_proba_ls...')
    clus_pred_proba_ls = np.mean([clus_pred.sel(time=t).precipitationCal.T.values > 1 for t in clus_pred.time.data], axis=0)

    print(f'{utils.time_now()} - calculating clus_brier_scores_dict[clus]...')
    clus_brier_scores_dict = {}
    clus_brier_scores_dict[clus] = [brier_score_loss(
        np.ravel(clus_gt.isel(time=i).precipitationCal.T.values > 1), np.ravel(clus_pred_proba_ls)) for i,t in enumerate(clus_gt.time)]
        
    utils.to_pickle(f'clus_pred_proba_ls_cluster_{clus}', clus_pred_proba_ls, model.alpha_cluster_scoring_dir)
    utils.to_pickle(f'clus_brier_scores_dict_cluster_{clus}', clus_brier_scores_dict, model.alpha_cluster_scoring_dir)

    time.sleep(1); gc.collect()

def generate_clus_brier_scores_dict_full(model, dest):
    clus_brier_scores_dict_cluster_paths = utils.find(f'*clus_brier_scores_dict_cluster_*', dest)
    clus_brier_scores = {}
    for dict_path in clus_brier_scores_dict_cluster_paths:
        dic = utils.open_pickle(dict_path)
        k,v = list(dic.items())[0] 
        clus_brier_scores[k] = v
    return clus_brier_scores


def mean_brier_individual_alpha(model, alpha):
    """
    Brier scoring for all clusters in one alpha/cv fold.
    """
    found = utils.find(f'*clus_brier_scores_dict_cluster_*', model.alpha_cluster_scoring_dir)
    if len(found) == model.tl_model.optimal_k: pass
    else:
        for clus in range(model.tl_model.optimal_k):
            if f'clus_brier_scores_dict_cluster_{clus}' in found: continue
            print('Generating Brier scores now...')
            generate_brier_scores(model, alpha, clus)

    # clus_brier_scores = utils.open_pickle(clus_brier_scores_path)
    clus_brier_scores = generate_clus_brier_scores_dict_full(model, model.alpha_cluster_scoring_dir)

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
    plt.plot([bootstrapped_mean,bootstrapped_mean],(0,height.max()), color='r', linestyle='--', alpha=0.4, lw=0.7)
    plt.annotate(f'Bootstrapped mean: {bootstrapped_mean:.3f}', xy=(bootstrapped_mean+0.03, .97*(height.max())), fontsize='x-small')
    plt.grid(True, alpha=0.3, color='k')
    plt.savefig(f'{model.alpha_cluster_scoring_dir}/Brier_scores_for_cluster_predictions_in_this_alpha-{alpha}_vs_y_test.png')
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
    plt.fill_betweenx([0,height.max()], lower, upper, color='r', alpha=0.1, linestyle='-', lw=0.7)
    plt.plot([bootstrapped_mean,bootstrapped_mean],(0,height.max()), color='grey', linestyle='--')
    plt.ylabel('Count')
    plt.xlabel('Brier Score Mean')
    plt.title(prompt, fontsize='small')
    plt.suptitle(f'Brier score mean for alpha-{alpha}, bootstrapped for {n_iterations} iterations (95% confidence): {np.mean(bootstrapped_means):.3f}', fontweight='bold')
    plt.savefig(f'{model.alpha_cluster_scoring_dir}/Mean_brier_score_({bootstrapped_mean:.3f})_in_alpha-{alpha}_{n_iterations}_iterations_bootstrap.png')
    plt.close('all')
    print(f'{utils.time_now()} - Bootstrapped mean plot for Brier scores -- printed!')

    utils.to_pickle(f'alpha_{alpha}_clus_brier_scores_flat', clus_brier_scores_flat, model.alpha_cluster_scoring_dir)
    utils.to_pickle(f'alpha_{alpha}_bootstrapped_mean', bootstrapped_mean, model.alpha_cluster_scoring_dir)
    utils.to_pickle(f'alpha_{alpha}_lower', lower, model.alpha_cluster_scoring_dir)
    utils.to_pickle(f'alpha_{alpha}_upper', upper, model.alpha_cluster_scoring_dir)


def generate_brier_dict_all(model):
    all_brier_scores_dict_clusters_pathls = np.array([*Path(model.tl_model.cluster_dir).glob('*/**/*scores_dict_cluster*')])
    alpha_siz = len(all_brier_scores_dict_clusters_pathls)
    all_brier_scores_dict_clusters_pathls = np.array_split(all_brier_scores_dict_clusters_pathls, model.ALPHAs)
    print(all_brier_scores_dict_clusters_pathls)
    brier_dict_all = {}
    for alp in range(model.ALPHAs):
        brier_dict_all[alp] = {}
        for i, path in enumerate(all_brier_scores_dict_clusters_pathls[alp]):
            brier_dict_all[alp][i] = utils.open_pickle(path)[i]
    utils.to_pickle(f'brier_dict_all', brier_dict_all, model.alpha_general_dir)


def mean_brier_scores_all_alphas(model):
    if [*Path(model.tl_model.cluster_dir).glob('**/*/*/*/*brier_scores_dict.pkl')]: pass
    else: generate_brier_dict_all(model)

    # brier_dict_all = [utils.open_pickle(i) for i in [*Path(model.tl_model.cluster_dir).glob('**/*/*/*/*brier_scores_dict.pkl')]]
    # for i in Path(model.tl_model.cluster_dir).glob('**/*/*/*/*brier_scores_dict.pkl'): print(i)
    brier_dict_all = utils.open_pickle(f'{model.alpha_general_dir}/brier_dict_all.pkl')

    # clus_brier_scores_flat_ALL = [val for alpha_dic in brier_dict_all for clus in alpha_dic for val in alpha_dic[clus]]
    # n_size = len(clus_brier_scores_flat_ALL)
    # print(f'n_size is {n_size}')
    cluster_sizes = np.array([utils.open_pickle(i) for i in [*Path(model.tl_model.cluster_dir).glob('**/**/*cluster_size*.pkl')]])
    tots = cluster_sizes.sum()
    print(f'n_size is {tots}')

    clus_brier_scores_flat_for_each_alpha = [
        [val for clus in brier_dict_all[alpha_dic] for val in brier_dict_all[alpha_dic][clus]
        ] for alpha_dic in brier_dict_all]
    clus_dist_for_each_alpha = [[len(brier_dict_all[alpha_dic][clus]) for clus in brier_dict_all[alpha_dic]] for alpha_dic in brier_dict_all]
    for clus_sizes_in_an_alpha in clus_dist_for_each_alpha:
        print(clus_sizes_in_an_alpha)
    print(len(brier_dict_all))
    print(len(clus_dist_for_each_alpha))
    print(len(clus_brier_scores_flat_for_each_alpha))
    print(len(clus_dist_for_each_alpha[0]))
    print(len(clus_brier_scores_flat_for_each_alpha[0]))

    plt.figure(figsize=(15,8))

    for i,d in enumerate(clus_brier_scores_flat_for_each_alpha):
        n_datapts = len(d)
        plt.plot(sorted(d), np.arange(len(d)), label=f'alpha{i+1}, sample_size of test set = {n_datapts}')
    
    plt.title(f'Brier score distributions across the test scenarios in 5-fold CV, total population size: {len([j for i in clus_brier_scores_flat_for_each_alpha for j in i])}', 
    size='xx-large', fontweight='bold')
    plt.legend()
    plt.savefig(f'{model.alpha_general_dir}/Brier_score_distribution_curves')
    plt.close('all')
    df = pd.DataFrame(clus_brier_scores_flat_for_each_alpha, index=[f'alpha{i}' for i in np.arange(1,6)]).T

    # mean_brier_for_each_alpha = [np.mean(i) for i in clus_brier_scores_flat_for_each_alpha]
    # print(mean_brier_for_each_alpha)
    # print(np.mean(mean_brier_for_each_alpha))

    bootstrapped_brier_scores = []
    n_iterations = 100
    for alpha in range(len(clus_brier_scores_flat_for_each_alpha)):
        for i in range(n_iterations):
            """
            Weighted for each cluster in each alpha
            """
            resamp = resample(clus_brier_scores_flat_for_each_alpha[alpha], 
                            n_samples=len(clus_brier_scores_flat_for_each_alpha[alpha]), 
    #                           stratify=clus_dist_for_each_alpha[alpha]
                            stratify=[val for i,v in enumerate(clus_dist_for_each_alpha[alpha]) for val in list(np.full(v, i))]
                            )
            bootstrapped_brier_scores.append(np.mean(resamp))
    bootstrapped_mean = np.mean(bootstrapped_brier_scores)

    alph = 0.95
    p = ((1.0-alph)/2.0) * 100
    lower = max(0.0, np.percentile(bootstrapped_brier_scores, p))
    p = (alph+((1.0-alph)/2.0)) * 100
    upper = min(1.0, np.percentile(bootstrapped_brier_scores, p))
    prompt = f'Weighted brier scores resampled according to cluster sizes in each cross-val split,'\
        f'\n{alph*100:.1f}% confidence interval for overall Brier score mean ({bootstrapped_mean:.4f}) is between {lower:.4f} and {upper:.4f}'
    print(prompt)

    fig = df.boxplot(figsize=(15,8), fontsize=15, flierprops=dict(markerfacecolor='g', marker='D'), color='b')
    plt.suptitle('Brier scores boxplot across all cross-val test sets', fontsize=25, fontweight='bold')
    plt.title(f'{prompt}', fontsize=12)
    plt.plot([i+1 for i in range(5)], [bootstrapped_mean for i in range(5)], 'r--', lw=3)
    plt.savefig(f'{model.alpha_general_dir}/Brier_scores_weighted-avg_boxplot_with_bootstrapped_whole-model_mean')
    plt.close('all')


def brier_along_grid_axis(true, pred):
    return np.apply_along_axis(func1d=brier_score_loss, axis=0, arr=true, y_prob=pred)

def generate_gridded_brier_for_cluster(clus, gt_dataset_path, clus_pred_proba_ls_cluster_pkl, model):
    # y_test for this cluster
    y_test = utils.open_pickle(gt_dataset_path) 
    cluster_size = y_test.time.size

    """ Too slow!
    
    # list of dates with array in lat-lon dimensions representing condition of >1 mm rainfall or not (Bool)
    gt_results = [y_test.isel(time=t).precipitationCal.T.values > 1 for t in range(cluster_size)]

    # array in same lat-lon dims for predicted rainfall values if they meet condition (of >1mm)
    pred_results = np.ravel(utils.open_pickle(clus_pred_proba_ls_cluster_pkl)) # y_train for cluster [i]

    date_to_prediction = np.array([[np.ravel(date), pred_results] for date in gt_results])

    gridded_brier_for_clus_i = np.array([brier_along_grid_axis(np.atleast_1d(gt_grid), np.atleast_1d(pred_proba_grid)) 
                                       for gt_row, pred_proba_row in date_to_prediction # for each day in testset
                                       for gt_grid, pred_proba_grid in zip(gt_row,pred_proba_row)] # for each grid in that day
                                       ).reshape((cluster_size,-1))            
    """

    gt_results = np.array([y_test.isel(time=t).precipitationCal.T.values > 1 for t in range(cluster_size)])
    pred_results = np.ravel(utils.open_pickle(clus_pred_proba_ls_cluster_pkl)) # y_train for cluster [i]
    date_to_prediction = np.array([[np.ravel(date), pred_results] for date in gt_results]) # shape is (row, 2, grids)

    """
    Hypothesis: gridded brier with only 1 element sounds terrible. Instead, try to aggregate all dates
    across all clusters, along with cluster sizes, then create empty grid,
    dump all date_to_prediction(s) to this grid,
    and use the bottom sped up way of calculating gridded brier!

    Now: creating all date_to_prediction to try on .ipynb
    """
    # gridded_brier_for_clus_i = np.array(
    #     [np.apply_along_axis(func1d=brier_score_loss, axis=0, arr=arr[:,None], y_prob=y_prob[:,None]) 
    #     for arr, y_prob in zip(date_to_prediction.T[:,0], date_to_prediction.T[:,1])]
    #     )

    # utils.to_pickle(f'{utils.time_now()}_gridded_brier_cluster_{clus}', gridded_brier_for_clus_i, model.alpha_cluster_scoring_dir)
    utils.to_pickle(f'cluster_{clus}_cluster_size', cluster_size, model.alpha_cluster_scoring_dir)
    utils.to_pickle(f'cluster_{clus}_date_to_prediction', date_to_prediction, model.alpha_cluster_scoring_dir)

    
def gridded_brier_individual_alpha(model, alpha):
    clus_pred_proba_ls_cluster_ls = utils.find('*clus_pred_proba_ls_cluster*', model.alpha_cluster_scoring_dir)
    clus_gt_ls = utils.find('*clus_gt*', model.alpha_cluster_dir)
    for clus,gt_dataset_path in enumerate(clus_gt_ls):
        if utils.find(f'*cluster_{clus}_date_to_prediction.pkl', model.alpha_cluster_scoring_dir) and \
            utils.find(f'*cluster_{clus}_cluster_size.pkl', model.alpha_cluster_scoring_dir):
            continue
        else:
            print(f'{utils.time_now()} - <alpha-{alpha}> No "cluster_{clus}_date_to_prediction" found, generating now...')
            generate_gridded_brier_for_cluster(clus, gt_dataset_path, clus_pred_proba_ls_cluster_ls[clus], model)

    print(f'{utils.time_now()} - Ended loop, now beginning preprocessing...')
    ds = utils.open_pickle([*Path(model.alpha_cluster_dir).glob('*RFprec_to_ClusterLabels_dataset*')][0])
    lon_coords = ds.lon.data
    lat_coords = ds.lat.data
    lon_coords_min = min(lon_coords)-.5
    lat_coords_min = min(lat_coords)-.5
    lon_coords_max = max(lon_coords)+.5
    lat_coords_max = max(lat_coords)+.5
    lat = ds.lat.size
    lon = ds.lon.size
    shape = (lat, lon)

    date_to_prediction_pathsls = utils.find(f'*_date_to_prediction.pkl', model.alpha_cluster_scoring_dir)
    cluster_size_pathsls = utils.find(f'*cluster_size.pkl', model.alpha_cluster_scoring_dir)
    tots = sum([utils.open_pickle(i) for i in cluster_size_pathsls])

    desired_flat_size = shape[0]*shape[1]
    grid_gt = np.zeros((tots,desired_flat_size))
    grid_pred = np.zeros((tots,desired_flat_size))
    print(grid_pred.shape)
    CURSOR = 0
    for pkl in date_to_prediction_pathsls:
        for date, arr in enumerate(utils.open_pickle(pkl)):
            grid_gt[CURSOR] = arr[0]
            grid_pred[CURSOR] = arr[1]
            CURSOR += 1

    gridded_brier_for_all_clus = np.array(
        [np.apply_along_axis(func1d=brier_score_loss, axis=0, arr=arr[:,None], y_prob=y_prob[:,None]) 
        for arr, y_prob in zip(grid_gt.T, grid_pred.T)]
        )

    gridded_brier_for_all_clus = gridded_brier_for_all_clus.reshape(shape)

    print(f'{utils.time_now()} - Now plotting...')
    fig = plt.figure(figsize=(20,8))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(model.tl_model.lon_formatter)
    ax.yaxis.set_major_formatter(model.tl_model.lat_formatter)
    ax.add_feature(cf.COASTLINE, color='black', linewidth=1, linestyle='-') 
    ax.set_extent([lon_coords_min, lon_coords_max, lat_coords_min, lat_coords_max])
    contf = ax.contourf(lon_coords, lat_coords, gridded_brier_for_all_clus, cmap="brg_r", alpha=0.8, 
                        vmin=0, vmax=1.0, levels=np.linspace(0,1,41))
    m = plt.cm.ScalarMappable(cmap="brg_r")
    m.set_array(gridded_brier_for_all_clus)
    m.set_clim(0.,1.)
    plt.title(f'Brier score average across all {model.tl_model.optimal_k} clusters in alpha-1', fontweight='bold', fontsize=17, y=1.02)
    ax.set_xticks([min(lon_coords), 
                min(lon_coords)+(max(lon_coords)-min(lon_coords))/4, 
                min(lon_coords)+(max(lon_coords)-min(lon_coords))/2,
                min(lon_coords)+3*(max(lon_coords)-min(lon_coords))/4, 
                max(lon_coords)], crs=ccrs.PlateCarree())
    ax.set_yticks([min(lat_coords), 
                min(lat_coords)+(max(lat_coords)-min(lat_coords))/4, 
                min(lat_coords)+(max(lat_coords)-min(lat_coords))/2,
                min(lat_coords)+3*(max(lat_coords)-min(lat_coords))/4, 
                max(lat_coords)], crs=ccrs.PlateCarree())
    cbar = plt.colorbar(contf, shrink=0.7)
    plt.savefig(f'{model.alpha_cluster_scoring_dir}/Gridded_brier_individual_alpha_{alpha}_v2', bbox_inches='tight', pad_inches=.7)
    plt.close('all')
    print(f'<alpha-{alpha}> Gridded brier individual plotted!')

    utils.to_pickle(f'alpha_{alpha}_gridded_brier_for_all_clus', gridded_brier_for_all_clus, model.alpha_general_dir)


def gridded_brier_all_alphas(model):
    alpha_gridded_briers_pathls = utils.find('*alpha_*_gridded_brier_for_all_clus.pkl', model.alpha_general_dir)
    print(alpha_gridded_briers_pathls)
    alpha_gridded_briers = np.array([np.array(utils.open_pickle(i)) for i in alpha_gridded_briers_pathls])
    print(alpha_gridded_briers.shape)

    # date_to_prediction_pathsls = [i for i in [*Path(model.tl_model.cluster_dir).glob('**/**/*date_to_prediction*.pkl')] if 'alpha_general' not in str(i)]
    cluster_sizes = np.array([utils.open_pickle(i) for i in [*Path(model.tl_model.cluster_dir).glob('**/**/*cluster_size*.pkl')]])
    tots = cluster_sizes.sum()

    alpha_sizes = np.array([i.sum() for i in np.array_split(cluster_sizes, model.ALPHAs)])
    print(alpha_sizes)
    # alph_weights = alpha_sizes/tots
    # weights = np.concatenate([np.full((alpha, ), alph_weights[i]) for i,alpha in enumerate(alpha_sizes)])
    # print(weights.shape)
    gridded_brier_for_all_alphas = np.average(alpha_gridded_briers, axis=0, weights=alpha_sizes)
    # gridded_brier_for_all_alphas = np.average(alpha_gridded_briers, axis=0, weights=weights)
    
    print(f'{utils.time_now()} - Taking coordinate data from {model.RFprec_to_ClusterLabels_dataset_path}')
    ds = utils.open_pickle(model.RFprec_to_ClusterLabels_dataset_path)
    lon_coords = ds.lon.data
    lat_coords = ds.lat.data
    lon_coords_min = min(lon_coords)-.5
    lat_coords_min = min(lat_coords)-.5
    lon_coords_max = max(lon_coords)+.5
    lat_coords_max = max(lat_coords)+.5
    shape = (ds.lat.size, ds.lon.size)

    # desired_flat_size = shape[0]*shape[1]
    # empty_grid_truths = np.zeros((tots,desired_flat_size))
    # empty_grid_preds = np.zeros((tots,desired_flat_size))
    # CURSOR = 0
    # for pkl in date_to_prediction_pathsls:
    #     for date, arr in enumerate(utils.open_pickle(pkl)):
    #         empty_grid_truths[CURSOR] = arr[0]
    #         empty_grid_preds[CURSOR] = arr[1]
    #         CURSOR += 1

    # gridded_brier_for_all_alphas = np.array(
    #     [np.apply_along_axis(func1d=brier_score_loss, axis=0, arr=arr[:,None], y_prob=y_prob[:,None]) 
    #     for arr, y_prob in zip(empty_grid_truths.T, empty_grid_preds.T)]
    #     )

    gridded_brier_for_all_alphas = gridded_brier_for_all_alphas.reshape(shape)

    print(f'{utils.time_now()} - Now plotting...')
    fig = plt.figure(figsize=(25,9))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(model.tl_model.lon_formatter)
    ax.yaxis.set_major_formatter(model.tl_model.lat_formatter)
    ax.add_feature(cf.COASTLINE, color='black', linewidth=1, linestyle='-') 
    ax.set_extent([lon_coords_min, lon_coords_max, lat_coords_min, lat_coords_max])

    contf = ax.contourf(lon_coords, lat_coords, gridded_brier_for_all_alphas, cmap="brg_r", alpha=0.8, 
                        vmin=0, vmax=1.0, levels=np.linspace(0,1,21))
    conts = ax.contour(contf, 'k--', alpha=0.5, lw=.1)

    ax.clabel(conts, conts.levels, colors='k', inline=False, fontsize=7)

    m = plt.cm.ScalarMappable(cmap="brg_r")
    m.set_array(gridded_brier_for_all_alphas)
    m.set_clim(0.,1.)
    plt.suptitle(f'Brier score averages across all 5 cross-val test sets (n={tots}).', 
        fontweight='bold', fontsize=17, x=.67, y=.99)
    plt.title('Scores have been averaged across all dates of each test set. \n' \
        'Scores approaching 0 indicate better calibrated predictive models, \n' \
        'while 0.25 may represent forecasts of 50%, regardless of outcome.', y=1.01)
    ax.set_xticks([min(lon_coords), 
                min(lon_coords)+(max(lon_coords)-min(lon_coords))/4, 
                min(lon_coords)+(max(lon_coords)-min(lon_coords))/2,
                min(lon_coords)+3*(max(lon_coords)-min(lon_coords))/4, 
                max(lon_coords)], crs=ccrs.PlateCarree())
    ax.set_yticks([min(lat_coords), 
                min(lat_coords)+(max(lat_coords)-min(lat_coords))/4, 
                min(lat_coords)+(max(lat_coords)-min(lat_coords))/2,
                min(lat_coords)+3*(max(lat_coords)-min(lat_coords))/4, 
                max(lat_coords)], crs=ccrs.PlateCarree())
    cbar = plt.colorbar(contf, shrink=0.85, ticks=np.linspace(0,1,11)[1:], pad=.02)
    cbar.set_label('Brier score (0.0 is optimal. 1.0 indicates completely wrong forecast).', labelpad=13)
    plt.savefig(f'{model.alpha_general_dir}/gridded_brier_whole-model_v2', bbox_inches='tight', pad_inches=.7)
    plt.close('all')


def ROC_AUC_individual_alpha(model, alpha):
    """
    ROC & AUC for a single alpha, across all clusters inside. 

    Adapted from : 
    - https://stats.stackexchange.com/questions/186337/average-roc-for-repeated-10-fold-cross-validation-with-probability-estimates
    - https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    """
    print('Preparing to plot ROCs & uncover AUCs')

    if utils.find('*y_test_with_cluster_memberships.pkl', model.alpha_cluster_scoring_dir):
        pass
    else:
        generate_y_test_with_cluster_memberships(model, alpha, model.alpha_cluster_scoring_dir)

    y_test_with_cluster_memberships_path = utils.find('*y_test_with_cluster_memberships.pkl', model.alpha_cluster_scoring_dir)
    if utils.find(f'*clus_gt_*', model.alpha_cluster_dir): pass
    else:
        generate_y_test_with_cluster_memberships_for_specific_clusters(model, model.alpha_cluster_dir, y_test_with_cluster_memberships_path)
    found_clus_gts = utils.find(f'*clus_gt_*', model.alpha_cluster_dir)
    print(f'found_clus_gts is: \n{found_clus_gts}')

    found = utils.find(f'*clus_pred_proba_ls_cluster*', model.alpha_cluster_scoring_dir)
    if len(found) == model.tl_model.optimal_k: pass
    else:
        for clus in range(model.tl_model.optimal_k):
            if f'clus_pred_proba_ls_cluster_{clus}' in found: continue
            print('Generating Brier scores now...')
            generate_brier_scores(model, alpha, clus)

    # clus predicted avg RF across the grid
    clus_pred_proba_ls_paths = utils.find('*clus_pred_proba_ls_cluster*', model.alpha_cluster_scoring_dir)
    clus_pred_proba_ls = [utils.open_pickle(path) for path in clus_pred_proba_ls_paths]
    print(f'clus_pred_proba_ls.shape is : \n{[i.shape for i in clus_pred_proba_ls]}')

    tpr = {} # holds TPR in dict form for use in intermediate steps
    tprs_alpha = [] # holds interpolated TPRs for all clusters in this alpha, used to calc mean TPR for this alpha at the end
    fpr = {}
    base_fpr = np.linspace(0, 1, 101)
    roc_auc = {}

    for clus in range(model.tl_model.optimal_k):
        print(f'<alpha-{alpha}> Plotting for cluster {clus}')
        tpr[clus] = {}; fpr[clus] = {}; roc_auc[clus] = {}
        tprs = []
        plt.figure(figsize=(10, 8))

        clus_gt = utils.open_pickle(found_clus_gts[clus])
        print(clus_gt)

        for i in range(clus_gt.time.size): # each row of the cluster
            fpr[clus][i], tpr[clus][i], _ = roc_curve( # across ALL grids
                np.ravel(clus_gt.isel(time=i).precipitationCal.T.values > 1), np.ravel(clus_pred_proba_ls[clus]), pos_label=True
                )
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
        plt.title(f'Receiver operating characteristic for alpha_{alpha}\'s cluster {clus}, AUC at {auc_val:.3f}')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.axes().set_aspect('equal', 'datalim')
        plt.savefig(f'{model.alpha_cluster_scoring_dir}/AUC-{auc_val:.3f}_ROCs_for_alpha_{alpha}_cluster-{clus}.png', bbox_inches='tight', pad_inches=.7)
        plt.close('all')
        print(f'{utils.time_now()} - Receiver operating characteristic for alpha-{alpha}\'s cluster-{clus} -- printed')
    
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
    plt.savefig(f'{model.alpha_cluster_scoring_dir}/AUC-{auc_val:.3f}_ROCs_for_alpha_{alpha}_ALL_cluster.png', bbox_inches='tight', pad_inches=.7)
    plt.close('all')
    print(f'{utils.time_now()} - Receiver operating characteristic for alpha-{alpha}\ - ALL clusters. -- printed')
    
    time.sleep(1); gc.collect()
    utils.to_pickle(f'alpha_{alpha}_roc_auc', roc_auc, model.alpha_cluster_scoring_dir)
    utils.to_pickle(f'alpha_{alpha}_tprs_alpha', tprs_alpha, model.alpha_cluster_scoring_dir)
    utils.to_pickle(f'alpha_{alpha}_mean_tprs_alpha', mean_tprs_alpha, model.alpha_cluster_scoring_dir)

def ROC_all_alphas(model):
    all_tprs_alphas = [utils.open_pickle(i) for i in [*Path(model.tl_model.cluster_dir).glob(r'**/*/*/*/alpha_*_tprs_alpha*')] if 'mean' not in i.stem]

    plt.figure(figsize=(13, 12))
    base_fpr = np.linspace(0, 1, 101)
    plt.plot([0, 1], [0, 1],'r:')

    markercols = ('b', 'g', 'silver', 'm', 'c', 'y', 'w')

    # for each alpha
    for i, alpha in enumerate(all_tprs_alphas):
        mean_tprs_alpha = alpha.mean(axis=0)
        auc_ = auc(base_fpr, mean_tprs_alpha)
        plt.plot(base_fpr, mean_tprs_alpha, markercols[i], linestyle='-', lw=2, label=f'Alpha-{i+1}, AUC at {auc_:.3f}')

    # for ALL alphas
    all_tprs_alphas_combined = np.concatenate(all_tprs_alphas)
    alpha_sizes = [len(arr) for arr in all_tprs_alphas]
    tots = sum([len(arr) for arr in all_tprs_alphas])
    alph_weights = [len(arr)/tots for arr in all_tprs_alphas]
    weights = np.concatenate([np.full((alpha, ), alph_weights[i]) for i,alpha in enumerate(alpha_sizes)])
    mean_tprs_alpha = np.average(all_tprs_alphas_combined,axis=0,weights=weights)
    auc_ = auc(base_fpr, mean_tprs_alpha)

    # mean_tprs_alpha = all_tprs_alphas_combined.mean(axis=0)
    std = all_tprs_alphas_combined.std(axis=0)
    tprs_alpha_upper = np.minimum(mean_tprs_alpha + std, 1)
    tprs_alpha_lower = mean_tprs_alpha - std

    plt.plot(base_fpr, mean_tprs_alpha, 'k--.', lw=5, markersize=7, label=f'ROC weighted-averaged across all CV splits, AUC at {auc_:.3f}')

    plt.fill_between(base_fpr, tprs_alpha_lower, tprs_alpha_upper, color='lightblue', alpha=0.3, hatch='x')
    plt.suptitle(f'Receiver operating characteristic all alphas ' \
                , fontweight='bold', fontsize=20, y=0.94
            )
    plt.title('All CV true positive rates individually combined & micro-averaged.', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.xlim([-0.0, 1.0])
    plt.ylim([-0.0, 1.0])
    plt.legend()
    plt.savefig(f'{model.alpha_general_dir}/ROC_whole-model-mean_all-alphas_micro-avged', bbox_inches='tight', pad_inches=.7)
    plt.close('all')

def gridded_AUC_individual_alpha(model, alpha):
    clus_gt_ls_paths = utils.find('*clus_gt_*', model.alpha_cluster_dir)
    clus_pred_proba_ls_cluster_ls = utils.find('*clus_pred_proba_ls_cluster*', model.alpha_cluster_scoring_dir)

    if utils.find('*cluster*_date_to_prediction*', model.alpha_cluster_scoring_dir): pass
    else:
        for i, gt_path in enumerate(clus_gt_ls_paths):
            y_test = utils.open_pickle(gt_path)
            cluster_size = y_test.time.size
            shape = (y_test.lat.size, y_test.lon.size)
            
            gt_results = [y_test.isel(time=t).precipitationCal.T.values > 1 for t in range(cluster_size)]
            pred_results = np.ravel (utils.open_pickle(clus_pred_proba_ls_cluster_ls[i])) # y_train for cluster [i]
            date_to_prediction = np.array([[np.ravel(date), pred_results] for date in gt_results]) # [gt, pred] for THIS cluster
            utils.to_pickle(f'cluster_{i}_date_to_prediction', date_to_prediction, model.alpha_cluster_scoring_dir)
    
    date_to_prediction_path_ls = utils.find('*cluster*date_to_prediction.pkl', model.alpha_cluster_scoring_dir)
    date_to_prediction = np.concatenate([utils.open_pickle(i) for i in date_to_prediction_path_ls])

    tpr = {} # holds TPR in dict form for use in intermediate steps
    tprs_alpha = [] # holds interpolated TPRs for all clusters in this alpha, used to calc mean TPR for this alpha at the end
    fpr = {}; base_fpr = np.linspace(0, 1, 101); roc_auc = {}
    for grid in range(date_to_prediction.shape[-1]):
        fpr[grid], tpr[grid], _ = roc_curve( # for each sample of a grid
            date_to_prediction[:,0,grid], date_to_prediction[:,1,grid], pos_label=True, drop_intermediate=False
        ) 
        roc_auc[grid] = auc(fpr[grid], tpr[grid])
        tpr_ = np.interp(base_fpr, fpr[grid], tpr[grid])
        tpr_[0] = 0.0
        tprs_alpha.append(tpr_) 

    ds = utils.open_pickle([*Path(model.alpha_cluster_dir).glob('*RFprec_to_ClusterLabels_dataset*')][0])
    lon_coords = ds.lon.data
    lat_coords = ds.lat.data
    lon_coords_min = min(lon_coords)-.5
    lat_coords_min = min(lat_coords)-.5
    lon_coords_max = max(lon_coords)+.5
    lat_coords_max = max(lat_coords)+.5
    shape = (ds.lat.size, ds.lon.size)

    aucs = np.array([*roc_auc.values()])
    aucs = aucs.reshape(shape)

    print(f'{utils.time_now()} - Now plotting...')
    fig = plt.figure(figsize=(20,8))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(model.tl_model.lon_formatter)
    ax.yaxis.set_major_formatter(model.tl_model.lat_formatter)
    ax.add_feature(cf.COASTLINE, color='black', linewidth=1, linestyle='-') 
    ax.set_extent([lon_coords_min, lon_coords_max, lat_coords_min, lat_coords_max])
    levels=np.linspace(0,1,21)
    contf = ax.contourf(lon_coords, lat_coords, aucs, cmap="nipy_spectral", alpha=0.6, 
                        vmin=0, vmax=1.0, levels=levels)
    plt.title(f'AUC average for all dates in all clusters of alpha-1', fontweight='bold', fontsize=20, y=1.02)
    ax.set_xticks([min(lon_coords), 
                min(lon_coords)+(max(lon_coords)-min(lon_coords))/4, 
                min(lon_coords)+(max(lon_coords)-min(lon_coords))/2,
                min(lon_coords)+3*(max(lon_coords)-min(lon_coords))/4, 
                max(lon_coords)], crs=ccrs.PlateCarree())
    ax.set_yticks([min(lat_coords), 
                min(lat_coords)+(max(lat_coords)-min(lat_coords))/4, 
                min(lat_coords)+(max(lat_coords)-min(lat_coords))/2,
                min(lat_coords)+3*(max(lat_coords)-min(lat_coords))/4, 
                max(lat_coords)], crs=ccrs.PlateCarree())
    cbar = plt.colorbar(contf, shrink=0.7, ticks=np.linspace(0,1,11)[:-1])
    plt.savefig(f'{model.alpha_cluster_scoring_dir}/Gridded_AUC_individual_alpha_{alpha}_v2', bbox_inches='tight', pad_inches=.7)
    plt.close('all')

    utils.to_pickle(f'alpha_{alpha}_aucs', aucs, model.alpha_general_dir)

def get_CV_testsets_lengths(path):
    return np.array([len(utils.open_pickle(pkl)) for pkl in path])

def get_all_date_to_prediction(arr_sizes, date_to_prediction_shapes, date_to_prediction_path_ls, model):
    date_to_prediction = np.zeros((arr_sizes.sum(), 2, date_to_prediction_shapes[0][-1]))
    cur_empty_grid_index = 0
    for path_index, path in enumerate(date_to_prediction_path_ls):
        print(f'{utils.time_now()} - Now on path_index {path_index}')
        new_arr = utils.open_pickle(path)
        for i,arr in enumerate(new_arr):
            date_to_prediction[cur_empty_grid_index] = arr
            cur_empty_grid_index += 1
    return date_to_prediction
    #utils.to_pickle('ALL_date_to_prediction', date_to_prediction, model.alpha_general_dir)
    #print(f'"ALL_date_to_prediction.pkl" pickled!')

def gridded_AUC_all_alphas(model):
    # trying on 8Jan:
    cluster_sizes = np.array([utils.open_pickle(i) for i in [*Path(model.tl_model.cluster_dir).glob('**/**/*cluster_size*.pkl')]])
    tots = cluster_sizes.sum()

    alpha_sizes = np.array([i.sum() for i in np.array_split(cluster_sizes, model.ALPHAs)])
    alph_weights = alpha_sizes/tots
    weights = np.concatenate([np.full((alpha, ), alph_weights[i]) for i,alpha in enumerate(alpha_sizes)])
    print(weights.shape)

    all_alpha_aucs = np.array([np.array(utils.open_pickle(i)) for i in utils.find("*alpha_*_aucs.pkl", model.alpha_general_dir)])
    number_cv_splits = all_alpha_aucs.shape[0]
    all_alpha_aucs = np.mean(all_alpha_aucs, axis=0)
    
    # all_alpha_aucs = np.average(all_alpha_aucs, axis=0, weights=weights)
    
    # alpha_cluster_scoring_dirs_ls = [*Path(model.tl_model.cluster_dir).glob('**/*Evaluation*/*/')]
    # alpha_cluster_dirs_ls = [i.parent for i in Path(model.tl_model.cluster_dir).glob('**/*Evaluation*/')]

    # # i.e. all pred_proba for all clusters in ALL alphas
    # clus_pred_proba_ls_cluster_paths = [utils.find('*clus_pred_proba_ls_cluster*', i) for i in alpha_cluster_scoring_dirs_ls]
    # # alpha_cluster_scoring_dirs_ls_extended = [alpha_cluster_scoring_dirs_ls[i] \
    # #     for i,path in enumerate(clus_pred_proba_ls_cluster_paths) \
    # #     for num in range(len(path))]

    # # same for gt, all gt across all ALPHAs, cluster-membership irrelevant
    # clus_gt_ls_paths = [utils.find('*clus_gt_*', i) for i in alpha_cluster_dirs_ls]
    # # alpha_cluster_dirs_ls_extended = [alpha_cluster_dirs_ls[i] \
    # #     for i,path in enumerate(clus_gt_ls_paths) \
    # #     for num in range(len(path))]

    # clus_pred_proba_ls_cluster_paths = list(itertools.chain.from_iterable(clus_pred_proba_ls_cluster_paths))
    # clus_gt_ls_paths = list(itertools.chain.from_iterable(clus_gt_ls_paths))

    # alpha_gt_paths = [*Path(model.tl_model.cluster_dir).glob('**/alpha_*_GT*/')]
    # number_cv_splits = len(alpha_gt_paths)

    # # CV_testset_lengths = get_CV_testsets_lengths(
    # #     [*Path(model.tl_model.cluster_dir).glob('**/alpha_*_GT*/**/Rain_days_1mm_and_above/*clus_brier_scores_flat*')])

    # for i, gt_path in enumerate(clus_gt_ls_paths):
    #     if utils.find(f'*{i}_date_to_prediction.pkl', model.alpha_general_dir): 
    #         print(f'"{i}_date_to_prediction.pkl" already exists.'); continue
    #     print(f'Creating "{i}_date_to_prediction.pkl"....')
    #     y_test = utils.open_pickle(gt_path)
    #     cluster_size = y_test.time.size
    #     shape = (y_test.lat.size, y_test.lon.size)
    #     # grids = shape[0] * shape[1]
    
    #     gt_results = [y_test.isel(time=t).precipitationCal.T.values > 1 for t in range(cluster_size)]
    #     pred_results = np.ravel (utils.open_pickle(clus_pred_proba_ls_cluster_paths[i])) # y_train for cluster [i]
    #     date_to_prediction = np.array([[np.ravel(date), pred_results] for date in gt_results]) # [gt, pred] for THIS cluster
    #     # alpha_cluster_scoring_dir = alpha_cluster_scoring_dirs_ls_extended[i]
    #     utils.to_pickle(f'{i}_date_to_prediction', date_to_prediction, model.alpha_general_dir)

    # """
    # Below code deemed too memory-hungry. No need to pickle then, as it creates a copy instead of using numpy's view mode
    # """
    # # date_to_prediction_path_ls = utils.find('*date_to_prediction.pkl', model.alpha_general_dir)
    # # date_to_prediction = np.concatenate([utils.open_pickle(i) for i in date_to_prediction_path_ls])

    # date_to_prediction_path_ls = [i for i in utils.find(rf'*date_to_prediction.pkl', model.alpha_general_dir) if 'ALL_date_to_prediction.pkl' not in i]
    # date_to_prediction_shapes = np.array([utils.open_pickle(pkl).shape for pkl in date_to_prediction_path_ls])
    # arr_sizes = date_to_prediction_shapes[:,0]

    # """
    # FIXME: This method is still too memory-hungry for sizes >4k
    # """
    # #if utils.find('*ALL_date_to_prediction', model.alpha_general_dir): pass
    # #else:
    #     #print('"ALL_date_to_prediction.pkl" not found, generating now...')
    # date_to_prediction = get_all_date_to_prediction(arr_sizes, date_to_prediction_shapes, date_to_prediction_path_ls, model)

    # #date_to_prediction = utils.open_pickle(utils.find('*ALL_date_to_prediction', model.alpha_general_dir)[0])

    # # tpr = {} # holds TPR in dict form for use in intermediate steps
    # # tprs_alpha = [] # holds interpolated TPRs for all clusters in this alpha, used to calc mean TPR for this alpha at the end
    # # fpr = {}; 
    # # base_fpr = np.linspace(0, 1, 101)
    # roc_auc = {}

    # print(f'{utils.time_now()} - Generating confusion matrices for {date_to_prediction.shape[-1]} grids...')
    # for grid in range(date_to_prediction.shape[-1]):
    #     # fpr[grid], tpr[grid], _ = roc_curve( # for each sample of a grid
    #     #     date_to_prediction[:,0,grid], date_to_prediction[:,1,grid], pos_label=True, drop_intermediate=False
    #     # ) 
    #     f, t , _ = roc_curve( # for each sample of a grid
    #         date_to_prediction[:,0,grid], date_to_prediction[:,1,grid], pos_label=True, drop_intermediate=False
    #     ) 
    #     roc_auc[grid] = auc(f, t)
    #     # tpr_ = interp(base_fpr, f, t)
    #     # tpr_[0] = 0.0
    #     # tprs_alpha.append(tpr_)
        
    # ds = utils.open_pickle([*Path(model.tl_model.cluster_dir).glob('*RFprec_to_ClusterLabels_dataset*')][0])
    print(f'{utils.time_now()} - Taking coordinate data from {model.RFprec_to_ClusterLabels_dataset_path}')
    ds = utils.open_pickle(model.RFprec_to_ClusterLabels_dataset_path)
    lon_coords = ds.lon.data
    lat_coords = ds.lat.data
    lon_coords_min = min(lon_coords)-.5
    lat_coords_min = min(lat_coords)-.5
    lon_coords_max = max(lon_coords)+.5
    lat_coords_max = max(lat_coords)+.5
    shape = (ds.lat.size, ds.lon.size)

    # aucs = np.array([*roc_auc.values()])
    # aucs = aucs.reshape(shape)

    aucs = all_alpha_aucs.reshape(shape)

    print(f'{utils.time_now()} - Now plotting...')
    fig = plt.figure(figsize=(25,10))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(model.tl_model.lon_formatter)
    ax.yaxis.set_major_formatter(model.tl_model.lat_formatter)
    ax.add_feature(cf.COASTLINE, color='black', linewidth=1.4, linestyle='-') 
    ax.set_extent([lon_coords_min, lon_coords_max, lat_coords_min, lat_coords_max])
    levels=np.linspace(0,1,11)
    contf = ax.contourf(lon_coords, lat_coords, aucs, cmap="jet", alpha=1, 
                        vmin=0, vmax=1.0, levels=levels)

    conts = ax.contour(contf, 'k:', alpha=0.5, lw=.05)
    ax.clabel(conts, conts.levels, colors='k', inline=False, fontsize=7)
    
    plt.title(f'AUC averages for all dates in all clusters,\nacross all {number_cv_splits} cross-val test sets.', 
                fontweight='bold', fontsize=20, y=1.01)
    ax.set_xticks([min(lon_coords), 
                min(lon_coords)+(max(lon_coords)-min(lon_coords))/4, 
                min(lon_coords)+(max(lon_coords)-min(lon_coords))/2,
                min(lon_coords)+3*(max(lon_coords)-min(lon_coords))/4, 
                max(lon_coords)], crs=ccrs.PlateCarree())
    ax.set_yticks([min(lat_coords), 
                min(lat_coords)+(max(lat_coords)-min(lat_coords))/4, 
                min(lat_coords)+(max(lat_coords)-min(lat_coords))/2,
                min(lat_coords)+3*(max(lat_coords)-min(lat_coords))/4, 
                max(lat_coords)], crs=ccrs.PlateCarree())
    cbar = plt.colorbar(contf, shrink=0.85, ticks=np.linspace(0,1,11)[1:], pad=.02)
    cbar.set_label('AUC (1.0 is optimal, 0.5 is random, <0.5 indicates the model works worse than random guessing.)', labelpad=13)    
    plt.savefig(f'{model.alpha_general_dir}/gridded_AUC_whole-model_v2', bbox_inches='tight', pad_inches=.7)
    plt.close('all')
