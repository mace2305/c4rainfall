import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.colors as colors
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cartopy import feature as cf
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from shapely import geometry
from timeit import default_timer as timer
from sklearn.preprocessing import minmax_scale, RobustScaler
from sklearn.metrics import brier_score_loss
import collections, gc, time, logging

def print_test_date_abv_1mm_bool(model, dest, sn, random_sampled_date, cluster):           
    RFprec_to_ClusterLabels_dataset = utils.open_pickle(utils.find('*RFprec_to_ClusterLabels_dataset.pkl', model.test_prepared_data_dir)[0])
    date_split = pd.DatetimeIndex(random_sampled_date).strftime("%Y-%m-%d").values
    print(f'{utils.time_now()} - printing >1mm (boolean) plot for {date_split}')

    rf_random_choice = RFprec_to_ClusterLabels_dataset.sel(time=random_sampled_date).precipitationCal[0]
    rf_random_choice_gt1mm = rf_random_choice > 1

    rf_ds_lon = RFprec_to_ClusterLabels_dataset.lon
    rf_ds_lat = RFprec_to_ClusterLabels_dataset.lat   

    fig = plt.Figure(figsize=(12,15))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    fig.suptitle(f"Rainfall received on {date_split[0]} above 1mm",
                fontweight='bold', fontsize=16, y=.95, ha='center')
    ax.set_title(f"Predicted cluster: {cluster}", fontsize=14, y=1.05)

    ax.set_facecolor('w')
    ax.set_extent([model.LON_W-1, model.LON_E+1, model.LAT_S-1, model.LAT_N+1])
    ax.coastlines("50m", linewidth=.8, color='orangered', alpha=1)

    RF = ax.contourf(rf_ds_lon, rf_ds_lat, 
        rf_random_choice_gt1mm.T,
        alpha=1, levels=1,
        colors = [
    #          '#1f181b',
            "#ffffff",
            '#b3e0f2'],
        extend='neither')

    cbar_rf = fig.colorbar(RF, label='Grid received >1mm of RF (bool)', orientation='horizontal', \
                            pad=0.05, shrink=.8, ticks=[0,1])
    cbar_rf.ax.xaxis.set_ticks_position('top')
    cbar_rf.ax.xaxis.set_label_position('top')
    ax.set_xticks(np.round(np.linspace(model.LON_W, model.LON_E, 10)), crs=ccrs.PlateCarree())
    ax.xaxis.tick_top()
    ax.set_xlabel('')

    ax.set_yticks(np.round(np.linspace(model.LAT_S, model.LAT_N, 10)), crs=ccrs.PlateCarree())
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel('')
    
    fn = f'{dest}/{model.period}_{model.dir_str}_clus_{cluster}_test_abv1mm_rainday_sn{sn}_{date_split}.png'
    fig.savefig(fn, bbox_inches='tight', pad_inches=1)
    print(f'Extent saved @:\n{fn}')
    plt.close('all')

def print_test_date_abv_1mm(model, dest, sn, random_sampled_date, cluster):
           
    RFprec_to_ClusterLabels_dataset = utils.open_pickle(utils.find('*RFprec_to_ClusterLabels_dataset.pkl', model.test_prepared_data_dir)[0])
    date_split = pd.DatetimeIndex(random_sampled_date).strftime("%Y-%m-%d").values
    print(f'{utils.time_now()} - printing >1mm plot for {date_split}')

    rf_random_choice = RFprec_to_ClusterLabels_dataset.sel(time=random_sampled_date).precipitationCal[0]
    rf_random_choice_gt1mm = np.ma.masked_where(rf_random_choice<=1, rf_random_choice)

    rf_ds_lon = RFprec_to_ClusterLabels_dataset.lon
    rf_ds_lat = RFprec_to_ClusterLabels_dataset.lat   

    fig = plt.Figure(figsize=(12,15))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    fig.suptitle(f"RF received {date_split[0]} over 1mm",
                fontweight='bold', fontsize=16, y=.95)
    ax.set_title(f"Predicted cluster: {cluster}. \n"\
                f"Areas in grey denote 0.0-0.99mm RF, and considered as no rain occurred.", 
                fontsize=14, y=1.04)

    ax.set_facecolor('silver')
    ax.set_extent([model.LON_W-1, model.LON_E+1, model.LAT_S-1, model.LAT_N+1])
    ax.coastlines("50m", linewidth=.8, color='k',)

    a = plt.cm.pink(np.linspace(.9, .2, 2))
    b = plt.cm.gnuplot2(np.linspace(0.4, .9, 6))
    all_colors = np.vstack((a,  b))
    terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

    RF = ax.contourf(rf_ds_lon, rf_ds_lat, 
        rf_random_choice_gt1mm.T,
        np.linspace(0,500,501),
        cmap=terrain_map, 
        extend='max')

    cbar_rf = fig.colorbar(RF, label='RF (mm)', orientation='horizontal', \
                            pad=0.05, shrink=.8, ticks=np.arange(0,500,50))
    cbar_rf.ax.xaxis.set_ticks_position('top')
    cbar_rf.ax.xaxis.set_label_position('top')
    ax.set_xticks(np.round(np.linspace(model.LON_W, model.LON_E, 10)), crs=ccrs.PlateCarree())
    ax.xaxis.tick_top()
    ax.set_xlabel('')

    ax.set_yticks(np.round(np.linspace(model.LAT_S, model.LAT_N, 10)), crs=ccrs.PlateCarree())
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel('')
    
    fn = f'{dest}/{model.period}_{model.dir_str}_clus_{cluster}_test_abv1mm_to500_sn{sn}_{date_split}.png'
    fig.savefig(fn, bbox_inches='tight', pad_inches=1)
    print(f'Extent saved @:\n{fn}')
    plt.close('all')

def print_test_date_zscore_against_fullmodel(model, dest, sn, random_sampled_date, cluster):           

    RFprec_to_ClusterLabels_dataset = utils.open_pickle(utils.find('*RFprec_to_ClusterLabels_dataset.pkl', model.test_prepared_data_dir)[0])
    rf_ds_lon = RFprec_to_ClusterLabels_dataset.lon
    rf_ds_lat = RFprec_to_ClusterLabels_dataset.lat   
    date_split = pd.DatetimeIndex(random_sampled_date).strftime("%Y-%m-%d").values
    print(f'{utils.time_now()} - printing z-score plot for {date_split}')
    rf_random_choice = RFprec_to_ClusterLabels_dataset.sel(time=random_sampled_date).precipitationCal[0]

    training_rf_ds = utils.open_pickle(model.RFprec_to_ClusterLabels_dataset_path)       
    rf_for_random_choice_cluster = training_rf_ds.precipitationCal.where(training_rf_ds.cluster==cluster-1, drop=True)
    gridmean = np.mean(rf_for_random_choice_cluster, axis=0)
    gridstd = np.std(rf_for_random_choice_cluster, axis=0)
    stdardized_rf_random_choice = ((rf_random_choice-gridmean)/gridstd).values

    fig = plt.Figure(figsize=(12,15))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    fig.suptitle(f"Z-Score for {date_split[0]}. Predicted cluster: {cluster}",
                fontweight='bold', fontsize=18, y=.99, ha='center')
    ax.set_title(f"Total dates/datapoints per grid in cluster {cluster}: "\
                f"{training_rf_ds.where(training_rf_ds.cluster==cluster, drop=True).time.size}"\
                f"\n-1.65<=Z<=1.65 == 90%\n-1.96<=Z<=1.96 == 95%\n-2.58<=Z<=2.58 == 99%", 
                fontsize=14, y=1.04)

    ax.set_facecolor('w')
    ax.set_extent([model.LON_W-1, model.LON_E+1, model.LAT_S-1, model.LAT_N+1])
    ax.coastlines("50m", linewidth=.5, color='w', alpha=1)

    two58_to_196 = plt.cm.gist_ncar(np.linspace(.75, .8, 3))
    one96_to_0 = plt.cm.copper(np.linspace(1, 0, 4))
    zero_to_196 = plt.cm.twilight_shifted(np.linspace(0, .4, 4))
    one96_to_258 = plt.cm.gist_rainbow(np.linspace(.55, .3, 3))
    all_colors = np.vstack((two58_to_196, one96_to_0, zero_to_196, one96_to_258))
    terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

    RF = ax.contourf(rf_ds_lon, rf_ds_lat, 
        stdardized_rf_random_choice.T,
        np.linspace(-3.3333,3.3333,21), 
        alpha=1,
        cmap=terrain_map,
        extend='both')

    cbar_rf = fig.colorbar(RF, label='Z-Score of grid computed against grid-mean & grid-SD of whole model.', orientation='horizontal', \
                            pad=0.05, shrink=.9, ticks=[-2.58, -1.96, -1.65, 0, 1.65, 1.96, 2.58])
    cbar_rf.ax.tick_params(size=5)
    cbar_rf.ax.xaxis.set_ticks_position('top')
    cbar_rf.ax.xaxis.set_label_position('top')
    ax.set_xticks(np.round(np.linspace(model.LON_W, model.LON_E, 10)), crs=ccrs.PlateCarree())
    ax.xaxis.tick_top()
    ax.set_xlabel('')

    ax.set_yticks(np.round(np.linspace(model.LAT_S, model.LAT_N, 10)), crs=ccrs.PlateCarree())
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel('')
    
    fn = f'{dest}/{model.period}_{model.dir_str}_clus_{cluster}_test_zscore_against_fullmodel_sn{sn}_{date_split}.png'
    fig.savefig(fn, bbox_inches='tight', pad_inches=1)
    print(f'Extent saved @:\n{fn}')
    plt.close('all')


def print_brier_gt1mm(model, dest, sn, random_sampled_date, cluster):

    date_split = pd.DatetimeIndex(random_sampled_date).strftime("%Y-%m-%d").values
    print(f'{utils.time_now()} - printing Brier >1mm plot for {date_split}')

    test_ds = utils.open_pickle(utils.find('*RFprec_to_ClusterLabels_dataset.pkl', model.test_prepared_data_dir)[0])
    rf_ds_lon = test_ds.lon
    rf_ds_lat = test_ds.lat   
    test_ds_random_date = test_ds.sel(time=random_sampled_date)
    test_ds_random_date_gt1mm = test_ds_random_date.precipitationCal > 1
    gt_arr = test_ds_random_date_gt1mm[0].values

    training_rf_ds = utils.open_pickle(model.RFprec_to_ClusterLabels_dataset_path)
    pred = np.mean(training_rf_ds.where(
    training_rf_ds.cluster==cluster-1, drop=True).sel(
            lon=slice(model.LON_W, model.LON_E), lat=slice(model.LAT_S, model.LAT_N)).precipitationCal > 1, axis=0).values

    gt_arr_flat = np.reshape(gt_arr, (gt_arr.shape[0]*gt_arr.shape[1]))[:,None]
    pred_flat = np.reshape(pred, (pred.shape[0]*pred.shape[1]))[:,None]

    gridded_brier_flat = np.array([np.apply_along_axis(
        func1d=brier_score_loss, axis=0, arr=e, y_prob=f) for e, f in zip(gt_arr_flat,pred_flat)])
    gridded_brier = gridded_brier_flat.reshape(gt_arr.shape)

    clus_size = training_rf_ds.where(training_rf_ds.cluster==cluster, drop=True).time.size

    fig = plt.Figure(figsize=(12,15))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    fig.suptitle(f"Brier scores for {date_split[0]} compared to \npredicted forecast of rainday (>1mm) for cluster {cluster}.",
                fontweight='bold', fontsize=16, y=.98)

    ax.set_title(f'Number of training dates in cluster {cluster}: {clus_size}. \n' \
            'Scores approaching 0 indicate better calibrated predictive models \n' \
            'while 0.25 may represent forecasts of 50%, regardless of outcome.', y=1.04)

    ax.set_facecolor('w')
    ax.set_extent([model.LON_W-1, model.LON_E+1, model.LAT_S-1, model.LAT_N+1])
    ax.coastlines("50m", linewidth=.8, color='k',)

    a = plt.cm.summer(np.linspace(0, 1, 6))
    b = plt.cm.autumn(np.linspace(1, 0, 4))
    all_colors = np.vstack((a,  b))
    terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

    briers = ax.contourf(rf_ds_lon, rf_ds_lat, 
        gridded_brier.T,
        np.linspace(0,1,11),
        cmap=terrain_map, 
        extend='neither')

    cbar_rf = fig.colorbar(briers, label='Brier score', orientation='horizontal', \
                            pad=0.05, shrink=.8, ticks=np.arange(0,1.1,.1))
    cbar_rf.ax.xaxis.set_ticks_position('top')
    cbar_rf.ax.xaxis.set_label_position('top')
    ax.set_xticks(np.round(np.linspace(model.LON_W, model.LON_E, 10)), crs=ccrs.PlateCarree())
    ax.xaxis.tick_top()
    ax.set_xlabel('')

    ax.set_yticks(np.round(np.linspace(model.LAT_S, model.LAT_N, 10)), crs=ccrs.PlateCarree())
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel('')
    
    fn = f'{dest}/{model.period}_{model.dir_str}_clus_{cluster}_test_brier_gt1mm_sn{sn}_{date_split}.png'
    fig.savefig(fn, bbox_inches='tight', pad_inches=1)
    print(f'Extent saved @:\n{fn}')
    plt.close('all')
    

def print_brier_gt50mm(model, dest, sn, random_sampled_date, cluster):

    date_split = pd.DatetimeIndex(random_sampled_date).strftime("%Y-%m-%d").values
    print(f'{utils.time_now()} - printing Brier >50mm plot for {date_split}')

    test_ds = utils.open_pickle(utils.find('*RFprec_to_ClusterLabels_dataset.pkl', model.test_prepared_data_dir)[0])
    rf_ds_lon = test_ds.lon
    rf_ds_lat = test_ds.lat   
    test_ds_random_date = test_ds.sel(time=random_sampled_date)
    test_ds_random_date_gt1mm = test_ds_random_date.precipitationCal > 50
    gt_arr = test_ds_random_date_gt1mm[0].values

    training_rf_ds = utils.open_pickle(model.RFprec_to_ClusterLabels_dataset_path)
    pred = np.mean(training_rf_ds.where(
    training_rf_ds.cluster==cluster-1, drop=True).sel(
            lon=slice(model.LON_W, model.LON_E), lat=slice(model.LAT_S, model.LAT_N)).precipitationCal > 50, axis=0).values

    gt_arr_flat = np.reshape(gt_arr, (gt_arr.shape[0]*gt_arr.shape[1]))[:,None]
    pred_flat = np.reshape(pred, (pred.shape[0]*pred.shape[1]))[:,None]

    gridded_brier_flat = np.array([np.apply_along_axis(
        func1d=brier_score_loss, axis=0, arr=e, y_prob=f) for e, f in zip(gt_arr_flat,pred_flat)])
    gridded_brier = gridded_brier_flat.reshape(gt_arr.shape)

    clus_size = training_rf_ds.where(training_rf_ds.cluster==cluster, drop=True).time.size

    fig = plt.Figure(figsize=(12,15))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    fig.suptitle(f"Brier scores for {date_split[0]} compared to \npredicted forecast of heavy (>50mm) for cluster {cluster}.",
        fontweight='bold', fontsize=16, y=.98)

    ax.set_title(f'Number of training dates in cluster {cluster}: {clus_size}. \n' \
            'Scores approaching 0 indicate better calibrated predictive models \n' \
            'while 0.25 may represent forecasts of 50%, regardless of outcome.', y=1.04)

    ax.set_facecolor('w')
    ax.set_extent([model.LON_W-1, model.LON_E+1, model.LAT_S-1, model.LAT_N+1])
    ax.coastlines("50m", linewidth=.8, color='k',)

    a = plt.cm.summer(np.linspace(0, 1, 6))
    b = plt.cm.autumn(np.linspace(1, 0, 4))
    all_colors = np.vstack((a,  b))
    terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

    briers = ax.contourf(rf_ds_lon, rf_ds_lat, 
        gridded_brier.T,
        np.linspace(0,1,11),
        cmap=terrain_map, 
        extend='neither')

    cbar_rf = fig.colorbar(briers, label='Brier score', orientation='horizontal', \
                            pad=0.05, shrink=.8, ticks=np.arange(0,1.1,.1))
    cbar_rf.ax.xaxis.set_ticks_position('top')
    cbar_rf.ax.xaxis.set_label_position('top')
    ax.set_xticks(np.round(np.linspace(model.LON_W, model.LON_E, 10)), crs=ccrs.PlateCarree())
    ax.xaxis.tick_top()
    ax.set_xlabel('')

    ax.set_yticks(np.round(np.linspace(model.LAT_S, model.LAT_N, 10)), crs=ccrs.PlateCarree())
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel('')
    
    fn = f'{dest}/{model.period}_{model.dir_str}_clus_{cluster}_test_brier_gt50mm_sn{sn}_{date_split}.png'
    fig.savefig(fn, bbox_inches='tight', pad_inches=1)
    print(f'Extent saved @:\n{fn}')
    plt.close('all')
