"""
- loading of SOM products
- functions for generating intermediate plots (SOM model)
- functions for generating final output plots (RHUM, Quiver, AR, kmeans model, RF)
- loading of validation metrices
- functions for generating metrices plots (elbow/CH/DBI, sil plots, DBSCAN)
- misc plot creation functions
- functions for generation of evaluations on full-model
"""
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

mpl.rcParams['savefig.dpi'] = 300

logger = logging.getLogger()
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
print = logger.info
    
    
def grid_width(cluster_num, i=0): # 3 closest is 4 so return 2; 6 closest is 9 so return 3; 11 closest is 16 so return 4, etc.
    """
    Function to acquire appropriate (square) grid width for plotting.
    """
    while i**2 < cluster_num:
        i+=1; 
    return i


def create_multisubplot_axes(n_expected_clusters, width_height=12):
    """
    Returns fig object, width/height of figure based off n_expected_clusters, and gridspec created for fig obj. 
    Good for creating fig obj to use in Jupyter Notebook
    """
    fig = plt.figure(constrained_layout=False, figsize=(width_height, width_height))
    gw = grid_width(n_expected_clusters)
    gridspec = fig.add_gridspec(gw, gw)
    return fig, gridspec


def create_solo_figure(width_height=15):
    fig = plt.figure(figsize=(width_height, width_height))
    return fig, fig.add_subplot(111)


def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
        if nc > plt.get_cmap(cmap).N:
            raise ValueError("Too many categories for colormap.")
        if continuous:
            ccolors = plt.get_cmap(cmap)(np.linspace(0,1,nc))
        else:
            ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
        cols = np.zeros((nc*nsc, 3))
        for i, c in enumerate(ccolors):
            chsv = mpl.colors.rgb_to_hsv(c[:3])
            arhsv = np.tile(chsv,nsc).reshape(nsc,3)
            arhsv[:,1] = np.linspace(chsv[1],0.25,nsc)
            arhsv[:,2] = np.linspace(chsv[2],1,nsc)
            rgb = mpl.colors.hsv_to_rgb(arhsv)
            cols[i*nsc:(i+1)*nsc,:] = rgb       
        cmap = mpl.colors.ListedColormap(cols)
        return cmap

def get_meshgrid_xy(model):
    x = np.arange(model.gridsize)
    y = np.arange(model.gridsize)
    return [pt for pt in np.meshgrid(x,y)]

def print_som_scatterplot_with_dmap(model, dest):
    # n_datapoints, model.month_names, years, hyperparam_profile, 
    # mg1, mg2, dmap, winner_coordinates, target_ds, uniq_markers,
    # data_prof_save_dir, startlooptime, model.month_names_joined):
    ## plot 1: dmap + winner scatterplot, obtained via SOM
    
    som_splot_withdmap_starttime = timer(); print(f"{utils.time_now()} - Drawing SOM scatterplot with distance map now.")
    
    iterations, gridsize, training_mode, sigma, learning_rate, random_seed = model.hyperparameters
    fig, ax_dmap_splot = create_solo_figure()
    mg1, mg2 = get_meshgrid_xy(model)

    winner_coordinates = utils.open_pickle(model.winner_coordinates_path)
    dmap = utils.open_pickle(model.dmap_path)
    target_ds = utils.open_pickle(model.target_ds_preprocessed_path)
    
    # dmap underlay
    dmap_col = "summer_r"

    ax_dmap_splot.set_title(f"Plots for months: {model.month_names}, {model.sigma} sigma, {model.learning_rate} learning_rate, {model.random_seed} random_seeds\n{model.n_datapoints} input datapoints mapped onto SOM, {iterations} iters, overlaying inter-node distance map (in {dmap_col}).", loc='left')
    ax_dmap_splot.use_sticky_edges=False
    ax_dmap_splot.set_xticks([i for i in np.linspace(0, gridsize-1, gridsize)])
    ax_dmap_splot.set_yticks([i for i in np.linspace(0, gridsize-1, gridsize)])
    dmap_plot = ax_dmap_splot.pcolor(mg1, mg2,
                                     dmap, cmap=(cm.get_cmap(dmap_col, gridsize)),
                                     vmin=0, alpha=0.6)

    # winners scatterplot
    winners_scatterpoints = winner_coordinates + (np.random.random_sample((model.n_datapoints,2))-0.5)/1.2 
    markers = np.array([model.uniq_markers[month-1] for month in target_ds['time.month'].data]) # list of markers pertaining to this data subset
    colors = sns.color_palette("copper", len(model.years)) # colors
    cmap, norm = mpl.colors.from_levels_and_colors(range(0, len(model.years)+1), colors)
    row_to_colors_dict = {yr:colors[i] for i, yr in enumerate(model.years)} # {2001: (RGB), 2002: (RGB), ...}
    years_to_colors = [row_to_colors_dict[yr] for yr in target_ds['time.year'].data] # 2001-01-01: (RGB), 2001-01-02: ...

    plots_for_legend = []
    for marker in model.uniq_markers:
        mask = markers == marker
        if len(winners_scatterpoints[:,1][mask])>0:
            plots_for_legend.append(ax_dmap_splot.scatter(
                winners_scatterpoints[:,1][mask], 
                winners_scatterpoints[:,0][mask],  
                norm=norm, marker=marker, 
                c=np.array(years_to_colors)[mask], 
                s = 130,
                alpha=0.8, 
                linewidths=1))

    # colorbars for dmap & winners_scatterpoints s-plot
    axins_dmap = inset_axes(ax_dmap_splot, width='100%', height='100%',
                            loc='lower left', bbox_to_anchor=(0,-.05,.99,.015), 
                            bbox_transform=ax_dmap_splot.transAxes);
    cbar_dmap = fig.colorbar(dmap_plot, cax=axins_dmap,
                             label='Distance from other nodes (0.0 indicates a complete similarity to neighboring node)',
                             orientation='horizontal', pad=0.01);
    axins_splot = inset_axes(ax_dmap_splot, width='100%', height='100%', 
                             loc='lower left', bbox_to_anchor=(-.1, 0, .01, .99), 
                             bbox_transform=ax_dmap_splot.transAxes); # geometry & placement of cbar
    cbar_splot = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axins_splot,
                              ticks=[i+.5 for i in range(len(model.years))], orientation='vertical', pad=0.5);
    cbar_splot.ax.set_yticklabels(model.years); cbar_splot.ax.tick_params(size=3)

    ax_dmap_splot.legend(plots_for_legend, model.month_names, ncol=4, loc=9);

    print(f"Time taken is {utils.time_since(som_splot_withdmap_starttime)}\n")

    fig.subplots_adjust(wspace=0.05,hspace=0.3)
    fn = f"{dest}/{model.RUN_time}_{utils.time_now()}-{model.month_names_joined}_prelim_SOMscplot_{gridsize}x{gridsize}"
    fig.savefig(fn, bbox_inches='tight', pad_inches=1)
    print(f'file saved @:\n{fn}')
    plt.close('all')

def print_kmeans_scatterplot(model, dest, optimal_k):
            
    start_kmeanscatter = timer(); print(f"{utils.time_now()} - starting kmeans scatterplot now...")
    
    dmap = utils.open_pickle(model.dmap_path)
    labels_ar = utils.open_pickle(model.labels_ar_path)
    labels_to_coords = utils.open_pickle(model.labels_to_coords_path)
    label_markers = utils.open_pickle(model.label_markers_path)
    
    fig, ax_kmeans_dmap_splot = create_solo_figure()
    mg1, mg2 = get_meshgrid_xy(model)

    # dmap
    dmap_base_ax = fig.add_subplot(111)
    dmap_base_ax.set_xticks([i for i in np.linspace(0, model.gridsize-1, model.gridsize)])
    dmap_base_ax.set_yticks([i for i in np.linspace(0, model.gridsize-1, model.gridsize)])
    dmap_col = "CMRmap_r"
    dmap_plot = ax_kmeans_dmap_splot.pcolor(mg1, mg2, 
                                            dmap, cmap=(mpl.cm.get_cmap(dmap_col, model.gridsize)), 
                                            vmin=0, alpha=.7);
    dmap_plot.use_sticky_edges=False;
    axins_dmap = inset_axes(ax_kmeans_dmap_splot, width='100%', height='100%',
                            loc='lower left', bbox_to_anchor=(0,-.05,.99,.015), 
                            bbox_transform=ax_kmeans_dmap_splot.transAxes);
    cbar_dmap = fig.colorbar(dmap_plot, cax=axins_dmap,
                             label='Distance from other nodes (0.0 indicates a complete similarity to neighboring node)',
                             orientation='horizontal', pad=0.01);

    # scatterplot
    num_col, sub_col = (int(optimal_k/2),2) if (optimal_k%2==0) & (optimal_k>9) else (
        optimal_k, 1);
    c2 = categorical_cmap(13, 2, cmap="tab20c");
    y = minmax_scale(labels_ar)
    x = labels_to_coords
    colors = c2(y)

    ax_kmeans_dmap_splot.set_title('2nd clustering via K-means')
    for marker in np.unique(label_markers):
        mask = label_markers == marker
        if len(x[:,1][mask]) > 0:
            ax_kmeans_dmap_splot.scatter(
                x[:,1][mask],
                x[:,0][mask], 
                alpha=1, marker=marker, s=140, c=colors[mask], linewidths=1, edgecolors=None)
    ax_kmeans_dmap_splot.set_facecolor('black')
    ax_kmeans_dmap_splot.use_sticky_edges=False
    ax_kmeans_dmap_splot.margins(.07,.07)

    print(f"Time taken is {utils.time_since(start_kmeanscatter)}\n")

    fig.subplots_adjust(wspace=0.05,hspace=0.3)
    fn = f"{dest}/{model.RUN_time}_{utils.time_now()}-{model.month_names_joined}_kmeans-scplot_{model.gridsize}x{model.gridsize}"
    fig.savefig(fn, bbox_inches='tight', pad_inches=1)
    print(f'file saved @:\n{fn}')
    plt.close('all')

def print_ar_plot(model, dest, optimal_k):
    
    ARMonthFracstarttime = timer(); print(f"{utils.time_now()} - starting ar drawing now...")

    target_ds_withClusterLabels = utils.open_pickle(model.target_ds_withClusterLabels_path)
    target_ds_withClusterLabels = utils.remove_expver(target_ds_withClusterLabels)

    fig, gs_ARMonthFrac = create_multisubplot_axes(optimal_k)
    half_month_names = np.ravel([(f'{i} 1st-half', f'{i} 2nd-half') for i in model.month_names])
    c4 = categorical_cmap(8, 4, cmap="Dark2_r")
    color_indices = np.ravel([(2*(i-1), 2*(i-1)+1) for i in model.months])

    for i in range(optimal_k):
        ax_ARMonthFrac = fig.add_subplot(gs_ARMonthFrac[i])

        cluster_months = target_ds_withClusterLabels.where(target_ds_withClusterLabels.cluster==i, drop=True)['time.month']
        firsthalf = cluster_months[cluster_months['time.day']<=15]
        secondhalf = cluster_months[cluster_months['time.day']>15]
        firsthalf_counts = collections.Counter(firsthalf.data)
        secondhalf_counts = collections.Counter(secondhalf.data)
        halfmth_label_fraction = np.ravel([(firsthalf_counts[mth], secondhalf_counts[mth]) for mth in model.months])
        
        total_occurances = sum(halfmth_label_fraction)
        perc_of_total_sampling = np.round((total_occurances/model.n_datapoints)*100, 1)

        patches, text = ax_ARMonthFrac.pie(halfmth_label_fraction,
                                           radius = perc_of_total_sampling/100+.3,
                                           colors=c4(color_indices))

        ax_ARMonthFrac.annotate((f"Clust-{i+1},\n won {perc_of_total_sampling}% of rounds ({total_occurances}/{model.n_datapoints})."), (-.5,1))
        if i==model.grid_width-1: ax_ARMonthFrac.legend(half_month_names, bbox_to_anchor=(0, 1.3), ncol=4)

    print(f"Time taken is {utils.time_since(ARMonthFracstarttime)}\n")

    fig.subplots_adjust(wspace=0.05,hspace=0.3)
    fn = f"{dest}/{model.RUN_time}_{utils.time_now()}-{model.month_names_joined}_ARmonthfrac_{model.gridsize}x{model.gridsize}"
    fig.savefig(fn, bbox_inches='tight', pad_inches=1)
    print(f'file saved @:\n{fn}')
    plt.close('all')


def print_ar_plot_granular(model, dest, optimal_k):
    
    ARMonthFracstarttime = timer(); print(f"{utils.time_now()} - starting ar (granular) drawing now...")

    target_ds_withClusterLabels = utils.open_pickle(model.target_ds_withClusterLabels_path)
    target_ds_withClusterLabels = utils.remove_expver(target_ds_withClusterLabels)
    all_clusters = np.unique(target_ds_withClusterLabels.cluster)

    fig, axs = plt.subplots(len(model.months),1, figsize=(20,5*len(model.months)), sharey=True)
    fig.subplots_adjust(right=0.8, hspace=0.3)
    cbar_ax = fig.add_axes([0.81, 0.3, 0.025, 0.4])

    cmap = plt.cm.summer_r
    cmap.set_bad(color='thistle', alpha=0.2)

    for mth_i, ax in enumerate(axs):
        ds = target_ds_withClusterLabels.where(target_ds_withClusterLabels.time.dt.month==model.months[mth_i], drop=True)
        
        n_bins=31
        arr = np.empty([len(all_clusters), 31])
        extent = (0, arr.shape[1], arr.shape[0], 0)
        
        for i, clus in enumerate(all_clusters):
            days = ds.where(ds.cluster==clus, drop=True)['time.day']
            unique_days = list(np.unique(days))
            c = {}
            for num in unique_days:
                c[num] = int(days.where(days==num).count().values)
            arr[i] = np.array([c[d] if d in c.keys() else 0 for d in np.arange(1,32)])

        arr = np.ma.masked_where(arr==0, arr)
        im = ax.imshow(arr, cmap=cmap, extent=extent)

        ax.set_yticks(np.arange(len(all_clusters))+.5)
        ax.set_yticklabels(np.arange(len(all_clusters))+1)

        ax.set_xticks(np.arange(0,31)+.5)
        ax.set_xticklabels(np.arange(1,32))

        ax.tick_params(labelsize=14)
        ax.grid(False)
        ax.set_title(f'{model.month_names[mth_i]}', fontweight='bold', fontsize=20, y=1.02)
        
    fig.add_subplot(111,frameon=False)
    plt.ylabel('Cluster', fontsize=32, fontweight='bold')
    plt.xlabel('Day of month', fontsize=25, labelpad=40, fontweight='bold')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    cbar = fig.colorbar(im,cax=cbar_ax,
                        boundaries=np.arange(1, np.max(arr)+2), 
                        ticks=np.arange(1,np.max(arr)+5)+.5)
    cbar.set_label('Number occurance on particular day of month (days)', labelpad=20)
    cbar.ax.set_yticklabels(np.arange(np.max(arr), dtype=int)+1);

    plt.suptitle('Distribution of clusters for each month', fontweight='bold', x=.46, y=.95, fontsize=33)
    plt.title('Greyed-out/non-colored regions indicate 0 occurances on such dates for these clusters.', y=1.04, fontsize=15)

    print(f"Time taken is {utils.time_since(ARMonthFracstarttime)}\n")
    fn = f"{dest}/{model.RUN_time}_{utils.time_now()}-{model.month_names_joined}_ARmonthfrac_granular_{model.gridsize}x{model.gridsize}"
    fig.savefig(fn, bbox_inches='tight', pad_inches=1)
    print(f'file saved @:\n{fn}')
    plt.close('all')

def print_rf_mean_plots(model, dest, optimal_k):

    rfstarttime = timer(); print(f'{utils.time_now()} - Plotting MEAN rainfall now.\nTotal of {optimal_k} clusters, now printing cluster: ')

    RFprec_to_ClusterLabels_dataset = utils.open_pickle(model.RFprec_to_ClusterLabels_dataset_path)
    labels_ar = utils.open_pickle(model.labels_ar_path)
    
    fig, gs_rf_plot = create_multisubplot_axes(optimal_k)
    rf_ds_lon = RFprec_to_ClusterLabels_dataset.lon
    rf_ds_lat = RFprec_to_ClusterLabels_dataset.lat
    
    zero_to_ten = plt.cm.pink(np.linspace(1, .2, 3))
    eleven_to_25 = plt.cm.gist_earth(np.linspace(0.75, 0.2, 4))
    twnty5_to_40 = plt.cm.gist_rainbow(np.linspace(0.7, 0, 5))
    all_colors = np.vstack((zero_to_ten, eleven_to_25, twnty5_to_40))
    terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

    fig.suptitle(f'Mean rainfall (mm) over {model.dir_str}', fontweight='bold')
    
    for clus in range(len(np.unique(labels_ar))):
        time.sleep(1); gc.collect()
        data = RFprec_to_ClusterLabels_dataset.where(RFprec_to_ClusterLabels_dataset.cluster==clus, drop=True).precipitationCal.mean("time").T
        time.sleep(1); gc.collect()

        ax_rf_plot = fig.add_subplot(gs_rf_plot[clus], projection=ccrs.PlateCarree())
        ax_rf_plot.xaxis.set_major_formatter(model.lon_formatter)
        ax_rf_plot.yaxis.set_major_formatter(model.lat_formatter)
        ax_rf_plot.set_facecolor('white')
        ax_rf_plot.add_feature(cf.LAND, facecolor='black')
        ax_rf_plot.set_extent([model.LON_W-1, model.LON_E+1, model.LAT_S-1, model.LAT_N+1])
        ax_rf_plot.coastlines("50m", linewidth=.5, color='k')
        ax_rf_plot.add_feature(cf.BORDERS, linewidth=.5, color='k', linestyle='dashed')

        if clus < model.grid_width: # top ticks  
            ax_rf_plot.set_xticks([model.LON_W, (model.LON_E - model.LON_W)/2 + model.LON_W, model.LON_E], crs=ccrs.PlateCarree())
            ax_rf_plot.set_xticklabels([model.LON_W, (model.LON_E - model.LON_W)/2 + model.LON_W, model.LON_E], rotation=55)
            ax_rf_plot.xaxis.tick_top()
        else: ax_rf_plot.set_xticks([])

        if clus % model.grid_width == model.grid_width - 1: # right-side ticks
            ax_rf_plot.set_yticks([model.LAT_S, (model.LAT_N - model.LAT_S)/2 + model.LAT_S, model.LAT_N], crs=ccrs.PlateCarree())
            ax_rf_plot.yaxis.set_label_position("right")
            ax_rf_plot.yaxis.tick_right()
        else: ax_rf_plot.set_yticks([])

        if clus == 0: # title
            ax_rf_plot.set_title(f"Rainfall plots from SOM nodes,\ncluster no.{clus+1}", loc='left')
        else: ax_rf_plot.set_title(f"cluster no.{clus+1}", loc='left')
        
        RF = ax_rf_plot.contourf(rf_ds_lon, rf_ds_lat, data, np.linspace(0,50,51), 
        # cmap="terrain_r", 
        cmap=terrain_map, 
        extend='max')

        time.sleep(1); gc.collect()

        if clus == model.cbar_pos: # cbar
            axins_rf = inset_axes(ax_rf_plot, width='100%', height='100%',
                                  loc='lower left', bbox_to_anchor=(0, -.8, model.grid_width, .1),
                                  bbox_transform=ax_rf_plot.transAxes)
            cbar_rf = fig.colorbar(RF, cax=axins_rf, label='Rainfall (mm)', orientation='horizontal', pad=0.01)
            cbar_rf.ax.xaxis.set_ticks_position('top')
            cbar_rf.ax.xaxis.set_label_position('top')

        print(f'\n{utils.time_now()}: {clus}.. ');

    print(f"\n -- Time taken is {utils.time_since(rfstarttime)}\n")

    fig.subplots_adjust(wspace=0.05,hspace=0.3)
    fn = f"{dest}/{model.RUN_time}_{utils.time_now()}-{model.month_names_joined}_RFplot_mean_{model.gridsize}x{model.gridsize}"
    fig.savefig(fn, bbox_inches='tight', pad_inches=1)
    print(f'file saved @:\n{fn}')
    plt.close('all')

def print_rf_max_plots(model, dest, optimal_k):

    rfstarttime = timer(); print(f'{utils.time_now()} - Plotting MAX rainfall now.\nTotal of {optimal_k} clusters, now printing cluster: ')

    RFprec_to_ClusterLabels_dataset = utils.open_pickle(model.RFprec_to_ClusterLabels_dataset_path)
    labels_ar = utils.open_pickle(model.labels_ar_path)
    
    fig, gs_rf_plot = create_multisubplot_axes(optimal_k)
    rf_ds_lon = RFprec_to_ClusterLabels_dataset.lon
    rf_ds_lat = RFprec_to_ClusterLabels_dataset.lat
    
    # zero_to_ten = plt.cm.pink(np.linspace(1, .2, 3))
    # eleven_to_25 = plt.cm.gist_earth(np.linspace(0.75, 0.2, 5))
    # twnty5_to_40 = plt.cm.gist_stern(np.linspace(0.3, 0.1, 5))
    # all_colors = np.vstack((zero_to_ten, eleven_to_25, twnty5_to_40))
    # terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

    a = plt.cm.pink(np.linspace(.9, .2, 2))
    b = plt.cm.gnuplot2(np.linspace(0.4, .9, 6))
    all_colors = np.vstack((a,  b))
    terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

    fig.suptitle(f'MAX rainfall (mm) over individual grids for domain {model.dir_str}', fontweight='bold')
    
    for clus in range(len(np.unique(labels_ar))):
        time.sleep(1); gc.collect()
        data = RFprec_to_ClusterLabels_dataset.where(RFprec_to_ClusterLabels_dataset.cluster==clus, drop=True).precipitationCal.max("time").T
        data_gt1mm = np.ma.masked_where(data<=1, data)
        time.sleep(1); gc.collect()

        ax_rf_plot = fig.add_subplot(gs_rf_plot[clus], projection=ccrs.PlateCarree())
        ax_rf_plot.xaxis.set_major_formatter(model.lon_formatter)
        ax_rf_plot.yaxis.set_major_formatter(model.lat_formatter)
        ax_rf_plot.set_facecolor('white')
        ax_rf_plot.add_feature(cf.LAND, facecolor='silver')
        ax_rf_plot.set_extent([model.LON_W-1, model.LON_E+1, model.LAT_S-1, model.LAT_N+1])
        ax_rf_plot.coastlines("50m", linewidth=.5, color='k')
        ax_rf_plot.add_feature(cf.BORDERS, linewidth=.3, color='k', linestyle='dashed')

        if clus < model.grid_width: # top ticks  
            ax_rf_plot.set_xticks([model.LON_W, (model.LON_E - model.LON_W)/2 + model.LON_W, model.LON_E], crs=ccrs.PlateCarree())
            ax_rf_plot.set_xticklabels([model.LON_W, (model.LON_E - model.LON_W)/2 + model.LON_W, model.LON_E], rotation=55)
            ax_rf_plot.xaxis.tick_top()
        else: ax_rf_plot.set_xticks([])

        if clus % model.grid_width == model.grid_width - 1: # right-side ticks
            ax_rf_plot.set_yticks([model.LAT_S, (model.LAT_N - model.LAT_S)/2 + model.LAT_S, model.LAT_N], crs=ccrs.PlateCarree())
            ax_rf_plot.yaxis.set_label_position("right")
            ax_rf_plot.yaxis.tick_right()
        else: ax_rf_plot.set_yticks([])
        
        RF = ax_rf_plot.contourf(rf_ds_lon, rf_ds_lat, data_gt1mm, 
        np.arange(0,500,50),
        # np.linspace(0,450,16), 
        cmap=terrain_map, 
        extend='max')

        time.sleep(1); gc.collect()

        if clus == model.cbar_pos: # cbar
            axins_rf = inset_axes(ax_rf_plot, width='100%', height='100%',
                                  loc='lower left', bbox_to_anchor=(0, -.8, model.grid_width, .1),
                                  bbox_transform=ax_rf_plot.transAxes)
            cbar_rf = fig.colorbar(RF, cax=axins_rf, label='Rainfall (mm)', orientation='horizontal', pad=0.01,
                                  ticks=np.arange(0,500,50))
            cbar_rf.ax.xaxis.set_ticks_position('top')
            cbar_rf.ax.xaxis.set_label_position('top')

        print(f'\n{utils.time_now()}: {clus}.. ');

    print(f"\n -- Time taken is {utils.time_since(rfstarttime)}\n")

    fig.subplots_adjust(wspace=0.05,hspace=0.3)
    fn = f"{dest}/{model.RUN_time}_{utils.time_now()}-{model.month_names_joined}_RFplot_max_v2_{model.gridsize}x{model.gridsize}"
    fig.savefig(fn, bbox_inches='tight', pad_inches=1)
    print(f'file saved @:\n{fn}')
    plt.close('all')


def print_rf_rainday_gt1mm_plots(model, dest, optimal_k):

    rfstarttime = timer(); print(f'{utils.time_now()} - Plotting proba of >1mm rainfall now.\nTotal of {optimal_k} clusters, now printing cluster: ')

    RFprec_to_ClusterLabels_dataset = utils.open_pickle(model.RFprec_to_ClusterLabels_dataset_path)
    labels_ar = utils.open_pickle(model.labels_ar_path)
    
    fig, gs_rf_plot = create_multisubplot_axes(optimal_k)
    rf_ds_lon = RFprec_to_ClusterLabels_dataset.lon
    rf_ds_lat = RFprec_to_ClusterLabels_dataset.lat
    
    # pt1to3 = plt.cm.BrBG(np.linspace(0, .25, 3))
    # pt3to6 = plt.cm.gist_earth(np.linspace(0.75, 0.4, 5))
    # pt6to8 = plt.cm.ocean(np.linspace(.8, .3, 4))
    # all_colors = np.vstack((pt1to3, pt3to6, pt6to8)) 
    # terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

    # zero_to_ten = plt.cm.gist_stern(np.linspace(1, .2, 5))
    # eleven_to_25 = plt.cm.gnuplot2(np.linspace(.9, 0.25, 5))
    # twnty5_to_40 = plt.cm.gist_earth(np.linspace(0.15, 0.9, 8))
    # all_colors = np.vstack((zero_to_ten, eleven_to_25, twnty5_to_40))
    # terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

    a = plt.cm.YlOrRd(np.linspace(.9, .2, 5))
    b = plt.cm.YlGnBu(np.linspace(.2, .8, 10))
    all_colors = np.vstack((a,b))
    terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

    fig.suptitle(f'Proportion of grid with >1 mm of rainfall (raindays), over region: {model.domain[0]}S {model.domain[1]}N {model.domain[2]}W {model.domain[3]}E\n' \
        f'Note: regions in black indicate 0.0% chance of >1mm rainfall across grid members.', fontweight='bold')
    
    for clus in range(len(np.unique(labels_ar))):
        time.sleep(1); gc.collect()
        data = RFprec_to_ClusterLabels_dataset.where(RFprec_to_ClusterLabels_dataset.cluster==clus, drop=True).sel(
            lon=slice(model.LON_W, model.LON_E), lat=slice(model.LAT_S, model.LAT_N))
        mean = np.mean([data.isel(time=t).precipitationCal.T.values > 1 for t in range(data.time.size)], axis=0)
        data_pred_proba_morethan1mm = np.ma.masked_where(mean<=0, mean)*100
        time.sleep(1); gc.collect()

        ax_rf_plot = fig.add_subplot(gs_rf_plot[clus], projection=ccrs.PlateCarree())
        ax_rf_plot.xaxis.set_major_formatter(model.lon_formatter)
        ax_rf_plot.yaxis.set_major_formatter(model.lat_formatter)
        ax_rf_plot.set_facecolor('k')
        ax_rf_plot.set_extent([model.LON_W-1, model.LON_E+1, model.LAT_S-1, model.LAT_N+1])
        ax_rf_plot.coastlines("50m", linewidth=.7, color='w')
        ax_rf_plot.add_feature(cf.BORDERS, linewidth=.5, color='w', linestyle='dashed')

        if clus < model.grid_width: # top ticks  
            ax_rf_plot.set_xticks([model.LON_W, (model.LON_E - model.LON_W)/2 + model.LON_W, model.LON_E], crs=ccrs.PlateCarree())
            ax_rf_plot.set_xticklabels([model.LON_W, (model.LON_E - model.LON_W)/2 + model.LON_W, model.LON_E], rotation=55)
            ax_rf_plot.xaxis.tick_top()
        else: ax_rf_plot.set_xticks([])

        if clus % model.grid_width == model.grid_width - 1: # right-side ticks
            ax_rf_plot.set_yticks([model.LAT_S, (model.LAT_N - model.LAT_S)/2 + model.LAT_S, model.LAT_N], crs=ccrs.PlateCarree())
            ax_rf_plot.yaxis.set_label_position("right")
            ax_rf_plot.yaxis.tick_right()
        else: ax_rf_plot.set_yticks([])
        
        RF = ax_rf_plot.contourf(rf_ds_lon, rf_ds_lat, data_pred_proba_morethan1mm, 
        np.linspace(0,100,11), 
        cmap=terrain_map, 
        extend='neither')
        conts = ax_rf_plot.contour(RF, 'w', linewidths=0)
        ax_rf_plot.clabel(conts, conts.levels, colors='w', inline=True, fmt='%1.f', fontsize=8)

        time.sleep(1); gc.collect()

        if clus == model.cbar_pos: # cbar
            axins_rf = inset_axes(ax_rf_plot, width='100%', height='100%',
                                  loc='lower left', bbox_to_anchor=(0, -.8, model.grid_width, .1),
                                  bbox_transform=ax_rf_plot.transAxes)
            cbar_rf = fig.colorbar(RF, cax=axins_rf, label='Proportion of grid with >1 mm rainfall (%)', orientation='horizontal', pad=0.01, ticks=np.arange(0,100,10))
            cbar_rf.ax.xaxis.set_ticks_position('top')
            cbar_rf.ax.xaxis.set_label_position('top')

        print(f'\n{utils.time_now()}: {clus}.. ');

    print(f"\n -- Time taken is {utils.time_since(rfstarttime)}\n")

    fig.subplots_adjust(wspace=0.05,hspace=0.3)
    fn = f"{dest}/{model.RUN_time}_{utils.time_now()}-{model.month_names_joined}_RFplot_rainday_gt1mm_v3_{model.gridsize}x{model.gridsize}"
    fig.savefig(fn, bbox_inches='tight', pad_inches=1)
    print(f'file saved @:\n{fn}')
    plt.close('all')


def print_rf_heavyrainday_gt50mm_plots(model, dest, optimal_k):

    rfstarttime = timer(); print(f'{utils.time_now()} - Plotting proba of HEAVY (>50mm) rainfall now.\nTotal of {optimal_k} clusters, now printing cluster: ')

    RFprec_to_ClusterLabels_dataset = utils.open_pickle(model.RFprec_to_ClusterLabels_dataset_path)
    labels_ar = utils.open_pickle(model.labels_ar_path)
    
    fig, gs_rf_plot = create_multisubplot_axes(optimal_k)
    rf_ds_lon = RFprec_to_ClusterLabels_dataset.lon
    rf_ds_lat = RFprec_to_ClusterLabels_dataset.lat
    
    # pt1to3 = plt.cm.terrain(np.linspace(.7, .6, 3))
    # pt3to6 = plt.cm.gist_ncar(np.linspace(.4, 1, 5))
    # pt6to8 = plt.cm.ocean(np.linspace(.8, .4, 4))
    # all_colors = np.vstack((pt1to3, pt3to6, pt6to8))
    # terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

    # zero_to_ten = plt.cm.gist_stern(np.linspace(1, .2, 2))
    # eleven_to_25 = plt.cm.gnuplot2(np.linspace(.9, 0.25, 10))
    # twnty5_to_40 = plt.cm.gist_earth(np.linspace(0.15, 0.9, 8))
    # all_colors = np.vstack((zero_to_ten, eleven_to_25, twnty5_to_40))
    # terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)
    
    zero_to_ten = plt.cm.pink(np.linspace(1, .2, 3))
    eleven_to_25 = plt.cm.gist_earth(np.linspace(0.75, 0.2, 5))
    twnty5_to_40 = plt.cm.gist_stern(np.linspace(0.3, 0.1, 5))
    all_colors = np.vstack((zero_to_ten, eleven_to_25, twnty5_to_40))
    terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

    fig.suptitle(f'Proportion of grid with >50 mm of rainfall (heavy rain), over region: {model.domain[0]}S {model.domain[1]}N {model.domain[2]}W {model.domain[3]}E', fontweight='bold')
    
    for clus in range(len(np.unique(labels_ar))):
        time.sleep(1); gc.collect()
        data = RFprec_to_ClusterLabels_dataset.where(RFprec_to_ClusterLabels_dataset.cluster==clus, drop=True).sel(
            lon=slice(model.LON_W, model.LON_E), lat=slice(model.LAT_S, model.LAT_N))
        ddata_pred_proba_morethan50mm = np.mean([data.isel(time=t).precipitationCal.T.values > 50 for t in range(data.time.size)], axis=0)
        time.sleep(1); gc.collect()

        ax_rf_plot = fig.add_subplot(gs_rf_plot[clus], projection=ccrs.PlateCarree())
        ax_rf_plot.xaxis.set_major_formatter(model.lon_formatter)
        ax_rf_plot.yaxis.set_major_formatter(model.lat_formatter)
        ax_rf_plot.set_facecolor('white')
        ax_rf_plot.add_feature(cf.LAND, facecolor='silver')
        ax_rf_plot.set_extent([model.LON_W-1, model.LON_E+1, model.LAT_S-1, model.LAT_N+1])
        ax_rf_plot.coastlines("50m", linewidth=.3, color='k')
        ax_rf_plot.add_feature(cf.BORDERS, linewidth=.3, color='k', linestyle='dashed')

        if clus < model.grid_width: # top ticks  
            ax_rf_plot.set_xticks([model.LON_W, (model.LON_E - model.LON_W)/2 + model.LON_W, model.LON_E], crs=ccrs.PlateCarree())
            ax_rf_plot.set_xticklabels([model.LON_W, (model.LON_E - model.LON_W)/2 + model.LON_W, model.LON_E], rotation=55)
            ax_rf_plot.xaxis.tick_top()
        else: ax_rf_plot.set_xticks([])

        if clus % model.grid_width == model.grid_width - 1: # right-side ticks
            ax_rf_plot.set_yticks([model.LAT_S, (model.LAT_N - model.LAT_S)/2 + model.LAT_S, model.LAT_N], crs=ccrs.PlateCarree())
            ax_rf_plot.yaxis.set_label_position("right")
            ax_rf_plot.yaxis.tick_right()
        else: ax_rf_plot.set_yticks([])
        
        RF = ax_rf_plot.contourf(rf_ds_lon, rf_ds_lat, ddata_pred_proba_morethan50mm, np.linspace(0,1,101), 
        cmap=terrain_map, 
        extend='max')

        time.sleep(1); gc.collect()

        if clus == model.cbar_pos: # cbar
            axins_rf = inset_axes(ax_rf_plot, width='100%', height='100%',
                                  loc='lower left', bbox_to_anchor=(0, -.8, model.grid_width, .1),
                                  bbox_transform=ax_rf_plot.transAxes)
            cbar_rf = fig.colorbar(RF, cax=axins_rf, label='Proportion of grid with >50 mm rainfall (over 1)', orientation='horizontal', pad=0.01)
            cbar_rf.ax.xaxis.set_ticks_position('top')
            cbar_rf.ax.xaxis.set_label_position('top')

        print(f'\n{utils.time_now()}: {clus}.. ');

    print(f"\n -- Time taken is {utils.time_since(rfstarttime)}\n")

    fig.subplots_adjust(wspace=0.05,hspace=0.3)
    fn = f"{dest}/{model.RUN_time}_{utils.time_now()}-{model.month_names_joined}_RFplot_heavyrainday_gt50mm_v2_{model.gridsize}x{model.gridsize}"
    fig.savefig(fn, bbox_inches='tight', pad_inches=1)
    print(f'file saved @:\n{fn}')
    plt.close('all')


def print_rf_90th_percentile_plots(model, dest, optimal_k):

    rfstarttime = timer(); print(f'{utils.time_now()} - Plotting 90th of rainfall over grids now.\nTotal of {optimal_k} clusters, now printing cluster: ')

    RFprec_to_ClusterLabels_dataset = utils.open_pickle(model.RFprec_to_ClusterLabels_dataset_path)
    # labels_ar = utils.open_pickle(model.labels_ar_path)
    
    fig, gs_rf_plot = create_multisubplot_axes(optimal_k)
    rf_ds_lon = RFprec_to_ClusterLabels_dataset.lon
    rf_ds_lat = RFprec_to_ClusterLabels_dataset.lat

    # zero_to_ten = plt.cm.pink(np.linspace(1, .2, 3))
    # eleven_to_25 = plt.cm.gist_earth(np.linspace(0.75, 0.3, 5))
    # twnty5_to_40 = plt.cm.gnuplot2(np.linspace(0.4, .9, 5))
    # all_colors = np.vstack((zero_to_ten, eleven_to_25, twnty5_to_40))
    # terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

    z = plt.cm.gist_stern(np.linspace(1, .9, 1))
    a = plt.cm.terrain(np.linspace(0.6, .1, 4))
    b = plt.cm.gnuplot2(np.linspace(0.4, .9, 12))
    all_colors = np.vstack((z, a, b))
    terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

    fig.suptitle(f'90th percentile RF over region: {model.domain[0]}S {model.domain[1]}N {model.domain[2]}W {model.domain[3]}E', fontweight='bold')

    # for clus in range(len(np.unique(labels_ar))):
    for i, clus in enumerate([i for i in np.unique(RFprec_to_ClusterLabels_dataset.cluster) if not np.isnan(i)]):
        time.sleep(1); gc.collect()
        data = RFprec_to_ClusterLabels_dataset.where(RFprec_to_ClusterLabels_dataset.cluster==clus, drop=True).sel(
            lon=slice(model.LON_W, model.LON_E), lat=slice(model.LAT_S, model.LAT_N)).precipitationCal.values
        data_gt1mm = np.ma.masked_where(data<=1, data)
        percen_90 = np.percentile(data_gt1mm, 90, axis=0).T
        time.sleep(1); gc.collect()

        ax_rf_plot = fig.add_subplot(gs_rf_plot[i], projection=ccrs.PlateCarree())
        ax_rf_plot.xaxis.set_major_formatter(model.lon_formatter)
        ax_rf_plot.yaxis.set_major_formatter(model.lat_formatter)
        ax_rf_plot.set_facecolor('white')
        ax_rf_plot.add_feature(cf.LAND, facecolor='silver')
        ax_rf_plot.set_extent([model.LON_W-1, model.LON_E+1, model.LAT_S-1, model.LAT_N+1])
        ax_rf_plot.coastlines("50m", linewidth=.8, color='k')
        ax_rf_plot.add_feature(cf.BORDERS, linewidth=.5, color='k', linestyle='dashed')

        if clus < model.grid_width: # top ticks  
            ax_rf_plot.set_xticks([model.LON_W, (model.LON_E - model.LON_W)/2 + model.LON_W, model.LON_E], crs=ccrs.PlateCarree())
            ax_rf_plot.set_xticklabels([model.LON_W, (model.LON_E - model.LON_W)/2 + model.LON_W, model.LON_E], rotation=55)
            ax_rf_plot.xaxis.tick_top()
        else: ax_rf_plot.set_xticks([])

        if clus % model.grid_width == model.grid_width - 1: # right-side ticks
            ax_rf_plot.set_yticks([model.LAT_S, (model.LAT_N - model.LAT_S)/2 + model.LAT_S, model.LAT_N], crs=ccrs.PlateCarree())
            ax_rf_plot.yaxis.set_label_position("right")
            ax_rf_plot.yaxis.tick_right()
        else: ax_rf_plot.set_yticks([])
        
        RF = ax_rf_plot.contourf(rf_ds_lon, rf_ds_lat, percen_90, 
        # np.linspace(0,100,21), 
        np.arange(0,500,12.5),
        cmap=terrain_map, 
        extend='max')

        time.sleep(1); gc.collect()

        if clus == model.cbar_pos: # cbar
            axins_rf = inset_axes(ax_rf_plot, width='100%', height='100%',
                                  loc='lower left', bbox_to_anchor=(0, -.8, model.grid_width, .1),
                                  bbox_transform=ax_rf_plot.transAxes)
            cbar_rf = fig.colorbar(RF, cax=axins_rf, label='Rainfall (mm)', orientation='horizontal', pad=0.01,
            ticks=np.arange(0,500,50))
            cbar_rf.ax.xaxis.set_ticks_position('top')
            cbar_rf.ax.xaxis.set_label_position('top')

        print(f'\n{utils.time_now()}: {clus}.. ');

    print(f"\n -- Time taken is {utils.time_since(rfstarttime)}\n")

    fig.subplots_adjust(wspace=0.05,hspace=0.3)
    fn = f"{dest}/{model.RUN_time}_{utils.time_now()}-{model.month_names_joined}_RFplot_90th_percentile_v2_{model.gridsize}x{model.gridsize}"
    fig.savefig(fn, bbox_inches='tight', pad_inches=1)
    print(f'file saved @:\n{fn}')
    plt.close('all')


def print_quiver_plots(model, dest, optimal_k):
        
    quiverstarttime = timer(); print(f"{utils.time_now()} - Drawing quiver sub-plots now...")

    target_ds_withClusterLabels = utils.open_pickle(model.target_ds_withClusterLabels_path)
    target_ds_withClusterLabels = utils.remove_expver(target_ds_withClusterLabels)

    # skip_interval = 3
    # lon_qp = model.X[::skip_interval].values
    # lat_qp = model.Y[::skip_interval].values

    area = (model.LON_E-model.LON_W)*(model.LAT_N-model.LAT_S)
    coastline_lw = .8
    minshaft=2; scale=33
    if area > 3000: skip_interval=4
    elif 2000 < area <= 3000: skip_interval=3
    elif 500 < area <= 2000 : skip_interval=2; minshaft=3; scale=33
    else: skip_interval=1; minshaft=3; scale=33
    lon_qp = model.X[::skip_interval].values
    lat_qp = model.Y[::skip_interval].values


    for idx, pressure in enumerate(model.uwnd_vwnd_pressure_lvls):
        print(f'Currently on {pressure}hpa...')
        fig, gs_qp = create_multisubplot_axes(optimal_k)

        for cluster in range(optimal_k):
            print(f"{utils.time_now()} - Cluster {cluster}: ")
            
            uwnd_gridded_centroids = target_ds_withClusterLabels.sel(level=pressure).where(
                target_ds_withClusterLabels.cluster==cluster, drop=True).uwnd.mean(
                "time")[::skip_interval, ::skip_interval].values

            vwnd_gridded_centroids = target_ds_withClusterLabels.sel(level=pressure).where(
                target_ds_withClusterLabels.cluster==cluster, drop=True).vwnd.mean(
                "time")[::skip_interval, ::skip_interval].values

            ax_qp = fig.add_subplot(gs_qp[cluster], projection=ccrs.PlateCarree())
            ax_qp.xaxis.set_major_formatter(model.lon_formatter)
            ax_qp.yaxis.set_major_formatter(model.lat_formatter)
            ax_qp.set_facecolor('white')
            ax_qp.add_feature(cf.LAND,facecolor='silver')
            ax_qp.set_extent([model.LON_W-1, model.LON_E+1, model.LAT_S-1, model.LAT_N+1])

            if cluster < model.grid_width: # top ticks    
                ax_qp.set_xticks([model.LON_W, (model.LON_E - model.LON_W)/2 + model.LON_W, model.LON_E], crs=ccrs.PlateCarree())
                ax_qp.set_xticklabels([model.LON_W, (model.LON_E - model.LON_W)/2 + model.LON_W, model.LON_E], rotation=55)
                ax_qp.xaxis.tick_top()
            else: ax_qp.set_xticks([])

            if cluster % model.grid_width == model.grid_width-1: # right-side ticks
                ax_qp.set_yticks([model.LAT_S, (model.LAT_N - model.LAT_S)/2 + model.LAT_S, model.LAT_N], crs=ccrs.PlateCarree())
                ax_qp.yaxis.set_label_position("right")
                ax_qp.yaxis.tick_right()
            else: ax_qp.set_yticks([])

            if cluster == 0: # title
                ax_qp.set_title(f"Pressure: {pressure} hpa,\ncluster no.{cluster+1}", loc='left')
            else: ax_qp.set_title(f"cluster no.{cluster+1}", loc='left')
            
            time.sleep(1); gc.collect()
            wndspd = np.hypot(vwnd_gridded_centroids,uwnd_gridded_centroids); 
            time.sleep(1); gc.collect()
            u = uwnd_gridded_centroids/wndspd; 
            v = vwnd_gridded_centroids/wndspd; 
            spd_plot = ax_qp.contourf(lon_qp, lat_qp, wndspd, np.linspace(0,18,19), 
                                      transform=ccrs.PlateCarree(), cmap='terrain_r', 
                                      alpha=1)
            Quiver = ax_qp.quiver(lon_qp, lat_qp, u, v, color='Black', minshaft=minshaft, scale=scale)  
            conts = ax_qp.contour(spd_plot, 'w', linewidths=.3)
            ax_qp.coastlines("50m", linewidth=coastline_lw, color='orangered')
            ax_qp.add_feature(cf.BORDERS, linewidth=.5, color='k', linestyle='dashed')
            ax_qp.clabel(conts, conts.levels, inline=True, fmt='%1.f', fontsize=5)
            time.sleep(1); gc.collect()

            if cluster == model.cbar_pos: # cbar
                axins_qp = inset_axes(ax_qp, width='100%', height='100%',
                                      loc='lower left', bbox_to_anchor=(0, -.8, model.grid_width, .1),
                                      bbox_transform=ax_qp.transAxes)
                cbar_qp = fig.colorbar(spd_plot, cax=axins_qp, label='Quiver (m/s)', orientation='horizontal',pad=0.01)
                cbar_qp.ax.xaxis.set_ticks_position('top')
                cbar_qp.ax.xaxis.set_label_position('top')

        print(f"=> Quiver plots plotted for {pressure}hpa")   

        fig.subplots_adjust(wspace=0.05,hspace=0.3)
        fn = f"{dest}/{model.RUN_time}_{utils.time_now()}-{model.month_names_joined}_qp_v3-at-{pressure}hpa_{model.gridsize}x{model.gridsize}"
        fig.savefig(fn, bbox_inches='tight', pad_inches=1)
        print(f'file saved @:\n{fn}')
        plt.close('all')
        
    print(f"\n\nQuiver plotting took {utils.time_since(quiverstarttime)}.\n\n")

def print_rhum_plots(model, dest, optimal_k):
    
    rhumstarttime = timer(); print(f"{utils.time_now()} - Finishing RHUM plots...")

    target_ds_withClusterLabels = utils.open_pickle(model.target_ds_withClusterLabels_path)
    target_ds_withClusterLabels = utils.remove_expver(target_ds_withClusterLabels)

    for idx, pressure in enumerate(model.rhum_pressure_levels):
        fig, gs_rhum = create_multisubplot_axes(optimal_k)

        for cluster in range(optimal_k):
            rhum_gridded_centroids = target_ds_withClusterLabels.sel(level=pressure).where(
                target_ds_withClusterLabels.cluster==cluster, drop=True).rhum.mean("time")

            ax_rhum = fig.add_subplot(gs_rhum[cluster], projection=ccrs.PlateCarree())
            ax_rhum.xaxis.set_major_formatter(model.lon_formatter)
            ax_rhum.yaxis.set_major_formatter(model.lat_formatter)
            ax_rhum.coastlines("50m", linewidth=.7, color='w')
            ax_rhum.add_feature(cf.BORDERS, linewidth=.5, color='w', linestyle='dashed')
            ax_rhum.set_facecolor('white')
            ax_rhum.add_feature(cf.LAND, facecolor='k')
            ax_rhum.set_extent([model.LON_W-1, model.LON_E+1, model.LAT_S-1, model.LAT_N+1])

            if cluster < model.grid_width: # top ticks    
                ax_rhum.set_xticks([model.LON_W, (model.LON_E - model.LON_W)/2 + model.LON_W, model.LON_E], crs=ccrs.PlateCarree())
                ax_rhum.set_xticklabels([model.LON_W, (model.LON_E - model.LON_W)/2 + model.LON_W, model.LON_E], rotation=55)
                ax_rhum.xaxis.tick_top()
            else: ax_rhum.set_xticks([])

            if cluster % model.grid_width == model.grid_width-1: # right-side ticks
                ax_rhum.set_yticks([model.LAT_S, (model.LAT_N - model.LAT_S)/2 + model.LAT_S, model.LAT_N], crs=ccrs.PlateCarree())
                ax_rhum.yaxis.set_label_position("right")
                ax_rhum.yaxis.tick_right()
            else: ax_rhum.set_yticks([])

            if cluster == 0: # title
                ax_rhum.set_title(f"Pressure: {pressure} hpa,\ncluster no.{cluster+1}", loc='left')
            else: ax_rhum.set_title(f"cluster no.{cluster+1}", loc='left')

            normi = mpl.colors.Normalize(vmin=model.min_maxes['rhum_min'], vmax=model.min_maxes['rhum_max']);
            Rhum = ax_rhum.contourf(model.X, model.Y, rhum_gridded_centroids,
                                    np.linspace(model.min_maxes['rhum_min'], model.min_maxes['rhum_max'], 21),
                                    norm=normi, cmap='jet_r')
            conts = ax_rhum.contour(Rhum, 'k:', linewidths=.5)
            ax_rhum.clabel(conts, conts.levels, inline=True, fmt='%1.f', fontsize=10)


            if cluster == model.cbar_pos: # cbar
                axins_rhum = inset_axes(ax_rhum, width='100%', height='100%', 
                                        loc='lower left', bbox_to_anchor=(0, -.8, model.grid_width, .1), 
                                        bbox_transform=ax_rhum.transAxes);
                cbar_rhum = fig.colorbar(Rhum, cax=axins_rhum, label='Relative humidity (%)', orientation='horizontal', pad=0.01);
                cbar_rhum.ax.xaxis.set_ticks_position('top')
                cbar_rhum.ax.xaxis.set_label_position('top')

            print(f"{utils.time_now()} - clus {cluster}")

        print(f"==> Rhum plots plotted for {pressure}hpa")

        fig.subplots_adjust(wspace=0.05,hspace=0.3)
        fn = f"{dest}/{model.RUN_time}_{utils.time_now()}-{model.month_names_joined}_rhum_v3-at-{pressure}hpa_{model.gridsize}x{model.gridsize}"
        fig.savefig(fn, bbox_inches='tight', pad_inches=1)
        print(f'file saved @:\n{fn}')
        plt.close('all')

    print(f"\n\nTime taken to plot RHUM: {utils.time_since(rhumstarttime)}.")


def print_ind_clus_proportion_above_90thpercentile(model, dest, clus):

    print(f'{utils.time_now()} - Generating >90th percentile RF plot for clus: {clus+1}')

    RFprec_to_ClusterLabels_dataset = utils.open_pickle(model.RFprec_to_ClusterLabels_dataset_path)
    coarsened_clus_rf_ds = RFprec_to_ClusterLabels_dataset.precipitationCal.where(
        RFprec_to_ClusterLabels_dataset.cluster==clus, drop=True).coarsen(lat=5, lon=5, boundary='trim').max()

    RFprec_to_ClusterLabels_dataset_vals = utils.open_pickle(f'{model.full_rf_5Xcoarsened_vals_path}')
    percen90 = np.percentile(RFprec_to_ClusterLabels_dataset_vals, 90, axis=0)

    compared_clus_to_90percent = (coarsened_clus_rf_ds > percen90).values
    time_averaged_gridwise_RF_of_cluster_compared_to_90pcent = np.mean(compared_clus_to_90percent, axis=0)*100

    rf_ds_lon = coarsened_clus_rf_ds.lon
    rf_ds_lat = coarsened_clus_rf_ds.lat

    fig = plt.Figure(figsize=(12,15))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    fig.suptitle(f"Proportion of cluster {int(clus+1)} grid members receiving more RF \nthan the 90th percentile value of corresponding grid within full model",
                fontweight='bold', fontsize=15, y=.95, ha='center')
    ax.set_title(f"Total dates for each grid in this cluster: {compared_clus_to_90percent.shape[0]}", fontsize=14, y=1.03)

    ax.set_facecolor('w')
    ax.set_extent([model.LON_W-1, model.LON_E+1, model.LAT_S-1, model.LAT_N+1])
    ax.coastlines("50m", linewidth=.8, color='lightseagreen', alpha=1)
    ax.add_feature(cf.BORDERS, linewidth=.5, color='lightseagreen', linestyle='dashed')

    zero_to_ten = plt.cm.gist_stern(np.linspace(1, .2, 2))
    eleven_to_25 = plt.cm.gnuplot2(np.linspace(.9, 0.25, 10))
    twnty5_to_40 = plt.cm.gist_earth(np.linspace(0.15, 0.9, 8))
    all_colors = np.vstack((zero_to_ten, eleven_to_25, twnty5_to_40))
    terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

    RF = ax.contourf(rf_ds_lon, rf_ds_lat, 
        time_averaged_gridwise_RF_of_cluster_compared_to_90pcent.T,
        np.linspace(0,100,51), 
        alpha=1,
        cmap=terrain_map,
        extend='max')
    conts = ax.contour(RF, 'w', linewidths=.1)
    ax.clabel(conts, conts.levels, inline=True, fmt='%1.f', fontsize=8)

    cbar_rf = fig.colorbar(RF, label='Proportion of grid members receiving RF that exceeds 90th percentile of corresponding grid within full model (%)', orientation='horizontal', \
                            pad=0.05, shrink=.8, ticks=np.arange(0,100,10))
    cbar_rf.ax.xaxis.set_ticks_position('top')
    cbar_rf.ax.xaxis.set_label_position('top')
    ax.set_xticks(np.round(np.linspace(model.LON_W, model.LON_E, 10)), crs=ccrs.PlateCarree())
    ax.xaxis.tick_top()
    ax.set_xlabel('')

    ax.set_yticks(np.round(np.linspace(model.LAT_S, model.LAT_N, 10)), crs=ccrs.PlateCarree())
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel('')

    fn = f"{dest}/RF_proportion_above_90thpercentile_cluster_{int(clus+1)}.png"
    fig.savefig(fn, bbox_inches='tight', pad_inches=1)
    print(f'file saved @:\n{fn}')
    plt.close('all')


def print_ind_clus_proportion_under_10thpercentile(model, dest, clus):

    print(f'{utils.time_now()} - Generating <10th percentile RF plot for clus: {clus+1}')

    RFprec_to_ClusterLabels_dataset = utils.open_pickle(model.RFprec_to_ClusterLabels_dataset_path)
    coarsened_clus_rf_ds = RFprec_to_ClusterLabels_dataset.precipitationCal.where(
        RFprec_to_ClusterLabels_dataset.cluster==clus, drop=True).coarsen(lat=5, lon=5, boundary='trim').max()

    RFprec_to_ClusterLabels_dataset_vals = utils.open_pickle(f'{model.full_rf_5Xcoarsened_vals_path}')
    percen10 = np.percentile(RFprec_to_ClusterLabels_dataset_vals, 10, axis=0)

    compared_clus_to_10percent = (coarsened_clus_rf_ds < percen10).values
    time_averaged_gridwise_RF_of_cluster_compared_to_10pcent = np.mean(compared_clus_to_10percent, axis=0)*100

    rf_ds_lon = coarsened_clus_rf_ds.lon
    rf_ds_lat = coarsened_clus_rf_ds.lat

    fig = plt.Figure(figsize=(12,15))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    fig.suptitle(f"Proportion of cluster {int(clus+1)} grid members receiving less RF \nthan the 10th percentile value of corresponding grid within full model",
                fontweight='bold', fontsize=15, y=.95, ha='center')
    ax.set_title(f"Total dates for each grid in this cluster: {compared_clus_to_10percent.shape[0]}", fontsize=14, y=1.03)

    ax.set_facecolor('w')
    ax.set_extent([model.LON_W-1, model.LON_E+1, model.LAT_S-1, model.LAT_N+1])
    ax.coastlines("50m", linewidth=.8, color='lightseagreen', alpha=1)
    ax.add_feature(cf.BORDERS, linewidth=.5, color='lightseagreen', linestyle='dashed')

    zero_to_ten = plt.cm.gist_stern(np.linspace(1, .2, 2))
    eleven_to_25 = plt.cm.gnuplot2(np.linspace(.9, 0.25, 10))
    twnty5_to_40 = plt.cm.gist_earth(np.linspace(0.15, 0.9, 8))
    all_colors = np.vstack((zero_to_ten, eleven_to_25, twnty5_to_40))
    terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

    RF = ax.contourf(rf_ds_lon, rf_ds_lat, 
        time_averaged_gridwise_RF_of_cluster_compared_to_10pcent.T,
        np.linspace(0,100,51), 
        alpha=1,
        cmap=terrain_map,
        extend='max')
    conts = ax.contour(RF, 'w', linewidths=.1)
    ax.clabel(conts, conts.levels, inline=True, fmt='%1.f', fontsize=8)

    cbar_rf = fig.colorbar(RF, label='Proportion of grid members receiving RF that falls below the 10th percentile of corresponding grid within full model (%)', orientation='horizontal', \
                            pad=0.05, shrink=.8, ticks=np.arange(0,100,10))
    cbar_rf.ax.xaxis.set_ticks_position('top')
    cbar_rf.ax.xaxis.set_label_position('top')
    ax.set_xticks(np.round(np.linspace(model.LON_W, model.LON_E, 10)), crs=ccrs.PlateCarree())
    ax.xaxis.tick_top()
    ax.set_xlabel('')

    ax.set_yticks(np.round(np.linspace(model.LAT_S, model.LAT_N, 10)), crs=ccrs.PlateCarree())
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel('')

    fn = f"{dest}/RF_proportion_under_10thpercentile_cluster_{int(clus+1)}.png"
    fig.savefig(fn, bbox_inches='tight', pad_inches=1)
    print(f'file saved @:\n{fn}')
    plt.close('all')
    

def print_ind_clus_proportion_above_fullmodel_mean(model, dest, clus):

    print(f'{utils.time_now()} - Generating > mean RF plot for clus: {clus+1}')

    RFprec_to_ClusterLabels_dataset = utils.open_pickle(model.RFprec_to_ClusterLabels_dataset_path)
    coarsened_clus_rf_ds = RFprec_to_ClusterLabels_dataset.precipitationCal.where(
        RFprec_to_ClusterLabels_dataset.cluster==clus, drop=True).coarsen(lat=5, lon=5, boundary='trim').max()

    RFprec_to_ClusterLabels_dataset_vals = utils.open_pickle(f'{model.full_rf_5Xcoarsened_vals_path}')
    gridmean = np.mean(RFprec_to_ClusterLabels_dataset_vals, axis=0)

    compared_clus_to_gridmean = (coarsened_clus_rf_ds > gridmean).values
    time_averaged_gridwise_RF_of_cluster_compared_to_gridmean = np.mean(compared_clus_to_gridmean, axis=0)*100

    rf_ds_lon = coarsened_clus_rf_ds.lon
    rf_ds_lat = coarsened_clus_rf_ds.lat

    fig = plt.Figure(figsize=(12,15))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    fig.suptitle(f"Proportion of cluster {int(clus+1)} grid members receiving more RF \nthan the mean RF value of corresponding grid within full model",
                fontweight='bold', fontsize=15, y=.95, ha='center')
    ax.set_title(f"Total dates for each grid in this cluster: {compared_clus_to_gridmean.shape[0]}", fontsize=14, y=1.03)

    ax.set_facecolor('w')
    ax.set_extent([model.LON_W-1, model.LON_E+1, model.LAT_S-1, model.LAT_N+1])
    ax.coastlines("50m", linewidth=.8, color='lightseagreen', alpha=1)
    ax.add_feature(cf.BORDERS, linewidth=.5, color='lightseagreen', linestyle='dashed')

    zero_to_ten = plt.cm.gist_stern(np.linspace(1, .2, 2))
    eleven_to_25 = plt.cm.gnuplot2(np.linspace(.9, 0.25, 10))
    twnty5_to_40 = plt.cm.gist_earth(np.linspace(0.15, 0.9, 8))
    all_colors = np.vstack((zero_to_ten, eleven_to_25, twnty5_to_40))
    terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

    RF = ax.contourf(rf_ds_lon, rf_ds_lat, 
        time_averaged_gridwise_RF_of_cluster_compared_to_gridmean.T,
        np.linspace(0,100,51), 
        alpha=1,
        cmap=terrain_map,
        extend='max')
    conts = ax.contour(RF, 'w', linewidths=.1)
    ax.clabel(conts, conts.levels, inline=True, fmt='%1.f', fontsize=8)

    cbar_rf = fig.colorbar(RF, label='Proportion of grid members receiving RF above full model\'s grid-mean (%)', orientation='horizontal', \
                            pad=0.05, shrink=.8, ticks=np.arange(0,100,10))
    cbar_rf.ax.xaxis.set_ticks_position('top')
    cbar_rf.ax.xaxis.set_label_position('top')
    ax.set_xticks(np.round(np.linspace(model.LON_W, model.LON_E, 10)), crs=ccrs.PlateCarree())
    ax.xaxis.tick_top()
    ax.set_xlabel('')

    ax.set_yticks(np.round(np.linspace(model.LAT_S, model.LAT_N, 10)), crs=ccrs.PlateCarree())
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel('')

    fn = f"{dest}/RF_proportion_above_fullmodel_mean_cluster_{int(clus+1)}.png"
    fig.savefig(fn, bbox_inches='tight', pad_inches=1)
    print(f'file saved @:\n{fn}')
    plt.close('all')


def print_ind_clus_proportion_above_250mm(model, dest, clus):

    print(f'{utils.time_now()} - Generating > 250mm RF plot for clus: {clus+1}')

    RFprec_to_ClusterLabels_dataset = utils.open_pickle(model.RFprec_to_ClusterLabels_dataset_path)
    coarsened_clus_rf_ds = RFprec_to_ClusterLabels_dataset.precipitationCal.where(
        RFprec_to_ClusterLabels_dataset.cluster==clus, drop=True).coarsen(lat=5, lon=5, boundary='trim').max()

    compared_clus_to_250mm = (coarsened_clus_rf_ds > 250).values
    time_averaged_gridwise_RF_of_cluster_compared_to_250mm = np.mean(compared_clus_to_250mm, axis=0)*100
    time_averaged_gridwise_RF_of_cluster_compared_to_250mm = np.ma.masked_where(time_averaged_gridwise_RF_of_cluster_compared_to_250mm==0, time_averaged_gridwise_RF_of_cluster_compared_to_250mm)

    rf_ds_lon = coarsened_clus_rf_ds.lon
    rf_ds_lat = coarsened_clus_rf_ds.lat

    fig = plt.Figure(figsize=(12,15))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    fig.suptitle(f"Proportion of cluster {int(clus+1)} grid members receiving more than 250mm of RF in a day.",
                fontweight='bold', fontsize=14, y=.97, ha='center')
    ax.set_title(f"Total dates for each grid in this cluster: {compared_clus_to_250mm.shape[0]}\n" 
             "Note that all regions in grey have 0% of the grid members with >250mm of RF.", fontsize=13, y=1.04)

    ax.set_facecolor('silver')
    ax.set_extent([model.LON_W-1, model.LON_E+1, model.LAT_S-1, model.LAT_N+1])
    ax.coastlines("50m", linewidth=.8, color='lightseagreen', alpha=1)
    ax.add_feature(cf.BORDERS, linewidth=.5, color='lightseagreen', linestyle='dashed')

    terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', np.vstack(plt.cm.CMRmap(np.linspace(1,0,12))))

    RF = ax.contourf(rf_ds_lon, rf_ds_lat, 
        time_averaged_gridwise_RF_of_cluster_compared_to_250mm.T,
        np.linspace(0,20,11), 
        alpha=1,
        cmap=terrain_map,
        extend='max')

    cbar_rf = fig.colorbar(RF, label='Proportion of grid members receiving >250mm (%)', orientation='horizontal', \
                            pad=0.05, shrink=.8, ticks=np.arange(0,21,1))
    cbar_rf.ax.xaxis.set_ticks_position('top')
    cbar_rf.ax.xaxis.set_label_position('top')
    ax.set_xticks(np.round(np.linspace(model.LON_W, model.LON_E, 10)), crs=ccrs.PlateCarree())
    ax.xaxis.tick_top()
    ax.set_xlabel('')

    ax.set_yticks(np.round(np.linspace(model.LAT_S, model.LAT_N, 10)), crs=ccrs.PlateCarree())
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel('')

    fn = f"{dest}/RF_proportion_above_250mm_cluster_{int(clus+1)}.png"
    fig.savefig(fn, bbox_inches='tight', pad_inches=1)
    print(f'file saved @:\n{fn}')
    plt.close('all')
    

def get_domain_geometry(model, dest):
    lat_s_lim, lat_n_lim, lon_w_lim, lon_e_lim = model.domain_limits

    plt.figure(figsize=(8,10))
    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(model.lon_formatter)
    ax.yaxis.set_major_formatter(model.lat_formatter)
    ax.set_extent([lon_w_lim, lon_e_lim, lat_s_lim, lat_n_lim])
    ax.set_title(f'Map extent: longitudes {model.LON_W} to {model.LON_E}E, latitudes {model.LAT_S} to {model.LAT_N}N')
    geom = geometry.box(minx=model.LON_W, maxx=model.LON_E, miny=model.LAT_S, maxy=model.LAT_N)
    ax.add_geometries([geom], ccrs.PlateCarree(), alpha=0.3)
    ax.set_facecolor('silver')
    ax.add_feature(cf.LAND, facecolor='white')
    ax.coastlines('110m')
    ax.add_feature(cf.BORDERS, linewidth=.5, color='k', linestyle='dashed')
    ax.set_yticks(np.linspace(lat_s_lim, -lat_s_lim, 5), crs=ccrs.PlateCarree())
    ax.set_xticks(np.linspace(lon_w_lim, lon_e_lim, 6), crs=ccrs.PlateCarree())

    fn = f'{dest}/extent_{model.dir_str}.png'
    plt.savefig(fn)
    print(f'Extent saved @:\n{fn}')
    plt.close('all')


