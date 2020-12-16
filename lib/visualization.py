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
import cartopy.crs as ccrs
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cartopy import feature as cf
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from timeit import default_timer as timer
from sklearn.preprocessing import minmax_scale, RobustScaler
import collections, gc, time, logging

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

def print_som_scatterplot_with_dmap(model):
    # n_datapoints, model.month_names, years, hyperparam_profile, 
    # mg1, mg2, dmap, winner_coordinates, target_ds, uniq_markers,
    # data_prof_save_dir, startlooptime, model.month_names_joined):
    ## plot 1: dmap + winner scatterplot, obtained via SOM
    
    som_splot_withdmap_starttime = timer(); print("Drawing SOM scatterplot with distance map now.")
    
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
    fn = f"{model.cluster_dir}/{model.RUN_time}_{utils.time_now()}-{model.month_names_joined}_prelim_SOMscplot_{gridsize}x{gridsize}"
    fig.savefig(fn, bbox_inches='tight', pad_inches=1)
    print(f'file saved @:\n{fn}')
    plt.close('all')

def print_kmeans_scatterplot(model):
            
    start_kmeanscatter = timer(); print("\nstarting kmeans scatterplot now...")
    
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
    num_col, sub_col = (int(model.optimal_k/2),2) if (model.optimal_k%2==0) & (model.optimal_k>9) else (
        model.optimal_k, 1);
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
    fn = f"{model.cluster_dir}/{model.RUN_time}_{utils.time_now()}-{model.month_names_joined}_kmeans-scplot_{model.gridsize}x{model.gridsize}"
    fig.savefig(fn, bbox_inches='tight', pad_inches=1)
    print(f'file saved @:\n{fn}')
    plt.close('all')

def print_ar_plot(model):
    
    ARMonthFracstarttime = timer(); print("\nstarting ar drawing now...")

    target_ds_withClusterLabels = utils.open_pickle(model.target_ds_withClusterLabels_path)
    target_ds_withClusterLabels = utils.remove_expver(target_ds_withClusterLabels)

    fig, gs_ARMonthFrac = create_multisubplot_axes(model.optimal_k)
    half_month_names = np.ravel([(f'{i} 1st-half', f'{i} 2nd-half') for i in model.month_names])
    c4 = categorical_cmap(8, 4, cmap="Dark2_r")
    color_indices = np.ravel([(2*(i-1), 2*(i-1)+1) for i in model.months])

    for i in range(model.optimal_k):
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
    fn = f"{model.cluster_dir}/{model.RUN_time}_{utils.time_now()}-{model.month_names_joined}_ARmonthfrac_{model.gridsize}x{model.gridsize}"
    fig.savefig(fn, bbox_inches='tight', pad_inches=1)
    print(f'file saved @:\n{fn}')
    plt.close('all')

#FIXME
def print_ar_plot_granular(model):
    pass

def print_rf_plots(model):

    rfstarttime = timer(); print(f'{utils.time_now()} - Plotting rainfall now.\nTotal of {model.optimal_k} clusters, now printing cluster: ')

    RFprec_to_ClusterLabels_dataset = utils.open_pickle(model.RFprec_to_ClusterLabels_dataset_path)
    labels_ar = utils.open_pickle(model.labels_ar_path)
    
    fig, gs_rf_plot = create_multisubplot_axes(model.optimal_k)
    rf_ds_lon = RFprec_to_ClusterLabels_dataset.lon
    rf_ds_lat = RFprec_to_ClusterLabels_dataset.lat
    
    for clus in range(len(np.unique(labels_ar))):
        time.sleep(1); gc.collect()
        data = RFprec_to_ClusterLabels_dataset.where(RFprec_to_ClusterLabels_dataset.cluster==clus, drop=True).precipitationCal.mean("time").T
        time.sleep(1); gc.collect()

        ax_rf_plot = fig.add_subplot(gs_rf_plot[clus], projection=ccrs.PlateCarree())
        ax_rf_plot.xaxis.set_major_formatter(model.lon_formatter)
        ax_rf_plot.yaxis.set_major_formatter(model.lat_formatter)
        ax_rf_plot.set_facecolor('white')
        ax_rf_plot.add_feature(cf.LAND, facecolor='silver')
        ax_rf_plot.set_extent([model.LON_W-2, model.LON_E+2, model.LAT_S-1, model.LAT_N+1])
        ax_rf_plot.coastlines("10m")

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

        RF = ax_rf_plot.contourf(rf_ds_lon, rf_ds_lat, data, np.linspace(0,40,9), cmap="terrain_r", extend='max')
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
    fn = f"{model.cluster_dir}/{model.RUN_time}_{utils.time_now()}-{model.month_names_joined}_RFplot_{model.gridsize}x{model.gridsize}"
    fig.savefig(fn, bbox_inches='tight', pad_inches=1)
    print(f'file saved @:\n{fn}')
    plt.close('all')


def print_quiver_plots(model):
        
    quiverstarttime = timer(); print("\n\nDrawing quiver sub-plots now...")

    target_ds_withClusterLabels = utils.open_pickle(model.target_ds_withClusterLabels_path)
    target_ds_withClusterLabels = utils.remove_expver(target_ds_withClusterLabels)

    skip_interval = 3
    lon_qp = model.X[::skip_interval].values
    lat_qp = model.Y[::skip_interval].values

    for idx, pressure in enumerate(model.uwnd_vwnd_pressure_lvls):
        print(f'Currently on {pressure}hpa...')
        fig, gs_qp = create_multisubplot_axes(model.optimal_k)

        for cluster in range(model.optimal_k):
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
            ax_qp.set_extent([model.LON_W-2, model.LON_E+2, model.LAT_S-1, model.LAT_N+1])
            ax_qp.coastlines("110m")

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
            
            print(f"{utils.time_now()} Beginning contourf & quiver plots... ")
            time.sleep(1); gc.collect()
            wndspd = np.hypot(vwnd_gridded_centroids,uwnd_gridded_centroids); print('CP1')
            time.sleep(1); gc.collect()
            u = uwnd_gridded_centroids/wndspd; print('CP2')
            v = vwnd_gridded_centroids/wndspd; print('CP3')
            spd_plot = ax_qp.contourf(lon_qp, lat_qp, wndspd, np.linspace(0,18,10), 
                                      transform=ccrs.PlateCarree(), cmap='terrain_r', 
                                      alpha=0.55)
            print('CP4')
            time.sleep(1); gc.collect()
            Quiver = ax_qp.quiver(lon_qp, lat_qp, u, v, color='Black', minshaft=2, scale=20)  
            print('CP5..! ')
            time.sleep(1); gc.collect()

            if cluster == model.cbar_pos: # cbar
                axins_qp = inset_axes(ax_qp, width='100%', height='100%',
                                      loc='lower left', bbox_to_anchor=(0, -.8, model.grid_width, .1),
                                      bbox_transform=ax_qp.transAxes)
                cbar_qp = fig.colorbar(spd_plot, cax=axins_qp, label='Quiver (m/s)', orientation='horizontal',pad=0.01)
                cbar_qp.ax.xaxis.set_ticks_position('top')
                cbar_qp.ax.xaxis.set_label_position('top')

        print(f"\n\nQuiver plots plotted for {pressure}hpa")   

        fig.subplots_adjust(wspace=0.05,hspace=0.3)
        fn = f"{model.cluster_dir}/{model.RUN_time}_{utils.time_now()}-{model.month_names_joined}_qp-at-{pressure}hpa_{model.gridsize}x{model.gridsize}"
        fig.savefig(fn, bbox_inches='tight', pad_inches=1)
        print(f'file saved @:\n{fn}')
        plt.close('all')
        
    print(f"\n\nQuiver plotting took {utils.time_since(quiverstarttime)}.\n\n")

def print_rhum_plots(model):
    
    rhumstarttime = timer(); print("Finishing RHUM plots...")

    target_ds_withClusterLabels = utils.open_pickle(model.target_ds_withClusterLabels_path)
    target_ds_withClusterLabels = utils.remove_expver(target_ds_withClusterLabels)

    for idx, pressure in enumerate(model.rhum_pressure_levels):
        fig, gs_rhum = create_multisubplot_axes(model.optimal_k)

        for cluster in range(model.optimal_k):
            rhum_gridded_centroids = target_ds_withClusterLabels.sel(level=pressure).where(
                target_ds_withClusterLabels.cluster==cluster, drop=True).rhum.mean("time")

            print('Acquiring rhum cluster mean...')

            ax_rhum = fig.add_subplot(gs_rhum[cluster], projection=ccrs.PlateCarree())
            ax_rhum.xaxis.set_major_formatter(model.lon_formatter)
            ax_rhum.yaxis.set_major_formatter(model.lat_formatter)
            ax_rhum.set_facecolor('white')
            ax_rhum.add_feature(cf.LAND, facecolor='silver')
            ax_rhum.set_extent([model.LON_W - 2, model.LON_E + 2, model.LAT_S - 1, model.LAT_N + 1])
            ax_rhum.coastlines("50m")

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

            print("Plotting contourf now.")
            normi = mpl.colors.Normalize(vmin=model.min_maxes['rhum_min'], vmax=model.min_maxes['rhum_max'])
            Rhum = ax_rhum.contourf(model.X, model.Y, rhum_gridded_centroids,
                                    np.linspace(model.min_maxes['rhum_min'], model.min_maxes['rhum_max'], 21),
                                    norm=normi, extend='both', cmap='RdBu', alpha=0.7)

            if cluster == model.cbar_pos: # cbar
                axins_rhum = inset_axes(ax_rhum, width='100%', height='100%', 
                                        loc='lower left', bbox_to_anchor=(0, -.8, model.grid_width, .1), 
                                        bbox_transform=ax_rhum.transAxes)
                cbar_rhum = fig.colorbar(Rhum, cax=axins_rhum, label='Relative humidity (%)', orientation='horizontal', pad=0.01)
                cbar_rhum.ax.xaxis.set_ticks_position('top')
                cbar_rhum.ax.xaxis.set_label_position('top')

            print(f"{utils.time_now()} - clus {cluster}")

        print(f"\n\nRhum plots plotted for {pressure}hpa")

        fig.subplots_adjust(wspace=0.05,hspace=0.3)
        fn = f"{model.cluster_dir}/{model.RUN_time}_{utils.time_now()}-{model.month_names_joined}_rhum-at-{pressure}hpa_{model.gridsize}x{model.gridsize}"
        fig.savefig(fn, bbox_inches='tight', pad_inches=1)
        print(f'file saved @:\n{fn}')
        plt.close('all')

    print(f"\n\nTime taken to plot RHUM: {utils.time_since(rhumstarttime)}.")
    



























