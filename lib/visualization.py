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
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from timeit import default_timer as timer
import matplotlib
    
    
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
    fig = plt.figure(constrained_layout=False, figsize=(width_height,width_height))
    mini_gsize = grid_width(n_expected_clusters)
    gs_rf = fig.add_gridspec(mini_gsize,mini_gsize)
    return fig, mini_gsize, gs_rf


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
            chsv = matplotlib.colors.rgb_to_hsv(c[:3])
            arhsv = np.tile(chsv,nsc).reshape(nsc,3)
            arhsv[:,1] = np.linspace(chsv[1],0.25,nsc)
            arhsv[:,2] = np.linspace(chsv[2],1,nsc)
            rgb = matplotlib.colors.hsv_to_rgb(arhsv)
            cols[i*nsc:(i+1)*nsc,:] = rgb       
        cmap = matplotlib.colors.ListedColormap(cols)
        return cmap
    

def print_som_scatterplot_with_dmap(model):
    # n_datapoints, model.month_names, years, hyperparam_profile, 
    # mg1, mg2, dmap, winner_coordinates, target_ds, uniq_markers,
    # data_prof_save_dir, startlooptime, model.month_names_joined):
    ## plot 1: dmap + winner scatterplot, obtained via SOM
    iterations, gridsize, training_mode, sigma, learning_rate, random_seed = model.hyperparameters
    fig, ax_dmap_splot = create_solo_figure()

    winner_coordinates = utils.open_pickle(model.winner_coordinates_path)
    dmap = utils.open_pickle(model.dmap_path)
    target_ds = utils.open_pickle(model.target_ds_preprocessed_path)
    x=np.arange(model.gridsize); y=np.arange(model.gridsize); mg1, mg2 = [pt for pt in np.meshgrid(x,y)]
    
    # dmap underlay
    dmap_col = "summer_r"

    ax_dmap_splot.set_title(f"Plots for months: {model.month_names}, {model.sigma} sigma, {model.learning_rate} learning_rate, {model.random_seed} random_seeds\n{model.n_datapoints} input datapoints mapped onto SOM, {iterations} iters, overlaying inter-node distance map (in {dmap_col}).", loc='left')
    ax_dmap_splot.use_sticky_edges=False
    ax_dmap_splot.set_xticks([i for i in np.linspace(0, gridsize-1, gridsize)])
    ax_dmap_splot.set_yticks([i for i in np.linspace(0, gridsize-1, gridsize)])
    dmap_plot = ax_dmap_splot.pcolor(mg1, mg2, dmap, cmap=(cm.get_cmap(dmap_col, gridsize)), vmin=0, alpha=0.6)

    # winners scatterplot
    winplotstarttime = timer(); print("Drawing winner scatterplot over dmap now...")
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
    cbarlegendDMAPSPLOTstarttime = timer(); print("Drawing cbars & legend for the dmap-splot now...")
    axins_dmap = inset_axes(ax_dmap_splot, width='100%', height='100%',
                            loc='lower left', bbox_to_anchor=(0,-.05,.99,.015), 
                            bbox_transform=ax_dmap_splot.transAxes)
    cbar_dmap = fig.colorbar(dmap_plot, cax=axins_dmap, label='Distance from other nodes (0.0 indicates a complete similarity to neighboring node)', orientation='horizontal', pad=0.01)

    axins_splot = inset_axes(ax_dmap_splot, width='100%', height='100%', 
                             loc='lower left', bbox_to_anchor=(-.1, 0, .01, .99), 
                             bbox_transform=ax_dmap_splot.transAxes) # geometry & placement of cbar
    cbar_splot = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axins_splot, ticks=[i+.5 for i in range(len(model.years))], orientation='vertical', pad=0.5) 
    cbar_splot.ax.set_yticklabels(model.years); cbar_splot.ax.tick_params(size=3)

    ax_dmap_splot.legend(plots_for_legend, model.month_names, ncol=4, loc=9)

    fig.subplots_adjust(wspace=0.05,hspace=0.3)
    fig.savefig(f"{model.models_dir_path}/{model.RUN_time}_{utils.time_now()}-{model.month_names_joined}_prelim_SOMscplot_{gridsize}x{gridsize}",bbox_inches='tight', pad_inches=1)
    plt.close('all')
