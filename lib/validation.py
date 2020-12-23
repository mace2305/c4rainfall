"""
- validation metrices (elbow/CH/DBI, sil plots, DBSCAN, yellow-brick)
- load from utils fx to search for working directory of full-model
- saving .csv for validation scores to compile for "k"
- function to derive either 'suggested k' or 'No "k" detected'
"""

import utils
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_samples, silhouette_score
from yellowbrick.cluster import KElbowVisualizer
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import collections, time, logging

logger = logging.getLogger()
print = logger.info

def print_elbow_CH_DBI_plot(model, som_weights_to_nodes, destination, up_to=26):
    ## Distortion "elbow method", CH-score & DBI indices plotted.
    distortions = [] 
    calinski_harabasz_scores = []
    dbi_scores = []

    for i in range(2, up_to):
        km = KMeans(n_clusters=i).fit(som_weights_to_nodes)
        distortions.append(km.inertia_)
        cluster_labels = km.fit_predict(som_weights_to_nodes)
        calinski_harabasz_scores.append(calinski_harabasz_score(som_weights_to_nodes, cluster_labels))
        dbi_scores.append(davies_bouldin_score(som_weights_to_nodes, cluster_labels))

    fig = plt.figure(figsize=(20,20))
    ax_iner = fig.add_subplot(311)
    ax_iner.plot(np.arange(2, up_to), distortions, 'bx-')
    plt.xlabel("Clusters")
    plt.ylabel('Distortion') 
    plt.title('The Elbow Method') 
    ax_iner.set_xticks(np.arange(2, up_to))

    ax_iner = fig.add_subplot(312)
    ax_iner.plot(np.arange(2, up_to), calinski_harabasz_scores, 'bx-')
    plt.xlabel("Clusters")
    plt.ylabel('Calinski-Harabasz Index')
    plt.title('The "higher" the index, the better defined the clusters.') 
    ax_iner.set_xticks(np.arange(2, up_to))

    ax_iner = fig.add_subplot(313)
    ax_iner.plot(np.arange(2, up_to), dbi_scores, 'bx-')
    plt.xlabel("Clusters")
    plt.ylabel('Davies-Bouldin score')
    plt.title('The closer the score the 0, the least "similarity" between-clusters.') 
    ax_iner.set_xticks(np.arange(2, up_to))

    fig.subplots_adjust(wspace=0.05,hspace=0.3)
    fig.savefig(f"{destination}/{model.RUN_time}_{utils.time_now()}-{model.month_names_joined}_Distortionelbow-CHscore-DBIscore",bbox_inches='tight', pad_inches=1)
    plt.close('all')
    return calinski_harabasz_scores, dbi_scores

def print_yellowbrickkelbow(model, som_weights_to_nodes, destination, up_to=26):
    kelbow_visualizer = KElbowVisualizer(KMeans(), k=(2,up_to))
    kelbow_visualizer.fit(som_weights_to_nodes)
    kelbow_visualizer.show(f'{destination}/{model.RUN_time}_{utils.time_now()}-{model.month_names_joined}_yellowbrickelbowfor-k2.png')
    if kelbow_visualizer.elbow_value_ != None:
        return kelbow_visualizer.elbow_value_

def print_silhoutte_plots(model, som_weights_to_nodes, destination, up_to=26):
    range_n_clusters = np.arange(2,up_to)

    ## Silhoutte plots to determine manually optimal cluster num
    print("Drawing silhoutte plots now...")
    fig = plt.figure(constrained_layout=True);
    gs = fig.add_gridspec(len(range_n_clusters)+2, 2)
    fig.set_size_inches(11, 6*len(range_n_clusters))    

    silhouette_avgs, raws, negatives, valid_perc_ls, reasonable_percs, stable_percs = [],[],[],[],[],[]
    total_dpts = len(som_weights_to_nodes)

    for i, n_clusters in enumerate(range_n_clusters):
        ax1 = fig.add_subplot(gs[2*(i+1)])
        ax1.set_title(f'i={i}')
        ax2 = fig.add_subplot(gs[2*(i+1)+1])
        ax2.set_title(f'i={i}')

        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, total_dpts + (n_clusters + 1) * 10])

        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(som_weights_to_nodes)
        silhouette_avg = silhouette_score(som_weights_to_nodes, cluster_labels)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avgs.append(silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(som_weights_to_nodes, cluster_labels)

        y_lower = 10
        raw_scores, neg_score, lt_25, mt_50, mt_70 = [], 0, 0, 0, 0
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
            raw_scores.append(ith_cluster_silhouette_values)
            negvals = sum(ith_cluster_silhouette_values<0)
            n_lt25 = sum(ith_cluster_silhouette_values<.25)-negvals
            n_mt50 = sum(ith_cluster_silhouette_values>.5)
            n_mt70 = sum(ith_cluster_silhouette_values>.7)
            neg_score += negvals
            lt_25 += n_lt25
            mt_50 += n_mt50
            mt_70 += n_mt70
        raws.append(raw_scores)
        negatives.append(neg_score)
        valid = total_dpts-lt_25-neg_score
        valid_perc = np.round(valid/total_dpts, 2); valid_perc_ls.append(valid_perc)
        reasonable = np.round(mt_50/total_dpts, 2); reasonable_percs.append(reasonable)
        stable = np.round(mt_70/total_dpts, 2); stable_percs.append(stable)
        

        # Sewell, Grandville, and P. J. Rousseau. "Finding groups in data: An introduction to cluster analysis." (1990).
        ax1.set_title(f"Clusters: {n_clusters}, probable error values: {neg_score}, "
                      f"\nWith {lt_25} datapts found to score <0.25, {mt_50} scoring >= 0.5, & {mt_70} over 0.7"
                      f"\nValid datapts: {valid}/{total_dpts} ({valid_perc}%). Sewell, Grandville, Rousseau (1990)") 
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(som_weights_to_nodes[:, 0], som_weights_to_nodes[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("Visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")


    ax_silavgs = fig.add_subplot(gs[0:1,:])
    ax_silavgs.set_title(f'i={i}')

    p, = ax_silavgs.plot(range(len(range_n_clusters)), silhouette_avgs, 'kx-', label='Average silhouette coefficient.')
    plt.xlabel("Clusters")
    plt.ylabel('Silhoutte coefficient')
    plt.title('Closer to 1: highly dense clustering, closer to -1 is incorrect clustering. Coefficients close to 0 indicate overlapping clusters.') 
    plt.xticks(np.arange(up_to), np.arange(2, up_to))
    
    ax2 = ax_silavgs.twinx()
    p1, = ax2.plot(valid_perc_ls, '--m', label='"Valid" membership (>0.25 silhouette coefficient)')
    ax2.grid(False)
    plt.ylim(0,1)
    plt.ylabel('Fraction of datapts (x100%)')
    
    p2, = ax2.plot(reasonable_percs, '--b', label='Reasonable cluster structure (>0.50)')
    ax2.grid(False)
    plt.ylim(0,1)
    
    p3, = ax2.plot(stable_percs, '--g', label='Stable cluster structure (>0.70)')
    ax2.grid(False)
    plt.ylim(0,1)
    
    lines = (p,p1,p2,p3)
    ax_silavgs.legend(lines, [l.get_label() for l in lines], loc=0)

    fig.suptitle(("Silhouette analysis for KMeans clustering on sample data."),
                 fontsize=14, fontweight='bold')
    
    fig.savefig(f"{destination}/{model.RUN_time}_{utils.time_now()}-{model.month_names_joined}_Silhoutte-plots-2-to-{up_to-1}-clusters",bbox_inches='tight', pad_inches=1)
    plt.close('all')
    return silhouette_avgs, reasonable_percs


def print_dbs_plots(model, som_weights_to_nodes, destination, up_to=10):
    ## DBSCAN: clustering for not necessarily convex clusters
    cluster_queue = collections.deque([0,0,0], up_to)

    eps = 0.5
    eps_ls = []; n_clusters_ls = []; n_noise_ls=  []; labels_ls = []
    ascending = False

    while True:
        db= DBSCAN(eps=eps, min_samples=10).fit(som_weights_to_nodes)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        labels_ls.append(labels)

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        eps_ls.append(np.round(eps, 2)); n_clusters_ls.append(n_clusters_); n_noise_ls.append(n_noise_)
        eps += 0.2

        cluster_queue.append(n_clusters_)
        if n_clusters_ > 0: ascending = True
        if (len(set(cluster_queue))==1) and ascending: break
    
    
    a = sorted([comb for comb in zip(eps_ls, n_clusters_ls, n_noise_ls, labels_ls)], key=lambda x: x[1], reverse=True)
    
    bigfig = plt.figure(constrained_layout=True, figsize=(5.5,4*up_to))
    gs = bigfig.add_gridspec(up_to,1)

    for i, tup in enumerate(a[:up_to]):
        ax = bigfig.add_subplot(gs[i])

        eps = tup[0]
        n_clusters_ = tup[1]
        n_noise_ = tup[2]
        labels = tup[3]

        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = som_weights_to_nodes[class_member_mask & core_samples_mask]
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=7)

            xy = som_weights_to_nodes[class_member_mask & ~core_samples_mask]
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=2)

        plt.title(f'Estimated number of clusters: {n_clusters_} @ {eps} "eps"\n'
                  f'Estimated number of noise points: {n_noise_} (in black)'
                 )

    bigfig.suptitle(f'Max clusters: {a[0][1]}, distribution of top-10 cluster numbers via DBSCAN, descending.', fontsize=20)
    
    bigfig.savefig(f"{destination}/{model.RUN_time}_{utils.time_now()}-{model.month_names_joined}_DBSCAN-varied-by-eps",bbox_inches='tight', pad_inches=1)
    plt.close('all')
    return a[:up_to]

def get_ch_dbi_tally_dictionary(ch_scores, dbi_scores):
    prev_ch, prev_dbi = 0, 0
    dbi_ch_tallies = {}
    for i, ch_dydx in enumerate(np.diff(ch_scores)):
        dbi_dydx = np.diff(dbi_scores)[i]
#         print(f'{i+2} - ch_dydx:{ch_dydx}, prev_ch: {prev_ch}, dbi_dydx:{dbi_dydx}, prev_dbi: {prev_dbi}')
        if i != 0:
            if (ch_dydx<0) & (prev_ch>0) & (dbi_dydx>0) & (prev_dbi<0): dbi_ch_tallies[i+2] = 1 # +2 since cluster starts at 2, & cluster 2 derivative not taken
            else: dbi_ch_tallies[i+2] = 0
        prev_ch = ch_dydx; prev_dbi = dbi_dydx

def get_silhouette_peaks(silhouette_avgs):
    prev_sil = 0
    sil_peaks = {}
    for i, sil in enumerate(np.diff(silhouette_avgs)):
        if i != 0:
            if (sil<0) & (prev_sil>0): sil_peaks[i+2] = 1 # +2 since cluster starts at 2, & cluster 2 derivative not taken
            else: sil_peaks[i+2] = 0
        prev_sil = sil
    return sil_peaks


def get_cluster_determination_vars(silhouette_avgs, ch_scores, dbi_scores, reasonable_silhoutte_scores_mt50, 
                                   dbs_k_ls, dbs_noisepts_ls, yellowbrick_expected_k):
    # derive variables from each cluster determination method
    # loop through said n_clusters, with a 
    # new dir created, with dir_name indicating the following:

    """
    1. expected clusters
    2. CH score peak or not + *tally with (3) i.e. CH peak & DBI trough
    3. DBI score trough or not + *tally with (2) i.e. CH peak & DBI trough
    4. silhoeutte (a) score (b) peak or not
    5. DBSCAN top-10 cluster numbers or not + @ what eps
    6. yellowbrick elbow or not

    order for naming (logic): 
    a. do (6) yellowbricks's expected cluster
    b. do (4) silhouette avgs peaks
    c. do highest peak for CH (2)
    d. do lowest DBI (3)
    e. do highest "reasonable" (4) silhouette valid datapoints
    f. indicate if expected clusters of (2) and (3) if tally i.e. (2)'s peaks, (3)'s troughs!

    variables KIV: 
    ch_scores = 24 scores (2 clusters to 25 clusters) 
    dbi_scores = 24 scores (2 clusters to 25 clusters) 
    yellowbrick_expected_k = 1 value
    silhouette_avgs = 24 scores (2 clusters to 25 clusters) 
    raw_silhoutte_scores = list of arrays for each silhouette plot
    eps_ls = top-10 eps, descending from top-1
    dbs_k_ls = cluster numbers, descending
    dbs_noisepts_ls = number noise per cluster
    dbs_labels = labels for classes to each input, will not be used
    """
    sil_peaks = [i for i,v in get_silhouette_peaks(silhouette_avgs).items() if v == 1]
    ch_max = np.argmax(ch_scores)+2
    dbi_min = np.argmin(dbi_scores)+2
    reasonable_sil = np.argmax(reasonable_silhoutte_scores_mt50[6:])+8 # "[6:]..+8" as starting from 8th cluster and beyond
    try:
        ch_dbi_tally = [i for i,v in get_ch_dbi_tally_dictionary(ch_scores, dbi_scores).items() if v == 1]
    except:
        ch_dbi_tally = []
    dbs_err_dict = {}
    for i,k in enumerate(dbs_k_ls):
        if k not in dbs_err_dict: dbs_err_dict[k] = dbs_noisepts_ls[i]
        elif dbs_err_dict[k] > dbs_noisepts_ls[i]: dbs_err_dict[k] = dbs_noisepts_ls[i]
    n_expected_clusters = np.unique([i for i in (yellowbrick_expected_k,*sil_peaks,ch_max,dbi_min,reasonable_sil)])
    return sil_peaks, ch_max, dbi_min, reasonable_sil, ch_dbi_tally, n_expected_clusters, dbs_err_dict