import utils, prepare, validation, visualization, evaluation, run_test
from minisom import MiniSom
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from timeit import default_timer as timer
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from scipy import interp
from sklearn.metrics import roc_curve, auc, brier_score_loss
from sklearn.preprocessing import minmax_scale, RobustScaler
import numpy as np
import xarray as xr
import os, calendar, functools, logging

logger = logging.getLogger()
print = logger.info

class LocalModelParams:
    def __init__(self, model, xr_dataset):
        model.n_datapoints = xr_dataset.time.shape[0] # length of xr_dataset
        model.lat_size = xr_dataset.lat.shape[0]
        model.lon_size = xr_dataset.lon.shape[0]
        model.months = np.unique(xr_dataset['time.month'].values) # month numbers
        model.month_names = [calendar.month_name[m][:3] for m in np.unique(xr_dataset['time.month'])]
        model.month_names_joined = '_'.join(model.month_names).upper() # to print months properly
        model.years = np.unique(xr_dataset['time.year'].values) # unique years
        model.X, model.Y = xr_dataset.lon, xr_dataset.lat
    def __str__(self):
        return str(self.__dict__)  

class TopLevelModel:
    CHOSEN_VARS = ["relative_humidity", "u_component_of_wind", "v_component_of_wind"]
    min_maxes={}
    min_maxes['air_min'], min_maxes['air_max'] = 6, 30 # for colorbar uniformity
    min_maxes['rhum_min'], min_maxes['rhum_max'] = 0, 100
    rhum_pressure_levels = [850, 700]
    uwnd_vwnd_pressure_lvls = [925, 850, 700]
    unique_pressure_lvls = np.unique(rhum_pressure_levels+uwnd_vwnd_pressure_lvls)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    uniq_markers = ["+" , "^" , "3" ,
                    "v", "1", # 4, 5
                    "8", ">","x","P", # 6-9
                    "*","X", # 10, 11
                    "o"]

    def __init__(self, domain, period, hyperparameters, domain_limits):
        self.LAT_S, self.LAT_N, self.LON_W, self.LON_E = domain
        self.domain = domain
        d_str = '_'.join(str(i) for i in self.domain)
        self.dir_str = f'{d_str}'
        self.iterations, self.gridsize, self.training_mode, self.sigma, self.learning_rate, self.random_seed = hyperparameters
        self.dir_hp_str = f'{d_str}_{self.iterations}iter_{self.gridsize}x{self.gridsize}_{self.sigma}sig_{self.learning_rate}lr'
        self.period = period
        self.hyperparameters = hyperparameters
        self.RUN_datetime = utils.datetime_now()
        self.RUN_time = utils.time_now()
        self.domain_limits = domain_limits
        self.domain_limits_str = '_'.join(str(i) for i in domain_limits)
        # evaluation params
        self.ALPHAs = None # i.d. of PSI split dataset, distributed automatically

    def __str__(self):
        string =f'\n' \
                f'-----------------------------------------------------------------------------------------------------------------\n' \
                f'TopLevelModel with parameters D:{self.domain}, P:{self.period}, HP:{self.hyperparameters}\n' \
                f' === Run-time started on {self.RUN_datetime}'
        return string

    def detect_serialized_datasets(self):
        """
        Finding raw data pickles.
        If none found, proceed to creating pickles out of raw data.
        calls - 
        1. prepare.get_raw_input_data
        2. prepare.get_raw_target_data
        3. prepare.prepare_dataset
        """
        prepared_data_dir = str(utils.prepared_data_folder / self.dir_str / self.period)
        os.makedirs(prepared_data_dir, exist_ok=True)
        self.prepared_data_dir = prepared_data_dir
        print(f'Looking for pickles in {self.prepared_data_dir}')

        if len(utils.find('*serialized.pkl', self.prepared_data_dir)) == 2:
            print('This domain-period combination has been serialized before, loading objects...')
            for pkl in utils.find('*.pkl', self.prepared_data_dir):
                if "input_ds" in pkl: self.input_ds_serialized_path = pkl
                elif "rf_ds" in pkl: self.rf_ds_serialized_path = pkl
        else: 
            print('Proceeding to load & serialize raw data. ')
            self.raw_input_dir = prepare.get_raw_input_data(self)
            self.raw_rf_dir = prepare.get_raw_target_data(self)
            print(f'Raw input datasets taken from @: \n{self.raw_input_dir}')
            print(f'Raw rainfall datasets taken from @: \n{self.raw_rf_dir}')
            self.input_ds_serialized_path, self.rf_ds_serialized_path = prepare.prepare_dataset(self, self.prepared_data_dir)
        print(f'Serialized raw input datasets @: \n{self.input_ds_serialized_path}')
        print(f'Serialized raw RF datasets @: \n{self.rf_ds_serialized_path}')

    def detect_prepared_datasets(self):
        """
        Pre-processing, including time-slicing, removal of NAs, stacking & standardizing.
        calls - 
        1. prepare.preprocess_time_series
        2. prepare.flatten_and_standardize_dataset`
        """
        if utils.find('*target_ds_preprocessed.pkl', self.prepared_data_dir) and \
            utils.find('*rf_ds_preprocessed.pkl', self.prepared_data_dir) and \
            utils.find('*standardized_stacked_arr.pkl', self.prepared_data_dir):
            print('Pickles (preprocessed) found.')
            for pkl in utils.find('*preprocessed.pkl', self.prepared_data_dir):
                if "target_ds" in pkl: self.target_ds_preprocessed_path = pkl
                elif "rf_ds" in pkl: self.rf_ds_preprocessed_path = pkl
            
            LocalModelParams(self, utils.open_pickle(self.target_ds_preprocessed_path))

            for pkl in utils.find('*standardized_stacked_arr.pkl', self.prepared_data_dir):
                self.standardized_stacked_arr_path = pkl
        else:
            print('Pickles of pre-processed data incomplete. Proceeding to load & process raw dataset pickles.')
            self.target_ds_preprocessed_path, self.rf_ds_preprocessed_path = prepare.preprocess_time_series(self, self.prepared_data_dir, self.ALPHAs)

            LocalModelParams(self, utils.open_pickle(self.target_ds_preprocessed_path)) # generate new local model params

            self.standardized_stacked_arr_path = prepare.flatten_and_standardize_dataset(self, self.prepared_data_dir)
        print(f'--> Months for this dataset are: {self.month_names}')


    def train_SOM(self, alpha=None):
        d_hp_dir_path = str(utils.models_dir / self.dir_hp_str)
        self.d_hp_dir_path = d_hp_dir_path
        os.makedirs(d_hp_dir_path, exist_ok=True)
        if not utils.find(f'*extent_{self.dir_str}.png', self.d_hp_dir_path):
            visualization.get_domain_geometry(self, self.d_hp_dir_path)
            
        models_dir_path = str(utils.models_dir / self.dir_hp_str / self.period) + f'_{self.month_names_joined}'
        os.makedirs(models_dir_path, exist_ok=True)
        self.models_dir_path = models_dir_path
        # utils.update_cfgfile('Paths', 'models_dir_path', self.models_dir_path)

        if alpha:
            destination = self.alpha_model_dir
            arr_path = self.alpha_standardized_stacked_arr_path
            prefix = f'alpha_{alpha}_'
            prompt = f'< alpha-{alpha} >'
        else:
            destination = self.models_dir_path
            arr_path = self.standardized_stacked_arr_path
            prefix = ''
            prompt = ''

        print(f'Destination: "{destination}", arr_path: "{arr_path}", prefix: "{prefix}"')

        if utils.find(f'*{prefix}som_model.pkl', destination):
            print(f'{utils.time_now()} - SOM model trained before, skipping...')
            self.som_model_path = utils.find(f'*{prefix}som_model.pkl', destination)[0]
        else:
            print(f'{utils.time_now()} - {prompt} No SOM model trained for {self.domain}, {self.period}, for {self.hyperparameters}, doing so now...')

            standardized_stacked_arr = utils.open_pickle(arr_path)

            sominitstarttime = timer(); print(f'{utils.time_now()} - Initializing MiniSom... ')
            som = MiniSom(self.gridsize, self.gridsize, # square
                        standardized_stacked_arr.shape[1],
                        sigma=self.sigma, learning_rate=self.learning_rate,
                        neighborhood_function='gaussian', random_seed=self.random_seed)
            """
            Note: initializing PCA for weights is faster (~1/2 hour), but for serialized arrays > 300mb, 
            chances are this will kill the RAM and halt the entire process. 
            """
##            try:
##                som.pca_weights_init(standardized_stacked_arr)
##            except MemoryError as e:
##                print(f'Memory error has occured: \n{e}')
            print(f"Initialization took {utils.time_since(sominitstarttime)}.\n")

            trainingstarttime = timer(); print(f"{utils.time_now()} - Beginning training.")
            getattr(som, self.training_mode)(standardized_stacked_arr, self.iterations, verbose=True)
            q_error = np.round(som.quantization_error(standardized_stacked_arr), 2)
            print(f"Training complete. Q error is {q_error}, time taken for training is {utils.time_since(trainingstarttime)}s\n")

            if alpha: self.som_model_path = utils.to_pickle(f'{self.RUN_datetime}_{prefix}som_model', som, destination)
            else: self.som_model_path = utils.to_pickle(f'{self.RUN_datetime}_{prefix}som_model', som, destination)
    
    def detect_som_products(self, alpha=None):
        if alpha:
            destination = self.alpha_model_dir
            arr_path = self.alpha_standardized_stacked_arr_path
            prefix = f'alpha_{alpha}_'
            prompt = f'< alpha-{alpha} >'
        else:
            destination = self.models_dir_path
            arr_path = self.standardized_stacked_arr_path
            prefix = ''
            prompt = ''

        print(f'Destination: "{destination}", arr_path: "{arr_path}", prefix: "{prefix}", prompt:"{prompt}"')

        for phrase in ('winner_coordinates', 'dmap', 'ar', 'som_weights_to_nodes'):
            if utils.find(f'*{prefix}{phrase}.pkl', destination): 
                p = utils.find(f'*{prefix}{phrase}.pkl', destination)
                print(f'{utils.time_now()} - {prefix}{phrase} is found @: \n{p[0]}')
                exec(f'self.{phrase}_path = {p}[0]')
            else:
                print(f'{utils.time_now()} - {prompt} Some SOM products found missing in, generating all products now...')
                som = utils.open_pickle(self.som_model_path)
                standardized_stacked_arr = utils.open_pickle(arr_path)
                winner_coordinates = np.array([som.winner(x) for x in standardized_stacked_arr]) 
                dmap = som.distance_map()
                ar = som.activation_response(standardized_stacked_arr) 
                som_weights = som.get_weights() # weights for training via k-means
                som_weights_to_nodes = np.array(
                    [som_weights[c,r] for r in range(self.gridsize) for c in range(self.gridsize)]) #kmeans clustering
                
                self.winner_coordinates_path = utils.to_pickle(f'{prefix}winner_coordinates', winner_coordinates, destination)
                self.dmap_path = utils.to_pickle(f'{prefix}dmap', dmap, destination)
                self.ar_path = utils.to_pickle(f'{prefix}ar', ar, destination)
                self.som_weights_to_nodes_path = utils.to_pickle(f'{prefix}som_weights_to_nodes', som_weights_to_nodes, destination)

                break

        print('SOM products serialized.')

    def generate_k(self, alpha=None):
        """
        - detection of metrices to infer "k", i.e. optimal_k value
        - creation of metrices pickles 
        - creation of folders in models_dir to indicate potential k values/cluster combinations
        """
        metrics_dir = str(utils.metrics_dir / self.dir_hp_str / self.period) + f'_{self.month_names_joined}'
        os.makedirs(metrics_dir, exist_ok=True)
        self.metrics_dir_path = metrics_dir

        if alpha:
            self.alpha_metrics_dir_path = str(Path(self.tl_model.metrics_dir_path) / f'alpha_{alpha}')
            metric_destination = self.alpha_metrics_dir_path
            os.makedirs(metric_destination, exist_ok=True)
            model_destination = self.alpha_model_dir
            prefix = f'alpha_{alpha}_'
            prompt = f'< alpha-{alpha} >'
        else:
            metric_destination = self.metrics_dir_path
            model_destination = self.models_dir_path
            prefix = ''
            prompt = ''

        print(f'metric_destination: "{metric_destination}", model_destination: "{model_destination}", prefix: "{prefix}", prompt:"{prompt}"')
        
        for phrase in ('sil_peaks', 'ch_max', 'dbi_min', 'reasonable_sil', 'ch_dbi_tally', 'n_expected_clusters', 'dbs_err_dict'):
            if utils.find(f'*{prefix}{phrase}*.pkl', metric_destination): pass
            else:
                print(f'{utils.time_now()} - {prompt} Not all metrices have been found in {metric_destination}, generating them now...')
                # print all metrices if even 1 not found
                som_weights_to_nodes = utils.open_pickle(self.som_weights_to_nodes_path)

                ch_scores, dbi_scores = validation.print_elbow_CH_DBI_plot(self, som_weights_to_nodes, metric_destination)
                yellowbrick_expected_k = validation.print_yellowbrickkelbow(self, som_weights_to_nodes, metric_destination)
                silhouette_avgs, reasonable_silhoutte_scores_mt50 = validation.print_silhoutte_plots(self, som_weights_to_nodes, metric_destination)
                dbstop10 = validation.print_dbs_plots(self, som_weights_to_nodes, metric_destination)
                
                eps_ls, dbs_k_ls, dbs_noisepts_ls, dbs_labels = [], [], [], []
                for i in dbstop10:
                    eps_ls.append(i[0])
                    dbs_k_ls.append(i[1])
                    dbs_noisepts_ls.append(i[2])
                    dbs_labels.append(i[3])

                sil_peaks, ch_max, dbi_min, reasonable_sil, ch_dbi_tally, n_expected_clusters, dbs_err_dict = validation.get_cluster_determination_vars(
                    silhouette_avgs, ch_scores, dbi_scores, reasonable_silhoutte_scores_mt50, dbs_k_ls, dbs_noisepts_ls, yellowbrick_expected_k)

                for cluster_num in n_expected_clusters:
                    if alpha: save_dir = fr"{self.alpha_model_dir}/k-{cluster_num}"
                    else: save_dir = fr"{self.models_dir_path}/k-{cluster_num}"
                    
                    if cluster_num == ch_max: save_dir += '_CHhighestPeak'
                    if cluster_num == dbi_min: save_dir += '_lowestDBItrough'
                    if cluster_num in sil_peaks: save_dir += '_SilhouetteAVG-peak'
                    if cluster_num == reasonable_sil: save_dir += '_mostReasonable-basedon-Silhouetteplot'
                    if cluster_num in ch_dbi_tally: save_dir += '_CHpeak-and-DBItrough'
                    if cluster_num == yellowbrick_expected_k: save_dir += '_Yellowbrickexpected-K'
                    if cluster_num in dbs_err_dict: save_dir += f'_DBSCANclusterErrorValsExpected-{dbs_err_dict[cluster_num]}'

                    os.makedirs(save_dir, exist_ok=True)
                    print(f'save_dir: {save_dir}')

                self.ch_max_path = utils.to_pickle(f"{prefix}ch_max", ch_max, metric_destination)
                self.dbi_min_path = utils.to_pickle(f"{prefix}dbi_min", dbi_min, metric_destination)
                self.sil_peaks_path = utils.to_pickle(f"{prefix}sil_peaks", sil_peaks, metric_destination)
                self.reasonable_sil_path = utils.to_pickle(f"{prefix}reasonable_sil", reasonable_sil, metric_destination)
                self.ch_dbi_tally_path = utils.to_pickle(f"{prefix}ch_dbi_tally", ch_dbi_tally, metric_destination)
                self.yellowbrick_expected_k_path = utils.to_pickle(f"{prefix}yellowbrick_expected_k", yellowbrick_expected_k, metric_destination)
                self.dbs_err_dict_path = utils.to_pickle(f"{prefix}dbs_err_dict", dbs_err_dict, metric_destination)
                self.n_expected_clusters_path = utils.to_pickle(f"{prefix}n_expected_clusters", n_expected_clusters, metric_destination)

                break

        print(f'{utils.time_now()} - Internal validation of clusters has been run, please view metrices folder @:\n{metric_destination} to determine optimal cluster number.\n'\
            f'\nYou can view the separate folders constructed for each discovered cluster combination. See @: \n{model_destination}.')

    def get_k(self, minimum_confidence=3, cluster_search_minimum=8):
        """
        This function automates looking through the models folder & determining by the number of positions/"awards"
        whether the cluster number with the most number of "awards"
        has at least a "minimum confidence".

        Minimum_confidence is set at least 3, or at least 3 positions/awards for that cluster configuration to be given.
        
        - choosing the optimal "k" value
        """
        clusters_and_counts = [(int(str(i.stem).split('_')[0].split('-')[1]),
                                len(str(i.stem).split('_')[1:])) for i in Path(self.models_dir_path).glob('*') if i.stem[0] == 'k']
        best_cluster_num, highest_count = functools.reduce(lambda x,y: x if x[1] > y[1] else y, clusters_and_counts)
        if (highest_count >= minimum_confidence) and (best_cluster_num >= cluster_search_minimum): 
            self.optimal_k = best_cluster_num
            for i in Path(self.models_dir_path).glob('**/*'): 
                if i.match(f'*k-{best_cluster_num}_*'): 
                    self.cluster_dir = i
                    if not utils.find('*extent*.png', self.cluster_dir):
                        visualization.get_domain_geometry(self, self.cluster_dir)
                    print(f'"cluster_dir" initialized @:\n{i}')
                    return best_cluster_num
        else:
            self.optimal_k = None
            return False


    def train_kmeans(self, alpha=None):
        if alpha:
            optimal_k = self.tl_model.optimal_k
            print(f'>> self.alpha_model_dir: {self.alpha_model_dir}')
            print(f'>> optimal_k: {optimal_k}')
            found = [i for i in Path(self.alpha_model_dir).glob(f'k-{optimal_k}_*')]
            if found: 
                self.alpha_cluster_dir = found[0]
            else: self.alpha_cluster_dir = str(Path(self.alpha_model_dir) / f"k-{optimal_k}_NOT-singled-out-as-potential-cluster-for-this-split")
            os.makedirs(self.alpha_cluster_dir, exist_ok=True)
            print(f'>> self.alpha_cluster_dir: {self.alpha_cluster_dir}')
            destination = self.alpha_cluster_dir
            prefix = f'alpha_{alpha}_'
        else:
            optimal_k = self.optimal_k
            destination = self.cluster_dir
            prefix = ''

        print(f'optimal_k: "{optimal_k}", destination: "{destination}", prefix: "{prefix}"')
            
        for phrase in ('kmeans_model', 'labels_ar', 'labels_to_coords', 'label_markers', 'target_ds_withClusterLabels', 'dates_to_ClusterLabels', 'RFprec_to_ClusterLabels_dataset'):
            if utils.find(f'*{phrase}*.pkl', destination): 
                print(f'>>>>>>>>> "self.{phrase}_path" initialized.')
                exec(f'self.{phrase}_path = utils.find(f\'*{phrase}*.pkl\', r"{destination}")[0]')
            else:
                print(f'{utils.time_now()} - No KMeans model trained for {self.domain}, {self.period}, for {self.hyperparameters}, doing so now...')
                som_weights_to_nodes = utils.open_pickle(self.som_weights_to_nodes_path)
                samples, features = som_weights_to_nodes.shape
                km = KMeans(n_clusters=optimal_k).fit(som_weights_to_nodes)
                print(f"n{utils.time_now()} - K-means estimator fitted, sample size is {samples} and number of features is {features}.")

                self.kmeans_model_path = utils.to_pickle(f'{self.RUN_datetime}_{prefix}kmeans_model', km, destination)
                self.serialize_kmeans_products(km, alpha)
                break
            

    def serialize_kmeans_products(self, km, alpha):
        if alpha:
            arr_path = self.alpha_standardized_stacked_arr_path
            uniq_markers = self.tl_model.uniq_markers
            destination = self.alpha_cluster_dir
        else:
            arr_path = self.standardized_stacked_arr_path
            uniq_markers = self.uniq_markers
            destination = self.cluster_dir

        print(f'arr_path: "{arr_path}", uniq_markers: "{uniq_markers}", destination: "{destination}"')

        standardized_stacked_arr = utils.open_pickle(arr_path)
        target_ds = utils.open_pickle(self.target_ds_preprocessed_path)
        rf_ds_preprocessed = utils.open_pickle(self.rf_ds_preprocessed_path)

        labels_ar = km.labels_
        labels_to_coords = np.zeros([len(labels_ar), 2])
        for i, var in enumerate(labels_ar): labels_to_coords[i] = i % self.gridsize, i // self.gridsize

        try:
            label_markers = np.array([uniq_markers[var] for i, var in enumerate(labels_ar)])
        except IndexError: # more than 12 clusters
            label_markers = np.array([(uniq_markers*3)[var] for i, var in enumerate(labels_ar)])
            
        target_ds_withClusterLabels = target_ds.assign_coords(cluster=("time", km.predict(standardized_stacked_arr.astype(np.float))))
        dates_to_ClusterLabels = target_ds_withClusterLabels.cluster.reset_coords()
        RFprec_to_ClusterLabels_dataset = xr.merge([rf_ds_preprocessed, dates_to_ClusterLabels])

        self.labels_ar_path = utils.to_pickle(f'{self.RUN_datetime}_labels_ar', labels_ar, destination)
        self.labels_to_coords_path = utils.to_pickle(f'{self.RUN_datetime}_labels_to_coords', labels_to_coords, destination)
        self.label_markers_path = utils.to_pickle(f'{self.RUN_datetime}_label_markers', label_markers, destination)
        self.target_ds_withClusterLabels_path = utils.to_pickle(f'{self.RUN_datetime}_target_ds_withClusterLabels', target_ds_withClusterLabels, destination)
        self.dates_to_ClusterLabels_path = utils.to_pickle(f'{self.RUN_datetime}_dates_to_ClusterLabels', dates_to_ClusterLabels, destination)
        self.RFprec_to_ClusterLabels_dataset_path = utils.to_pickle(f'{self.RUN_datetime}_RFprec_to_ClusterLabels_dataset', RFprec_to_ClusterLabels_dataset, destination)
        

    def print_outputs(self, alpha=None):
        """
        if xx model output not found in self.cluster_dir with model.RUN_time at start of name
        """
        if alpha:
            cluster_dir = self.alpha_cluster_dir
            optimal_k = self.tl_model.optimal_k
            area = (self.tl_model.LON_E-self.tl_model.LON_W)*(self.tl_model.LAT_N-self.tl_model.LAT_S)
        else:
            cluster_dir = self.cluster_dir
            optimal_k = self.optimal_k
            area = (self.LON_E-self.LON_W)*(self.LAT_N-self.LAT_S)
            ind_cluster_plots_dir = str(Path(self.cluster_dir) / "indiv_cluster_plots")
            self.ind_cluster_plots_dir = ind_cluster_plots_dir
            os.makedirs(self.ind_cluster_plots_dir, exist_ok=True)
            print(f'self.ind_cluster_plots_dir is @:\n{self.ind_cluster_plots_dir}')

        print(f'cluster_dir: "{cluster_dir}", optimal_k: "{optimal_k}"')
        min_area = 2000
        if area > min_area: too_large = True
        else: too_large = False


        if not utils.find('*_prelim_SOMscplot_*.png', cluster_dir): visualization.print_som_scatterplot_with_dmap(self, cluster_dir)
        if not utils.find('*_kmeans*.png', cluster_dir): visualization.print_kmeans_scatterplot(self, cluster_dir, optimal_k)
        self.grid_width = visualization.grid_width(optimal_k)
        self.cbar_pos = np.max([(((clus)//self.grid_width)*self.grid_width)
                                for i,clus in enumerate(np.arange(optimal_k))
                                if i==(((clus)//self.grid_width)*self.grid_width)])
        if not utils.find('*_ARmonthfrac_*.png', cluster_dir): visualization.print_ar_plot(self, cluster_dir, optimal_k)
        if not utils.find('*_ARmonthfrac_granular_*.png', cluster_dir): visualization.print_ar_plot_granular(self, cluster_dir, optimal_k)
        if not utils.find('*_RFplot_mean_*.png', cluster_dir): visualization.print_rf_mean_plots(self, cluster_dir, optimal_k)
        if not utils.find('*_RFplot_max_v2*.png', cluster_dir): visualization.print_rf_max_plots(self, cluster_dir, optimal_k)
        if not utils.find('*_RFplot_rainday_gt1mm_v3_*.png', cluster_dir): visualization.print_rf_rainday_gt1mm_plots(self, cluster_dir, optimal_k)
        if not utils.find('*_RFplot_rainday_gt1mm_SGonly_v3_*.png', cluster_dir): visualization.print_rf_rainday_gt1mm_SGonly_plots(self, cluster_dir, optimal_k)
        if not utils.find('*_RFplot_heavyrainday_gt50mm_v2_*.png', cluster_dir): visualization.print_rf_heavyrf_gt50mm_plots(self, cluster_dir, optimal_k)
        if not utils.find('*_RFplot_90th_percentile_v2*.png', cluster_dir): visualization.print_rf_90th_percentile_plots(self, cluster_dir, optimal_k)
        if not utils.find('*_qp_v3*.png', cluster_dir): visualization.print_quiver_plots(self, cluster_dir, optimal_k)
        if not utils.find('*_qp_v5*.png', cluster_dir): visualization.print_quiver_plots(self, cluster_dir, optimal_k)
        if not utils.find('*_qp_v1_ANOM*.png', cluster_dir): visualization.print_quiver_ANOM_whole(self, cluster_dir, optimal_k)
        if not utils.find('*_qp_sgonly*.png', cluster_dir): visualization.print_quiver_plots_sgonly(self, cluster_dir, optimal_k)
        
        if not utils.find('*qp_baseline*.png', cluster_dir): visualization.print_quiver_baseline(self, cluster_dir, optimal_k)
        if not utils.find('*qp_baseline_Regionalonly*.png', cluster_dir): visualization.print_quiver_baseline_regional(self, cluster_dir, optimal_k)
        if not utils.find('*_rhum_v3-at*.png', cluster_dir): visualization.print_rhum_plots(self, cluster_dir, optimal_k)
        if not utils.find('*_rhum_sgonly-at*.png', cluster_dir): visualization.print_rhum_plots_sgonly(self, cluster_dir, optimal_k)
        if not utils.find('*_rhum_v5_ANOM-at*.png', cluster_dir): visualization.print_RHUM_ANOM_whole(self, cluster_dir, optimal_k)
        if not utils.find('*rhum_baseline*.png', cluster_dir): visualization.print_rhum_baseline(self, cluster_dir, optimal_k)

        if not utils.find('*RFplot_90th_percentile_baseline*.png', cluster_dir): visualization.print_RF_baselines(self, cluster_dir, optimal_k, 
        too_large)

        if not utils.find('*_qp_Regionalonly*.png', cluster_dir): visualization.print_quiver_Regionalonly(self, cluster_dir, optimal_k)
        if not utils.find('*_rhum_Regionalonly*.png', cluster_dir): visualization.print_RHUM_Regionalonly(self, cluster_dir, optimal_k)
        if not utils.find('*_rhum_Regionalonly_ANOM_v2*.png', cluster_dir): visualization.print_RHUM_ANOM_Regionalonly(self, cluster_dir, optimal_k)
        if not utils.find('*_qp_Regionalonly_ANOM*.png', cluster_dir): visualization.print_quiver_ANOM_Regionalonly(self, cluster_dir, optimal_k)
        if not utils.find('*_RFplot_90th_percentile_SGonly_ANOM_v1_*.png', cluster_dir): visualization.print_rf_90th_percentile_SGonly_ANOM_plots(self, cluster_dir, optimal_k)
        if not utils.find('*_RFplot_rainday_gt1mm_SGonly_ANOM_v3_*.png', cluster_dir): visualization.print_rf_rainday_gt1mm_SGonly_ANOM_plots(self, cluster_dir, optimal_k)
        if not utils.find('*_RFplot_heavy_gt50mm_SGonly_ANOM_v2_*.png', cluster_dir): visualization.print_rf_heavy_gt50mm_SGonly_ANOM_plots(self, cluster_dir, optimal_k)
        if not utils.find('*_RFplot_rainday_gt1mm_ANOM_v1_*.png', cluster_dir): visualization.print_rf_rainday_gt1mm_ANOM_plots(self, cluster_dir, optimal_k)

        if not utils.find('*_RFplot_heavy_gt50mm_ANOM_v2_*.png', cluster_dir): visualization.print_rf_heavy_gt50mm_ANOM_plots(self, cluster_dir, optimal_k)
        if not utils.find('*_RFplot_rainday_gt1mm_zscores_v2_*.png', cluster_dir): visualization.print_rf_gt1mm_zscore(self, cluster_dir, optimal_k, 
        too_large)
        if not utils.find('*_RFplot_heavy_gt50mm_zscores_v2_*.png', cluster_dir): visualization.print_rf_heavy_gt50mm_zscore(self, cluster_dir, optimal_k, 
        too_large)
        if not utils.find('*_RFplot_rainday_gt1mm_SGonly_zscores_v2_*.png', cluster_dir): visualization.print_rf_heavy_gt1mm_SGonly_zscore(self, cluster_dir, optimal_k, 
        too_large)
        if not utils.find('*_RFplot_heavy_gt50mm_SGonly_zscores_v3_*.png', cluster_dir): visualization.print_rf_heavy_gt50mm_SGonly_zscore(self, cluster_dir, optimal_k, 
        too_large)
        if not utils.find('*_RFplot_90th_percentile_ANOM_v1_*.png', cluster_dir): visualization.print_rf_90th_percentile_ANOM_plots(self, cluster_dir, optimal_k, 
        too_large)
        
        if not utils.find('*_RFplot_rainday_gt1mm_Regionalonly_ANOM_v2_*.png', cluster_dir): visualization.print_rf_gt1mm_ANOM_Regionalonly(self, cluster_dir, optimal_k)
        if not utils.find('*_RFplot_heavy_gt50mm_Regionalonly_ANOM_v2_*.png', cluster_dir): visualization.print_rf_gt50mm_ANOM_Regionalonly(self, cluster_dir, optimal_k)
        

        if not alpha:
            if not utils.find('*RFprec_to_ClusterLabels_dataset_vals_5xcoarsened_maxed.pkl', ind_cluster_plots_dir):
                print(f'5X coarsened RF full dataset not found. Generating now @:\n{ind_cluster_plots_dir}')
                self.full_rf_5Xcoarsened_vals_path = prepare.prepare_RF_vals_5x_coarsened(self, ind_cluster_plots_dir)
            else:
                self.full_rf_5Xcoarsened_vals_path = utils.find('*RFprec_to_ClusterLabels_dataset_vals_5xcoarsened_maxed.pkl', ind_cluster_plots_dir)[0]
            # for clus in range(optimal_k):
            #     """
            #     Individual test plots
            #     """
            #     if not utils.find(f"*RF_proportion_above_90thpercentile_cluster_{int(clus+1)}.png", ind_cluster_plots_dir): visualization.print_ind_clus_proportion_above_90thpercentile(self, ind_cluster_plots_dir, clus)
                # if not utils.find(f"*RF_proportion_under_10thpercentile_cluster_v2_{int(clus+1)}.png", ind_cluster_plots_dir): visualization.print_ind_clus_proportion_under_10thpercentile(self, ind_cluster_plots_dir, clus)
            #     if not utils.find(f"*RF_proportion_above_fullmodel_mean_cluster_{int(clus+1)}.png", ind_cluster_plots_dir): visualization.print_ind_clus_proportion_above_fullmodel_mean(self, ind_cluster_plots_dir, clus)
            #     if not utils.find(f"*RF_proportion_above_250mm_cluster_{int(clus+1)}.png", ind_cluster_plots_dir): visualization.print_ind_clus_proportion_above_250mm(self, ind_cluster_plots_dir, clus)
                
    
        print(
            f'Outputs are available @: \n{cluster_dir}\n\nThese are clustering results for domain {self.domain} ({self.period}), '\
            f'SOM-trained with hpparams {self.hyperparameters} \nand 2nd-level k-means clustered with k at {optimal_k}.')

    def assign_test_clusters_to_datasets(self):
        target_ds_preprocessed = utils.open_pickle(utils.find('*target_ds_preprocessed.pkl', self.test_prepared_data_dir)[0])
        rf_ds_preprocessed = utils.open_pickle(utils.find('*rf_ds_preprocessed.pkl', self.test_prepared_data_dir)[0])
        standardized_stacked_arr = utils.open_pickle(utils.find('*standardized_stacked_arr.pkl', self.test_prepared_data_dir)[0])
        
        self.n_datapoints = target_ds_preprocessed.time.shape[0] # length of xr_dataset
        self.lat_size = target_ds_preprocessed.lat.shape[0]
        self.lon_size = target_ds_preprocessed.lon.shape[0]
        self.months = np.unique(target_ds_preprocessed['time.month'].values) # month numbers
        self.month_names = [calendar.month_name[m][:3] for m in np.unique(target_ds_preprocessed['time.month'])]
        self.month_names_joined = '_'.join(self.month_names).upper() # to print months properly
        self.years = np.unique(target_ds_preprocessed['time.year'].values) # unique years
        self.X, self.Y = target_ds_preprocessed.lon, target_ds_preprocessed.lat

        km = utils.open_pickle(self.kmeans_model_path)
        predicted_clusters = km.predict(standardized_stacked_arr.astype(np.float))
        target_ds_withClusterLabels = target_ds_preprocessed.assign_coords(cluster=("time", predicted_clusters))
        dates_to_ClusterLabels = target_ds_withClusterLabels.cluster.reset_coords()
        RFprec_to_ClusterLabels_dataset = xr.merge([rf_ds_preprocessed, dates_to_ClusterLabels])
        utils.to_pickle('target_ds_withClusterLabels', target_ds_withClusterLabels, self.test_prepared_data_dir)
        utils.to_pickle('RFprec_to_ClusterLabels_dataset', RFprec_to_ClusterLabels_dataset, self.test_prepared_data_dir)

    def test_random_dates(self, dates_to_test, plots=5):
        test_dir = str(Path(__file__).resolve().parents[1] / 'test/2021_Jan_28_testing2020randomdates')
        self.test_RF_raw_data_dir = str(Path(__file__).resolve().parents[1] / "data/external/casestudytesting_29_Jan/raw/GPM_L3")
        self.test_indp_vars_raw_data_dir = str(Path(__file__).resolve().parents[1] / "data/external/casestudytesting_29_Jan/raw/downloadERA")
        self.test_prepared_data_dir = str(Path(__file__).resolve().parents[1] / f"data/external/casestudytesting_29_Jan/{self.period}_{self.dir_str}_prepared")
        os.makedirs(self.test_prepared_data_dir, exist_ok=True)

        number_of_test_plots_created = len(utils.find(f'*{self.period}_{self.dir_str}*_test_zscore_against_fullmodel*.png', test_dir))
        # number_of_test_plots_needed = date_to_test*plots
        if number_of_test_plots_created >= dates_to_test: 
            print(f"{number_of_test_plots_created} random dates have already been tested, please review at {test_dir}")
            return
        else:
            print(f"{number_of_test_plots_created} dates tested so far.")
            if not utils.find('*target_ds_preprocessed.pkl', self.test_prepared_data_dir) \
                or not utils.find('*rf_ds_preprocessed.pkl', self.test_prepared_data_dir) \
                    or not utils.find('*standardized_stacked_arr.pkl', self.test_prepared_data_dir): prepare.prep_for_testing_random_dates(self)

            if not utils.find('*target_ds_withClusterLabels.pkl', self.test_prepared_data_dir) \
                or not utils.find('*RFprec_to_ClusterLabels_dataset.pkl', self.test_prepared_data_dir): 
                self.assign_test_clusters_to_datasets()

            target_ds_withClusterLabels = utils.open_pickle(utils.find('*target_ds_withClusterLabels.pkl', self.test_prepared_data_dir)[0])
            if self.period == "NE_mon": target_ds_withClusterLabels = target_ds_withClusterLabels.sel(time=prepare.is_NE_mon(target_ds_withClusterLabels['time.month']))
            elif self.period == "SW_mon": target_ds_withClusterLabels = target_ds_withClusterLabels.sel(time=prepare.is_SW_mon(target_ds_withClusterLabels['time.month']))
            elif self.period == "inter_mon": target_ds_withClusterLabels = target_ds_withClusterLabels.sel(time=prepare.is_inter_mon(target_ds_withClusterLabels['time.month']))
            if target_ds_withClusterLabels.time.size == 0: 
                print(f'There are no dates available in your test dataset to use for this {self.period} monsoon period, please verify. Ending testing here.')
                return 

            random_sampled_dates = np.array(np.random.choice(target_ds_withClusterLabels.time.data, dates_to_test-number_of_test_plots_created, replace=False))
            random_sampled_dates.sort()
            print(random_sampled_dates)

            for i, sn in enumerate(range(number_of_test_plots_created+1, dates_to_test+1)):
            # for sn in range(number_of_test_plots_created+1, dates_to_test+1):
                print(f'Printing {sn} out of {dates_to_test} test plots now:')  
                # random_sampled_date = np.random.choice(target_ds_withClusterLabels.time.data, 1)
                random_sampled_date = [random_sampled_dates[i]]
                
                cluster = int(target_ds_withClusterLabels.sel(time=random_sampled_date).cluster.data)+1
                # run_test.print_test_date_abv_1mm_bool(self, test_dir, sn, random_sampled_date, cluster)
                run_test.print_test_date_abv_1mm_to500mm(self, test_dir, sn, random_sampled_date, cluster)   
                run_test.print_brier_gt1mm(self, test_dir, sn, random_sampled_date, cluster)   
                run_test.print_heavyrfforecastcomparison_gt50mm(self, test_dir, sn, random_sampled_date, cluster)   
                run_test.print_test_date_zscore_against_fullmodel(self, test_dir, sn, random_sampled_date, cluster)




class AlphaLevelModel(TopLevelModel):
    """
    PSI is number of years taken together to be co-dependent & will be regarded as ground-truth.
    Order of ground-truth group will be in ascending order from earliest year to latest, with overlaps according to PSI_overlap. 
    """
    def __init__(self, tl_model, domain, period, hyperparameters, domain_limits, PSI=3, PSI_overlap=0):
        super().__init__(domain, period, hyperparameters, domain_limits)
        self.alpha = 1
        self.PSI = PSI
        self.PSI_overlap = PSI_overlap
        self.nyears = len(tl_model.years)
        self.ALPHAs = self.nyears // self.PSI
        self.runoff_years = self.nyears % self.PSI
        self.tl_model = tl_model
        self.alpha_general_dir = Path(self.tl_model.cluster_dir) / f'alpha_general'
        os.makedirs(self.alpha_general_dir, exist_ok=True)

    def __str__(self):
        string =f'\n' \
                f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~@@~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
                f'=~ ===~ ==========~~~ ==================~~~~@@~~~~================== ~~~========== ~=== ~=\n' \
                f'AlphaLevelModel with ALPHAs:{self.ALPHAs}, PSI:{self.PSI}, PSI_overlap:{self.PSI_overlap}, ' \
                f'"alpha_general_dir" @:\n{self.alpha_general_dir}\n' \
                f' === Run-time started on {self.tl_model.RUN_datetime}'
        return string

        
    def check_evaluated(self, section=0):
        if section == 0: # i.e. whole model
            print('Checking if model has been evaluated before...')
            flag = [i.parent for i in Path(self.alpha_general_dir).glob(f'flag')]
            if flag:
                return flag[0]
            else: return False
        else: # specific alpha level has been provided (as "section")
            flags = [i.parent.stem for i in Path(self.alpha_general_dir).glob(f'*/flag') if 'alpha' in str(i.parent.stem)]
            if flags:
                for i in flags:
                    if int(i.split('_')[1]) == section: 
                        self.alpha +=1
                        return True
            self.alpha_prepared_dir = str(Path(self.tl_model.prepared_data_dir) / f'alpha_{section}')
            print(f'No flag found for alpha ({section}), proceeding with evaluation procedure.\n\n\n**[ALPHA-{section}]**')
            return False

    def prepare_nfold_datasets(self): # i.e. split into different train/ground-truth(test) dataset
        """
        Take previously pre-processed datasets & split into
        input: x_train, x_test,
        target/RF: y_train, y_test
        """
        for alpha in range(1, self.ALPHAs+1):
            if alpha != self.ALPHAs:
                gt_years = np.array2string(self.tl_model.years[(alpha-1)*self.PSI : alpha*self.PSI], separator='-')
            else:
                gt_years = np.array2string(self.tl_model.years[(alpha-1)*self.PSI : alpha*self.PSI+self.runoff_years], separator='-')
            new_cluster_dir = str(Path(self.tl_model.cluster_dir) / f'alpha_{alpha}_GT-{gt_years}')
            os.makedirs(new_cluster_dir, exist_ok=True)

            new_prepared_data_dir = str(Path(self.tl_model.prepared_data_dir) / f'alpha_{alpha}')
            os.makedirs(new_prepared_data_dir, exist_ok=True)
            
            if utils.find(f'*alpha_{alpha}_preprocessed.pkl', new_prepared_data_dir) and utils.find(f'*alpha_{alpha}_standardized_stacked_arr.pkl', new_prepared_data_dir):
                pass
            else:
                if not utils.find(f'*target*alpha_{alpha}_preprocessed.pkl', new_prepared_data_dir):
                    print(f"=> No input datasets pre-processed for alpha of {alpha}")
                    prepare.cut_target_dataset(self, alpha, new_prepared_data_dir)

                if not utils.find(f'*rf*alpha_{alpha}_preprocessed.pkl', new_prepared_data_dir):
                    print(f"=> No rainfall datasets pre-processed for alpha of {alpha}")
                    prepare.cut_rf_dataset(self, alpha, new_prepared_data_dir)
                
                print(f'Preprocessed pickles for alpha split {alpha} can be found @:\n{new_prepared_data_dir}')           

    def prepare_alphafold_dataset(self, alpha):
        print(f'Preparing dataset for alpha-{alpha}')
        if alpha != self.ALPHAs:
            self.gt_years = np.array2string(self.tl_model.years[(alpha-1)*self.PSI : alpha*self.PSI], separator='-')
        else:
            self.gt_years = np.array2string(self.tl_model.years[(alpha-1)*self.PSI : alpha*self.PSI+self.runoff_years], separator='-')

        self.alpha_prepared_dir = str(Path(self.tl_model.prepared_data_dir) / f'alpha_{alpha}')
        self.alpha_model_dir = str(Path(self.tl_model.cluster_dir) / f'alpha_{alpha}_GT-{self.gt_years}')
        
        for pkl in utils.find(f'*alpha_{alpha}_preprocessed.pkl', self.alpha_prepared_dir):
            if "target_ds_train" in pkl: self.target_ds_preprocessed_path = pkl
            elif "rf_ds_train" in pkl: self.rf_ds_preprocessed_path = pkl
            elif "target_ds_test" in pkl: self.x_test_path = pkl
            elif "rf_ds_test" in pkl: self.y_test_path = pkl
        
        LocalModelParams(self, utils.open_pickle(self.target_ds_preprocessed_path))

        if utils.find('*standardized_stacked_arr.pkl', self.alpha_prepared_dir):
            self.alpha_standardized_stacked_arr_path = utils.find(f'*standardized_stacked_arr.pkl', self.alpha_prepared_dir)[0]
        else:
            self.alpha_standardized_stacked_arr_path = prepare.flatten_and_standardize_dataset(self, self.alpha_prepared_dir)
        print(f'--> Months for this dataset are: {self.month_names}')

        print(
            f'paths created @ prepare_alphafold_dataset():\nself.alpha_prepared_dir: "{self.alpha_prepared_dir}", \nself.alpha_model_dir: "{self.alpha_model_dir}"'
            f'\nself.target_ds_preprocessed_path: "{self.target_ds_preprocessed_path}", \nself.rf_ds_preprocessed_path: "{self.rf_ds_preprocessed_path}"' \
            f'\nself.rf_ds_preprocessed_path: "{self.rf_ds_preprocessed_path}", \nself.x_test_path: "{self.x_test_path}", \nself.y_test_path: "{self.y_test_path}"' \
            f'\nself.alpha_standardized_stacked_arr_path: "{self.alpha_standardized_stacked_arr_path}", \nself.gt_years: {self.gt_years}' \
            )

    def evaluation_procedure(self, alpha):
        """
        AUC, Brier, probabilities, etc.
        """
        self.alpha_cluster_scoring_dir = str(Path(self.alpha_cluster_dir) / f'Evaluation_scoring' / f'Rain_days_1mm_and_above')
        os.makedirs(self.alpha_cluster_scoring_dir, exist_ok=True)

        print(f'<alpha-{alpha}> self.alpha_cluster_scoring_dir @: \n{self.alpha_cluster_scoring_dir}')
        if utils.find(f'*Mean_brier_score*in_alpha_{alpha}*.png', self.alpha_cluster_scoring_dir) and \
            utils.find('*Brier_scores_for_cluster_predictions*alpha-{alpha}*.png', self.alpha_cluster_scoring_dir) and \
            len(utils.find(f'*clus_gt_*', self.alpha_cluster_dir)) == self.tl_model.optimal_k and \
            len(utils.find(f'*clus_pred_*', self.alpha_cluster_dir)) == self.tl_model.optimal_k : 
            pass
        else:
            print(f'Acquiring mean brier scores for each cluster in alpha-{alpha}!')
            evaluation.mean_brier_individual_alpha(self, alpha)


        if len(utils.find(f'*ROCs_for_alpha_{alpha}*.png', self.alpha_cluster_scoring_dir)) == (self.tl_model.optimal_k + 1): 
            pass
        else:
            print('Plotting ROC curves for individual alpha-{alpha} now!')
            evaluation.ROC_AUC_individual_alpha(self, alpha)



        if utils.find(f'*Gridded_brier_individual_alpha_{alpha}_v2*.png', self.alpha_cluster_scoring_dir): 
            pass
        else:
            print(f'{utils.time_now()} - Plotting gridded brier scores for individual alpha-{alpha}...')
            evaluation.gridded_brier_individual_alpha(self, alpha)

        # print(f'DEBUGGING: {utils.time_now()} - Plotting gridded brier scores for individual alpha-{alpha}...')
        # evaluation.gridded_brier_individual_alpha(self, alpha)



        
        if utils.find(f'*Gridded_AUC_individual_alpha_{alpha}_v2*.png', self.alpha_cluster_scoring_dir) and \
            utils.find(f'*alpha_{alpha}_aucs.pkl', self.alpha_general_dir): 
            pass
        else:
            print(f'{utils.time_now()} - Plotting gridded AUC for individual alpha-{alpha}...')
            evaluation.gridded_AUC_individual_alpha(self, alpha)

        # print(f'DEBUGGING: {utils.time_now()} - Plotting gridded AUC for individual alpha-{alpha}...')
        # evaluation.gridded_AUC_individual_alpha(self, alpha)



        print(f'Evaluation completed for raindays (1mm & above) predictions for alpha-{alpha}.')

    def compile_scores(self):
        if utils.find(f'*Brier_scores_weighted-avg_boxplot_with_bootstrapped_whole-model_mean*.png', self.alpha_general_dir): pass
        else:
            print('Plotting mean brier scores across all alphas...')
            evaluation.mean_brier_scores_all_alphas(self)

        
        if utils.find(f'*ROC_whole-model-mean_all-alphas_micro-avged*.png', self.alpha_general_dir): pass
        else:
            print('Plotting ROC across all alphas...')
            evaluation.ROC_all_alphas(self)

        
        if utils.find(f'*gridded_AUC_whole-model_v2*.png', self.alpha_general_dir): pass
        else:
            print('Plotting gridded AUC across all alphas...')
            evaluation.gridded_AUC_all_alphas(self)

                
        if utils.find(f'*gridded_brier_whole-model_v2*.png', self.alpha_general_dir): pass
        else:
            print('Plotting gridded brier scores across all alphas...')
            evaluation.gridded_brier_all_alphas(self)

        # print(f'DEBUGGING: {utils.time_now()} - Plotting gridded AUC across all alphas...')
        # evaluation.gridded_AUC_all_alphas(self)

        # print(f'DEBUGGING: {utils.time_now()} - Plotting gridded brier scores across all alphas...')
        # evaluation.gridded_brier_all_alphas(self)
        
        with open(f'{self.alpha_general_dir}/flag', 'w+') as flag: pass # write flag to signal evaluation completed



