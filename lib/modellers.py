"""
- SOM training
- extraction of SOM products
- kmeans clustering
- loading from utils where to save models
- saving of SOM & kmeans model 
- storing of metadata in Model Classes


## Classes for model tracking
# Note: storing state is very expensive, this is for creating methods to serialize objects,
# and loading objects where necessary. 

- (1) full model, overview: RUN/full model i.d.,
metadata, arguments/parameters (d,p,hp), "k"-generated/validated, 
full model ran (date), visualized (bool), evaluated (bool), 
PSI/year dependency, _years split, _ALPHA_completed, _ALPHA_remaining, ALPHA/split i.d.,
scores for all splits,

* CHOSEN_VARS, min_maxes, unique_pressure_lvls, lon_formatter, lat_formatter, uniq_markers
Y (from config file)* raw input data directory, raw RF data dir
* saved datasets dir
**ds_len/"n_datapoints", latsize/lonsize, month, month_names...

= method to surface which years split
= method for which split done (_ALPHA_completed)
= method to display average scoring for all splits (up to PSI, i.e. for this entire model RUN)


- (2) ALPHA-split model(RUN, ALPHA) for evaluation: 
DELTA/bootstrap sample no., _BETA/bootstrap sample i.d., score_total(_BETA, DELTA), 
X_ALPHA_train, X_ALPHA_test,
Y_ALPHA_train, Y_ALPHA_test, 
avg RF CI for each cluster based off bootstrap evaluations - for this split,
X_ALPHA_train_predicted_clusters, X_ALPHA_train_predicted_clusters dates,
Y_ALPHA_pred/Y_ALPHA_train predicted rainfall, Y_ALPHA_GT/Y_ALPHA_test "ground truth" rainfall, 
score_fit(Y_ALPHA_pred, Y_ALPHA_GT)/scores for clusters for DELTA samples - i.e. for split ALPHA,
= method to display average scoring for all bootstrap samples (up to DELTA, i.e. for this split ALPHA)
= method to change (**ds_len/"n_datapoints", latsize/lonsize, month, month_names...) accordingly


-- (3) BETA/bootstrap sample's model(X_ALPHA_train, _BETA), for low-level evaluation:  XXX not needed already
_i/predicting for which cluster,
_Y_pred_i/predicted RF for cluster i (of k) in BETA re-sample, Y_pred_j/sum of all _Y_pred_i's for cluster_i, 
BETA_Y_pred/sum of all predicted RF for BETA sample, X_BETA_train, 
X_BETA_test/validation set for all samples not picked up in bootstrap resampling, 
Y_BETA_test_i/observed RF for cluster i of validation set, Y_BETA_test,
*score_fit(Y_pred_j, Y_BETA_test) == {might be score_fit(_Y_pred_i, Y_BETA_test_i)}/score for unseen dataset for BETA,
= return predicted RF for all clusters, up to k
= method to change (**ds_len/"n_datapoints", latsize/lonsize, month, month_names...) accordingly


"""
import utils, prepare, validation, visualization
from minisom import MiniSom
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from timeit import default_timer as timer
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
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
        self.iterations, self.gridsize, self.training_mode, self.sigma, self.learning_rate, self.random_seed = hyperparameters
        d_str = '_'.join(str(i) for i in self.domain)
        self.dir_hp_str = f'{d_str}_{self.iterations}iter_{self.gridsize}x{self.gridsize}_{self.sigma}sig_{self.learning_rate}lr'
        self.dir_str = f'{d_str}'
        self.period = period
        self.hyperparameters = hyperparameters
        self.RUN_datetime = utils.datetime_now()
        self.RUN_time = utils.time_now()
        self._full_train = False
        self.visualized = False
        self.domain_limits = domain_limits
        self.domain_limits_str = '_'.join(str(i) for i in domain_limits)
        # evaluation params
        self.ALPHAs = None # i.d. of PSI split dataset, distributed automatically

        print(self)

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
        if utils.find('*preprocessed.pkl', self.prepared_data_dir) and utils.find('*standardized_stacked_arr.pkl', self.prepared_data_dir):
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
        print(f'--> Months for this dataset are: {self.month_names}\n')


    def train_SOM(self):
        models_dir_path = str(utils.models_dir / self.dir_hp_str / self.period) + f'_{self.month_names_joined}'
        os.makedirs(models_dir_path, exist_ok=True)
        self.models_dir_path = models_dir_path
        # utils.update_cfgfile('Paths', 'models_dir_path', self.models_dir_path)

        if utils.find('*som_model.pkl', self.models_dir_path):
            print(f'{utils.time_now()} - SOM model trained before, skipping...')
            self.som_model_path = utils.find('*som_model.pkl', self.models_dir_path)[0]
        else:
            print(f'{utils.time_now()} - No SOM model trained for {self.domain}, {self.period}, for {self.hyperparameters}, doing so now...')

            standardized_stacked_arr = utils.open_pickle(self.standardized_stacked_arr_path)

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

            self.som_model_path = utils.to_pickle(f'{self.RUN_datetime}_som_model', som, self.models_dir_path)
    
    def detect_som_products(self):        
        for phrase in ('winner_coordinates', 'dmap', 'ar', 'som_weights_to_nodes'):
            if utils.find(f'*{phrase}.pkl', self.models_dir_path): 
                p = utils.find(f'*{phrase}.pkl', self.models_dir_path)
                print(f'{utils.time_now()} - {phrase} is found @: \n{p[0]}')
                exec(f'self.{phrase}_path = {p}[0]')
            else:
                print(f'{utils.time_now()} - Some SOM products found missing in, generating all products now...')
                som = utils.open_pickle(self.som_model_path)
                standardized_stacked_arr = utils.open_pickle(self.standardized_stacked_arr_path)
                winner_coordinates = np.array([som.winner(x) for x in standardized_stacked_arr]) 
                dmap = som.distance_map()
                ar = som.activation_response(standardized_stacked_arr) 
                som_weights = som.get_weights() # weights for training via k-means
                som_weights_to_nodes = np.array(
                    [som_weights[c,r] for r in range(self.gridsize) for c in range(self.gridsize)]) #kmeans clustering
                
                self.winner_coordinates_path = utils.to_pickle('winner_coordinates', winner_coordinates, self.models_dir_path)
                self.dmap_path = utils.to_pickle('dmap', dmap, self.models_dir_path)
                self.ar_path = utils.to_pickle('ar', ar, self.models_dir_path)
                self.som_weights_to_nodes_path = utils.to_pickle('som_weights_to_nodes', som_weights_to_nodes, self.models_dir_path)

                break

        print('SOM products serialized.')

    def generate_k(self):
        """
        - detection of metrices to infer "k", i.e. optimal_k value
        - creation of metrices pickles 
        - creation of folders in models_dir to indicate potential k values/cluster combinations
        """
        metrics_dir = str(utils.metrics_dir / self.dir_hp_str / self.period) + f'_{self.month_names_joined}'
        os.makedirs(metrics_dir, exist_ok=True)
        self.metrics_dir_path = metrics_dir
        # utils.update_cfgfile('Paths', 'metrics_dir', self.metrics_dir_path)

        for phrase in ('sil_peaks', 'ch_max', 'dbi_min', 'reasonable_sil', 'ch_dbi_tally', 'n_expected_clusters', 'dbs_err_dict'):
            if utils.find(f'*{phrase}*.pkl', self.metrics_dir_path): pass
            else:
                print(f'{utils.time_now()} - Not all metrices have been found in {self.metrics_dir_path}, generating them now...')
                # print all metrices if even 1 not found
                som_weights_to_nodes = utils.open_pickle(self.som_weights_to_nodes_path)

                ch_scores, dbi_scores = validation.print_elbow_CH_DBI_plot(self, som_weights_to_nodes)
                yellowbrick_expected_k = validation.print_yellowbrickkelbow(self, som_weights_to_nodes)
                silhouette_avgs, reasonable_silhoutte_scores_mt50 = validation.print_silhoutte_plots(self, som_weights_to_nodes)
                dbstop10 = validation.print_dbs_plots(self, som_weights_to_nodes)
                
                eps_ls, dbs_k_ls, dbs_noisepts_ls, dbs_labels = [], [], [], []
                for i in dbstop10:
                    eps_ls.append(i[0])
                    dbs_k_ls.append(i[1])
                    dbs_noisepts_ls.append(i[2])
                    dbs_labels.append(i[3])

                sil_peaks, ch_max, dbi_min, reasonable_sil, ch_dbi_tally, n_expected_clusters, dbs_err_dict = validation.get_cluster_determination_vars(
                    silhouette_avgs, ch_scores, dbi_scores, reasonable_silhoutte_scores_mt50, dbs_k_ls, dbs_noisepts_ls, yellowbrick_expected_k)

                for cluster_num in n_expected_clusters:
                    
                    save_dir = fr"{self.models_dir_path}/k-{cluster_num}"
                    if cluster_num == ch_max: save_dir += '_CHhighestPeak'
                    if cluster_num == dbi_min: save_dir += '_lowestDBItrough'
                    if cluster_num in sil_peaks: save_dir += '_SilhouetteAVG-peak'
                    if cluster_num == reasonable_sil: save_dir += '_mostReasonable-basedon-Silhouetteplot'
                    if cluster_num in ch_dbi_tally: save_dir += '_CHpeak-and-DBItrough'
                    if cluster_num == yellowbrick_expected_k: save_dir += '_Yellowbrickexpected-K'
                    if cluster_num in dbs_err_dict: save_dir += f'_DBSCANclusterErrorValsExpected-{dbs_err_dict[cluster_num]}'

                    os.makedirs(save_dir, exist_ok=True)

                self.ch_max_path = utils.to_pickle("ch_max", ch_max, self.metrics_dir_path)
                self.dbi_min_path = utils.to_pickle("dbi_min", dbi_min, self.metrics_dir_path)
                self.sil_peaks_path = utils.to_pickle("sil_peaks", sil_peaks, self.metrics_dir_path)
                self.reasonable_sil_path = utils.to_pickle("reasonable_sil", reasonable_sil, self.metrics_dir_path)
                self.ch_dbi_tally_path = utils.to_pickle("ch_dbi_tally", ch_dbi_tally, self.metrics_dir_path)
                self.yellowbrick_expected_k_path = utils.to_pickle("yellowbrick_expected_k", yellowbrick_expected_k, self.metrics_dir_path)
                self.dbs_err_dict_path = utils.to_pickle("dbs_err_dict", dbs_err_dict, self.metrics_dir_path)
                self.n_expected_clusters_path = utils.to_pickle("n_expected_clusters", n_expected_clusters, self.metrics_dir_path)

                break

        print(f'{utils.time_now()} - Internal validation of clusters has been run, please view metrices folder @:\n{self.metrics_dir_path} to determine optimal cluster number.\n'\
            f'\nYou can view the separate folders constructed for each discovered cluster combination. See @: \n{self.models_dir_path}.')

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
                    print(f'"cluster_dir" initialized @:\n{i}')
                    return best_cluster_num
        else:
            self.optimal_k = None
            return False


    def train_kmeans(self):
        if utils.find('*kmeans_model.pkl', self.cluster_dir):
            print(f'{utils.time_now()} - KMeans model trained before, skipping...')
            self.kmeans_model_path = utils.find('*kmeans_model.pkl', self.cluster_dir)[0]
            for phrase in ('labels_ar', 'labels_to_coords', 'label_markers', 'target_ds_withClusterLabels', 'dates_to_ClusterLabels', 'RFprec_to_ClusterLabels_dataset'):
                print(f'>>>>>>>>> "self.{phrase}_path" initialized.')
                exec(f'self.{phrase}_path = utils.find(f\'*{phrase}*.pkl\', self.cluster_dir)[0]')

        else:
            print(f'{utils.time_now()} - No KMeans model trained for {self.domain}, {self.period}, for {self.hyperparameters}, doing so now...')
            som_weights_to_nodes = utils.open_pickle(self.som_weights_to_nodes_path)
            samples, features = som_weights_to_nodes.shape
            km = KMeans(n_clusters=self.optimal_k).fit(som_weights_to_nodes)
            print(f"n{utils.time_now()} - K-means estimator fitted, sample size is {samples} and number of features is {features}.")

            self.kmeans_model_path = utils.to_pickle(f'{self.RUN_datetime}_kmeans_model', km, self.cluster_dir)
            self.serialize_kmeans_products(km)
            

    def serialize_kmeans_products(self, km):

        standardized_stacked_arr = utils.open_pickle(self.standardized_stacked_arr_path)
        target_ds = utils.open_pickle(self.target_ds_preprocessed_path)
        rf_ds_preprocessed_path = utils.open_pickle(self.rf_ds_preprocessed_path)

        labels_ar = km.labels_
        labels_to_coords = np.zeros([len(labels_ar), 2])
        for i, var in enumerate(labels_ar): labels_to_coords[i] = i % self.gridsize, i // self.gridsize

        try:
            label_markers = np.array([self.uniq_markers[var] for i, var in enumerate(labels_ar)])
        except IndexError: # more than 12 clusters
            label_markers = np.array([(self.uniq_markers*3)[var] for i, var in enumerate(labels_ar)])
        target_ds_withClusterLabels = target_ds.assign_coords(cluster=("time", km.predict(standardized_stacked_arr.astype(np.float))))
        dates_to_ClusterLabels = target_ds_withClusterLabels.cluster.reset_coords()
        RFprec_to_ClusterLabels_dataset = xr.merge([rf_ds_preprocessed_path, dates_to_ClusterLabels])

        self.labels_ar_path = utils.to_pickle(f'{self.RUN_datetime}_labels_ar', labels_ar, self.cluster_dir)
        self.labels_to_coords_path = utils.to_pickle(f'{self.RUN_datetime}_labels_to_coords', labels_to_coords, self.cluster_dir)
        self.label_markers_path = utils.to_pickle(f'{self.RUN_datetime}_label_markers', label_markers, self.cluster_dir)
        self.target_ds_withClusterLabels_path = utils.to_pickle(f'{self.RUN_datetime}_target_ds_withClusterLabels', target_ds_withClusterLabels, self.cluster_dir)
        self.dates_to_ClusterLabels_path = utils.to_pickle(f'{self.RUN_datetime}_dates_to_ClusterLabels', dates_to_ClusterLabels, self.cluster_dir)
        self.RFprec_to_ClusterLabels_dataset_path = utils.to_pickle(f'{self.RUN_datetime}_RFprec_to_ClusterLabels_dataset', RFprec_to_ClusterLabels_dataset, self.cluster_dir)
        

    def print_outputs(self):
        """
        if xx model output not found in self.cluster_dir with model.RUN_time at start of name
        """
        for phrase in ('_prelim_SOMscplot_', '_kmeans-scplot_', '_ARmonthfrac_', '_RFplot_', '_qp-at', '_rhum-at'):
            if utils.find(f'*{phrase}*.png', self.cluster_dir): pass
            else:
                print(f'{utils.time_now()} - Not all model outputs have been found in {self.cluster_dir}, generating them now...')
                # print all visuals if even 1 not found
                
                visualization.print_som_scatterplot_with_dmap(self);
                visualization.print_kmeans_scatterplot(self)
                self.grid_width = visualization.grid_width(self.optimal_k)
                self.cbar_pos = np.max([(((clus)//self.grid_width)*self.grid_width)
                                        for i,clus in enumerate(np.arange(self.optimal_k))
                                        if i==(((clus)//self.grid_width)*self.grid_width)])
                visualization.print_ar_plot(self)
                #FIXME: print_ar_plot_granular(self)
                visualization.print_ar_plot_granular(self)
                visualization.print_rf_plots(self)
                visualization.print_quiver_plots(self)
                visualization.print_rhum_plots(self)

                break
                
        print(
            f'Please open directory @: \n{self.cluster_dir}\n\nto view the clustering outputs for domain {self.domain} ({self.period}), '\
            f'trained at {self.dir_hp_str} \nand 2nd-level k-means clustered with k at {self.optimal_k}.')

    def nfold_cv_evaluation(self):
        pass


class AlphaLevelModel(TopLevelModel):
    """
    PSI = number of years taken as "ground-truth"/test set
    PSI_overlap = number of years overlapping, currently #FIXME
    """
    def __init__(self, domain, period, hyperparameters, domain_limits, PSI=3, PSI_overlap=0):
        super().__init__(domain, period, hyperparameters, domain_limits)
        self.alpha = 0
        if self.years % PSI:
            self.ALPHAs = (self.years // PSI) + 1
        else:
            self.ALPHAs = self.years // PSI

    def __str__(self):
        orig_str = super().__str__()
        return (f'{orig_str}\n===> Currently operating EVALUATION mode, self.ALPHAS is {self.ALPHAS}, & current alpha is: {self.alpha}.')

    def prepare_nfold_datasets(self): # i.e. split into different train/ground-truth(test) dataset
        pass

    def evaluation_procedure(self):
        """
        AUC, Brier, probabilities, etc.
        """
        pass


if __name__ == "__main__":
    a = TopLevelModel([-12,28,90,120], "NE_mon", [80, 10, 'train_batch', 4, .15, 0])
    a.train_SOM()
    print(a.month_names)
    a.detect_som_products()
    a.generate_k()
    a.get_k() 
    a.train_kmeans()
    a.print_outputs()
    a.nfold_cv_evaluation()
