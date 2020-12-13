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
import os
import calendar
import logging
from minisom import MiniSom
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from timeit import default_timer as timer
from pathlib import Path
import numpy as np
import xarray as xr

# logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
# logger.addHandler(logging.FileHandler(f'runtime_{utils.datetime_now()}.log', 'w'))
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

    def __init__(self, domain, period, hyperparameters, k=0):
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
        # evaluation params
        self.evaluated = False
        self.PSI = 0 # 3 means 3-year ground-truth, etc.
        self._totalALPHA = 0 
        self.ALPHA = 0 # i.d. of PSI split dataset
        
        print(self)
        TopLevelModel.detect_serialized_datasets(self)

    def __str__(self):
        string =f'' \
                f'-----------------------------------------------------------------------------------------------------------------\n' \
                f'TopLevelModel with parameters D:{self.domain}, P:{self.period}, HP:{self.hyperparameters}\n' \
                f'Run-time started on {self.RUN_datetime}'
        return string

    def detect_serialized_datasets(self):
        prepared_data_dir = utils.prepared_data_folder / self.dir_str
        try: os.mkdir(fr"{prepared_data_dir}")
        except OSError: pass # folder already exists

        prepared_data_dir = prepared_data_dir / self.period 
        try: os.mkdir(fr"{prepared_data_dir}")
        except OSError: pass # folder already exists
        self.prepared_data_dir = prepared_data_dir
        # utils.update_cfgfile('Paths', 'prepared_data_dir', self.prepared_data_dir)

        if len(utils.find('*serialized.pkl', self.prepared_data_dir)) == 2:
            print('This domain-period combination has been serialized before, loading objects...')
            for pkl in utils.find('*.pkl', self.prepared_data_dir):
                if "input_ds" in pkl: self.input_ds_serialized_path = pkl
                elif "rf_ds" in pkl: self.rf_ds_serialized_path = pkl
        else: 
            print('This domain-period combination has not been modelled before, proceeding to load & serialize raw data. ')
            self.input_ds_serialized_path, self.rf_ds_serialized_path = prepare.prepare_dataset(self, self.prepared_data_dir)
        print(f'Serialized input datasets @: {self.input_ds_serialized_path}\n'\
              f'Serialized RF datasets @: {self.rf_ds_serialized_path}')


        if utils.find('*preprocessed.pkl', self.prepared_data_dir) and utils.find('*standardized_stacked_arr.pkl', self.prepared_data_dir):
            print('Pickles (preprocessed) found.')
            for pkl in utils.find('*preprocessed.pkl', self.prepared_data_dir):
                if "target_ds" in pkl: self.target_ds_preprocessed_path = pkl
                elif "rf_ds" in pkl: self.rf_ds_preprocessed_path = pkl
            
            LocalModelParams(self, utils.open_pickle(self.target_ds_preprocessed_path))

            for pkl in utils.find('*standardized_stacked_arr.pkl', self.prepared_data_dir):
                self.standardized_stacked_arr_path = pkl
        else:
            print('No pre-processed pickles captured previously. Proceeding to load & process raw dataset pickles.')
            self.target_ds_preprocessed_path, self.rf_ds_preprocessed_path = prepare.preprocess_time_series(self, self.prepared_data_dir)

            LocalModelParams(self, utils.open_pickle(self.target_ds_preprocessed_path)) # generate new local model params

            self.standardized_stacked_arr_path = prepare.flatten_and_standardize_dataset(self, self.prepared_data_dir)

    def train_SOM(self):
        models_dir_path = utils.models_dir / self.dir_hp_str
        try: os.mkdir(fr"{models_dir_path}")
        except OSError: pass # folder already exists

        models_dir_path = models_dir_path / self.period
        try: os.mkdir(fr"{models_dir_path}")
        except OSError: pass # folder already exists
        self.models_dir_path = models_dir_path
        # utils.update_cfgfile('Paths', 'models_dir_path', self.models_dir_path)

        if utils.find('*som_model.pkl', self.models_dir_path):
            print(f'\n{utils.time_now()} - SOM model trained before, skipping...')
            self.som_model_path = utils.find('*som_model.pkl', self.models_dir_path)[0]
        else:
            print(f'\n{utils.time_now()} - No SOM model trained for {self.domain}, {self.period}, for {self.hyperparameters}, doing so now...')

            standardized_stacked_arr = utils.open_pickle(self.standardized_stacked_arr_path)

            sominitstarttime = timer(); print(f'{utils.time_now()} - Initializing MiniSom... ')
            som = MiniSom(self.gridsize, self.gridsize, # square
                        standardized_stacked_arr.shape[1],
                        sigma=self.sigma, learning_rate=self.learning_rate,
                        neighborhood_function='gaussian', random_seed=self.random_seed)
            som.pca_weights_init(standardized_stacked_arr)
            print(f"Initialization took {utils.time_since(sominitstarttime)}.\n")

            trainingstarttime = timer(); print(f"\n{utils.time_now()} - Beginning training.")
            getattr(som, self.training_mode)(standardized_stacked_arr, self.iterations, verbose=True)
            q_error = np.round(som.quantization_error(standardized_stacked_arr), 2)
            print(f"Training complete. Q error is {q_error}, time taken for training is {utils.time_since(trainingstarttime)}s\n")

            self.som_model_path = utils.to_pickle(f'{self.RUN_datetime}_som_model', som, models_dir_path)
            # utils.update_cfgfile('SOM', 'Train_date', utils.datetime_now()) # fucking key error
            # utils.update_cfgfile('SOM', 'Path', self.som_model_path)
    
    def detect_som_products(self):        
        for phrase in ('winner_coordinates', 'dmap', 'ar', 'som_weights_to_nodes'):
            if utils.find(f'*{phrase}.pkl', self.models_dir_path): 
                p = utils.find(f'*{phrase}.pkl', self.models_dir_path)
                print(f'\n{utils.time_now()} - {phrase} is found @ {p}')
                exec(f'self.{phrase}_path = {p}[0]')
            else:
                print(f'\n{utils.time_now()} - Some SOM products found missing in, generating all products now...')
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

                visualization.print_som_scatterplot_with_dmap(self)
        print('\nAll SOM products generated.')

    def generate_k(self):
        metrics_dir = utils.metrics_dir / self.dir_hp_str 
        try: os.mkdir(fr"{metrics_dir}")
        except OSError: pass # folder already exists

        metrics_dir = metrics_dir / self.period 
        try: os.mkdir(fr"{metrics_dir}")
        except OSError: pass # folder already exists
        self.metrics_dir_path = metrics_dir
        # utils.update_cfgfile('Paths', 'metrics_dir', self.metrics_dir_path)

        for phrase in ('sil_peaks', 'ch_max', 'dbi_min', 'reasonable_sil', 'ch_dbi_tally', 'n_expected_clusters', 'dbs_err_dict'):
            if utils.find(f'*{phrase}*.pkl', self.metrics_dir_path): pass
            else:
                print(f'\n{utils.time_now()} - Not all metrices have been found in {self.metrics_dir_path}, generating them now...')
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

                    try: 
                        os.mkdir(save_dir)
                        print(f'Creating {save_dir}...')
                    except OSError: pass # folder already exists

                self.ch_max_path = utils.to_pickle("ch_max", ch_max, self.metrics_dir_path)
                self.dbi_min_path = utils.to_pickle("dbi_min", dbi_min, self.metrics_dir_path)
                self.sil_peaks_path = utils.to_pickle("sil_peaks", sil_peaks, self.metrics_dir_path)
                self.reasonable_sil_path = utils.to_pickle("reasonable_sil", reasonable_sil, self.metrics_dir_path)
                self.ch_dbi_tally_path = utils.to_pickle("ch_dbi_tally", ch_dbi_tally, self.metrics_dir_path)
                self.yellowbrick_expected_k_path = utils.to_pickle("yellowbrick_expected_k", yellowbrick_expected_k, self.metrics_dir_path)
                self.dbs_err_dict_path = utils.to_pickle("dbs_err_dict", dbs_err_dict, self.metrics_dir_path)
                self.n_expected_clusters_path = utils.to_pickle("n_expected_clusters", n_expected_clusters, self.metrics_dir_path)

                break

        print(f'\n{utils.time_now()} - Internal validation of clusters has been run, please view metrices folder at:\n{self.metrics_dir_path} to determine optimal cluster number.')
        print(f'\nAlternatively, folders have been generated for each discovered cluster combination. See\n{self.models_dir_path} for these plots.')

if __name__ == "__main__":
    a = TopLevelModel([-12,28,90,120], "NE_mon", [80, 10, 'train_batch', 4, .15, 0])
    a.train_SOM()
    print(a.month_names)
    a.detect_som_products()
    a.generate_k()
    print(a.n_datapoints)    
