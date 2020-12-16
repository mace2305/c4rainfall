"""
main user interface for cli run
"""
import utils, modellers
import numpy as np
import sys, logging


# from utils import parse_args

domains_NE_mon = [
    [-10.0, 20.0, 90.0, 150.0], [-10.0, 26.0, 87.0, 143.0], [-13.5, 29.5, 92.0, 138.0], [-16.0, 32.0, 94.0, 136.0], 
    [-30.0, 18.0, 75.0, 168.0], [-30.0, 32.0, 75.0, 148.0], [-30.0, 42.0, 75.0, 138.0], [-5.0, 9.0, 95.0, 112.5], 
    [-6.0, 10.0, 96.0, 111.5], 
    [-10.0, 14.0, 90.0, 120.0]
]

domains_SW_mon = [
   [-4.0, 8.0, 93.5, 114.0], 
   [-5.0, 9.0, 95.0, 112.5], [-6.0, 10.0, 82.5, 127.5], [-6.0, 10.0, 96.0, 111.5], 
   [-10.0, 26.0, 87.0, 143.0], [-12.0, 16.0, 92.0, 118.0], [-12.0, 28.0, 90.0, 140.0], [-13.5, 29.5, 92.0, 138.0], 
   [-20.0, 20.0, 75.0, 125.0], [-20.0, 20.0, 75.0, 150.0], [-20.0, 20.0, 90.0, 140.0], [-20.0, 25.0, 75.0, 142.0], 
   [-20.0, 30.0, 90.0, 130.0], [-20.0, 35.0, 75.0, 130.0], [-30.0, 25.0, 65.0, 147.0], [-30.0, 25.0, 75.0, 130.0], 
   [-30.0, 30.0, 65.0, 140.0], [-30.0, 32.0, 75.0, 148.0], 
   [-30.0, 42.0, 75.0, 138.0]
]

domains_inter_mon = [
    [-10.0, 20.0, 90.0, 150.0], [-12.0, 28.0, 90.0, 140.0], [-13.5, 29.5, 92.0, 138.0], [-4.0, 8.0, 93.5, 114.0], 
    [-5.0, 9.0, 95.0, 112.5], [-6.0, 10.0, 82.5, 127.5]    
]

hpparam = [60000, 16, 'train_batch', 4, .15, 0]

logfmt = '\n# %(module)s (%(lineno)d):      %(message)s'
formatter = logging.Formatter(logfmt)
logging.basicConfig(level=logging.DEBUG, format=logfmt)
logger = logging.getLogger()
fh = logging.FileHandler(f'{utils.logs_dir}/runtime_{utils.datetime_now()}.log', 'a')
fh.setFormatter(formatter)
logger.addHandler(fh)
print = logger.info
logging.getLogger('matplotlib.font_manager').disabled = True

def trainSOM_getK(model):
    """
    Both training as well as generation (& picking) of metrics for optimal/"natural" cluster number.
    """
    model.detect_serialized_datasets()
    model.detect_prepared_datasets()
    model.train_SOM()
    model.detect_som_products()
    model.generate_k()
    cluster_num = model.get_k()
    return cluster_num

def trainKMeans_getOutputs(top_level_model, cluster_num):
    """
    Only for generation of cluster_num outputs for trained model!
    """
    top_level_model.train_kmeans()
    top_level_model.print_outputs()

def NFoldcrossvalidation_eval(alpha_level_model):
    alpha_level_model.prepare_nfold_datasets()
    for alpha in alpha_level_model.ALPHAs:
        alpha_level_model.detect_serialized_datasets(alpha)
        alpha_level_model.detect_prepared_datasets(alpha) 
        alpha_level_model.train_SOM(alpha)
        alpha_level_model.detect_som_products(alpha)
        alpha_level_model.generate_k(alpha)
        ## no need to run get_k since model.optimal_k is already set, and that is the configuration to evaluate!
        # alpha_level_model.get_k(alpha) 
        alpha_level_model.train_kmeans()
        ## FIXME the bread and butter of the evaluation process
        alpha_level_model.evaluation_procedure()
    
        
all_ds = [j for i in (domains_SW_mon, domains_NE_mon, domains_inter_mon) for j in i]
lat_min = np.min([i[0] for i in all_ds])
lat_max = np.max([i[1] for i in all_ds])
lon_min = np.min([i[2] for i in all_ds])
lon_max = np.max([i[3] for i in all_ds])
domain_limits = (lat_min, lat_max, lon_min, lon_max)


domains_NE_mon = [] # dummy list
seq_strings = "SW_mon", "NE_mon"

for i,d in enumerate((domains_SW_mon, domains_NE_mon)):
    perms = [(dims, seq_strings[i], hpparam, domain_limits) for dims in d]
    for p in perms: print(f'Generating optimal cluster number (k) for {seq_strings[i]}: {p}, ')
    for p in perms:
        try:
            model = modellers.TopLevelModel(domain=p[0], period=p[1], hyperparameters=p[2], domain_limits=p[3])
            cluster_num = trainSOM_getK(model)
            if cluster_num: # i.e. cluster found (i.e. meets "minimum_confidence", see modellers.get_k())
                trainKMeans_getOutputs(model, cluster_num)
                # modellers.AlphaLevelModel(p[0], p[1], p[2], p[3]) 
            else:
                print('\n==> Clustering configuration sub-optimal, no outputs will be generated for\n' \
                    f'{p[0]} trained with hpparams {p[2]}')
        except:
            logger.info('\n\n\nError:======================\n\n', exc_info=True)
            sys.exit()

"""
Below are means to get flattened, standardized arrays. This allows user to manually copy these "prepared" datasets.pkl
to other machines to carry out model training, instead of porting over raw data.
"""
# seq_strings = "SW_mon", "NE_mon", "inter_mon"

# for i,d in enumerate((domains_SW_mon, domains_NE_mon, domains_inter_mon)):
#     perms = [(dims, seq_strings[i], hpparam, domain_limits) for dims in d]
#     for p in perms: print(f'Generating optimal cluster number (k) for {seq_strings[i]}: {p}, ')
#     for p in perms:
#         try:
#             modellers.TopLevelModel(p[0], p[1], p[2], p[3])
#         except:
#             logger.info('\n\n\nError:======================\n\n', exc_info=True)
#             raise











    
