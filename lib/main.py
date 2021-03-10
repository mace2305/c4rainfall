#!/home/yixiang/anaconda3/envs/SOM/bin/python
"""
main user interface for cli run
"""
import utils, modellers, average_model_scoring
import numpy as np
import sys, logging


domains_NE_mon = [
    [-6.0, 10.0, 96.0, 111.5], # 248.0 DONE
    [-16.0, 32.0, 94.0, 136.0], # 2016.0 DONE
    
## Attempted but no viable clustering
#   [-10.0, 20.0, 90.0, 150.0], [-10.0, 26.0, 87.0, 143.0], [-13.5, 29.5, 92.0, 138.0], [-30.0, 18.0, 75.0, 168.0]
#   [-30.0, 32.0, 75.0, 148.0], [-30.0, 42.0, 75.0, 138.0], [-5.0, 9.0, 95.0, 112.5], [-10.0, 14.0, 90.0, 120.0]
]

domains_SW_mon = [
   [-4.0, 8.0, 93.5, 114.0], # 246.0 DONE
   [-6.0, 10.0, 82.5, 127.5], # 720.0 DONE
   [-12.0, 16.0, 92.0, 118.0], # 728.0 DONE
   [-30.0, 25.0, 65.0, 147.0], # 4510.0 DONE
   [-30.0, 42.0, 75.0, 138.0], # 4536.0 DONE
   [-30.0, 30.0, 65.0, 140.0], # 4526.0 left alpha_general gridded pngs
   
## Attempted but no viable clustering   
#    [-5.0, 9.0, 95.0, 112.5], [-6.0, 10.0, 96.0, 111.5], [-10.0, 26.0, 87.0, 143.0], [-12.0, 28.0, 90.0, 140.0], [-13.5, 29.5, 92.0, 138.0], 
#    [-20.0, 20.0, 75.0, 125.0], [-20.0, 20.0, 75.0, 150.0], [-20.0, 20.0, 90.0, 140.0], [-20.0, 25.0, 75.0, 142.0], [-20.0, 30.0, 90.0, 130.0], 
#    [-20.0, 35.0, 75.0, 130.0], [-30.0, 32.0, 75.0, 148.0], [-30.0, 25.0, 75.0, 130.0], 
]

domains_inter_mon = [
    [-6.0, 10.0, 82.5, 127.5], # 720.0 DONE
    [-12.0, 28.0, 90.0, 140.0], # 2000.0 DONE

## Attempted but no viable clustering    
#    [-4.0, 8.0, 93.5, 114.0], [-5.0, 9.0, 95.0, 112.5],  [-10.0, 20.0, 90.0, 150.0], [-13.5, 29.5, 92.0, 138.0], 
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
    """
    Sequence of evaluation procedure
    1. prepare_nfold_datasets(): split into years according to PSI & PSI_overlap
    2. check_evaluated(): skip in scores already generated for that split/n^th-fold
    3. else, for this alpha level = 
    - detect_serialized_datasets(): is there such data
    - detect_prepared_datasets(): is there such cleaned datasets
    - train_SOM() & detect_som_products(): has SOM model & products already been generated
    - generate_k(): using internal val. to create folders of clustering quality
    - train_kmeans(): using optimal_k to generate kmeans model
    - print_outputs(): print outputs to see differences across the ALPHAs
    - evaluation_procedure(): generating AUC, Brier, etc. scores for this alpha split
    4. compile_scores(): generating an overall score for this clustering attempt (domain, period, hpparam)
    """
    finished_evaluation = alpha_level_model.check_evaluated()
    if finished_evaluation: 
        print(f'Evaluation has already been finished & you can find the evaluation results @:\n{finished_evaluation}')
        return 
    
    print(f'Commencing evaluation!')
    alpha_level_model.prepare_nfold_datasets()

    for alpha in range(1, alpha_level_model.ALPHAs+1):
        evaluated = alpha_level_model.check_evaluated(alpha)
        if evaluated:
            continue
        
        alpha_level_model.prepare_alphafold_dataset(alpha)
        alpha_level_model.train_SOM(alpha)
        alpha_level_model.detect_som_products(alpha)
        alpha_level_model.generate_k(alpha)
        alpha_level_model.train_kmeans(alpha)
        #alpha_level_model.print_outputs(alpha)
        alpha_level_model.evaluation_procedure(alpha)
    utils.to_pickle('alpha_level_model', alpha_level_model, alpha_level_model.alpha_general_dir)
    alpha_level_model.compile_scores()
    

all_ds = [j for i in (domains_NE_mon, domains_SW_mon, domains_inter_mon) for j in i]
lat_min = np.min([i[0] for i in all_ds])
lat_max = np.max([i[1] for i in all_ds])
lon_min = np.min([i[2] for i in all_ds])
lon_max = np.max([i[3] for i in all_ds])
domain_limits = (lat_min, lat_max, lon_min, lon_max)


seq_strings = ("NE_mon", "inter_mon", "SW_mon")
successful_evals = []

for i,d in enumerate((domains_NE_mon, domains_inter_mon, domains_SW_mon)):
    perms = [(dims, seq_strings[i], hpparam, domain_limits) for dims in d]
    for p in perms: print(f'Generating optimal cluster number (k) for {seq_strings[i]}: {p}, ')
    for p in perms:
        try:
            tl_model = modellers.TopLevelModel(domain=p[0], period=p[1], hyperparameters=p[2], domain_limits=p[3])
            print(tl_model)
            cluster_num = trainSOM_getK(tl_model)
            if cluster_num: # i.e. cluster found (i.e. meets "minimum_confidence", see modellers.get_k())
                trainKMeans_getOutputs(tl_model, cluster_num)
                alpha_model = modellers.AlphaLevelModel(tl_model, p[0], p[1], p[2], p[3], PSI=3, PSI_overlap=0);
                print(alpha_model)
                NFoldcrossvalidation_eval(alpha_model)
                print(f'### Evaluation completed for domain {p[0]} in {p[1]}.\n')
                successful_evals.append(f'{p[1]}_{p[0]}') # period, domain
            else:
                print('\n==> Clustering configuration sub-optimal, no outputs will be generated for\n' \
                    f'{p[0]} trained with hpparams {p[2]}')
            dates_to_test = 10
            tl_model.test_random_dates(dates_to_test)
            print(f"{dates_to_test} dates have been randomly selected for testing.\n\n\n")
        except:
            logger.info('\n\n\nError:======================\n\n', exc_info=True)
            sys.exit() # exit if exception arises
        # sys.exit() # debugging
        

average_model_scoring.main(successful_evals)
print('Relevant arrays have been stashed in sqlite3 database.')








    
