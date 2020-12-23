"""
Y- "time.now()" function
- functions for parsing cli arguments, user-interface
- setting of cfg.ini for state of current RUN, based off "full-model" object
- loading of full-model object to acquire state
- logging of sys.stdout for both terminal output and also write-into another file
- functions for storing (1) arguments inputted
- (2) saving of objects/pickles
"""

from timeit import default_timer as timer
from pathlib import Path
import numpy as np
import argparse, time, pickle, fnmatch, datetime, os, configparser, logging

logger = logging.getLogger()
print = logger.info

default_cfgfile = Path(__file__).resolve().parents[0] / './cfg.ini'
logs_dir = Path(__file__).resolve().parents[1] / 'logs/'
prepared_data_folder = Path(__file__).resolve().parents[1] / "data/prepared"
models_dir = Path(__file__).resolve().parents[1] / 'models'
metrics_dir = Path(__file__).resolve().parents[1] / 'metrics'
raw_data_dir = Path(__file__).resolve().parents[1] / "data/raw"

def remove_expver(xr_ds):
    """
    16Dec401pm: interim solution to remove this one variable introduced in the newly downloaded ERA datasets, dimension expver, only seen in 2020 datasets.
    """
    try:
        xr_ds = xr_ds.drop('expver').isel(expver=0)
    except ValueError: # in event there is no "expver" dimension, usually the case for not-current-year ERA datasets
        pass
    return xr_ds

##def cut_year(xr_ds, year):
##    return xr_ds.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))

def cut_year(xr_ds, startyr, endyr):
    return xr_ds.sel(time=slice(f'{startyr}', f'{endyr}'))

def datetime_now():
    return time.strftime(f"%Y-%m-%d_%H-%M-%S", time.localtime())

def time_now():
    return time.strftime(f"%H-%M-%S", time.localtime())

def time_since(initial_timer):
    return str(datetime.timedelta(seconds=round(timer()-initial_timer, 2))).split(".")[0]

def create_cfgfile(newrun=False, cfgfile=default_cfgfile):
    cfg = configparser.ConfigParser()
    cfg.read(cfgfile)
    cfg['Paths'] = {}
    cfg['Paths']['raw_inputs_dir'] = os.fspath(Path(__file__).resolve().parents[1] / "data/raw/downloadERA")
    cfg['Paths']['raw_rf_dir'] = os.fspath(Path(__file__).resolve().parents[1] / "data/raw/GPM_L3")
    cfg['Top-level'] = {}
    if newrun: cfg['Top-level']['RUN started'] = datetime_now()
    else: cfg['Top-level']['RUN started'] = ''
    cfg['Top-level']['PSI'] = ''
    cfg['Top-level']['ALPHA'] = ''
    cfg['Alpha-level'] = {} # for yearly splits
    cfg['Alpha-level']['DELTA'] = ''
    cfg['Beta-level'] = {} # for bootstrap models
    cfg['Beta-level']['_i'] = ''
    cfg['Beta-level']['_beta'] = ''
    with open(cfgfile, 'w+') as f: cfg.write(f)
    return cfg

def update_cfgfile(section, key='', val='', cfgfile=default_cfgfile):
    cfg = configparser.ConfigParser()
    cfg.read(cfgfile)
    with open(logs_dir / f'{datetime_now()}_cfg.backup', 'w+') as f: cfg.write(f)
    if val:
        cfg[section][key] = str(val)
    else:
        if key:
            cfg[section][key] = str(val)
        else:
            cfg[section] = {}
    with open(cfgfile, 'w+') as f: cfg.write(f)

def scan_for_cfgfile(cfgfile=default_cfgfile):
    cfg = configparser.ConfigParser()
    cfg.read(cfgfile)
    if cfg.sections(): return cfg # cfg file exists
    else: return create_cfgfile()

def make_copy_cfgfile(cfgfile=default_cfgfile, logs_dir=logs_dir):
    cfg = configparser.ConfigParser()
    cfg.read(cfgfile)
    origin = cfg.get('Top-level', 'RUN started') 
    if origin:
        with open(f'{logs_dir}/{origin}_cfg.backup', 'w+') as f: cfg.write(f)

class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "program"
        return super().find_class(module, name)

def open_pickle(filepath):
    with open(filepath, 'rb') as f: 
        unpickler = MyCustomUnpickler(f)
        return unpickler.load()

def to_pickle(name, dataset, directory):
    extension = '.pkl'
    newpath = f'{directory}/{name}{extension}'
    with open(newpath, 'wb') as f:
        pickle.dump(dataset, f)
        print(f'Pickled "{name}" @ {newpath}')
    return newpath

def find(pattern, path):
    return [os.path.realpath(path)+'/'+f for f in os.listdir(path) if fnmatch.fnmatch(f, pattern)]

def get_cli_args(version):
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
                                     description='Provides arguments & parameters to boot-up the clustering model.'
                                     )
    parser.add_argument('objective', metavar='O', choices=['training', 'validation', 'evaluation', 'visualization'],
                        help='Objective of end-user, choices incl: training, validation, evaluation, visualization.')
    parser.add_argument('period', metavar='P', choices=['NE_mon', 'SW_mon', 'inter_mon', 'all'],
                        help='Monsoon period of model, choices incl: NE_mon (NDJFM), SW_mon (JJAS), inter_mon (AM, ON), all (JFMAMJJASOND).')
    parser.add_argument('domain', metavar='D', type=float, nargs=4,
                        help='Domain/extent/region of model (default is South, North, West, East), requires 4 arguments.')
    parser.add_argument('-md', '--multi-domain', metavar='{TOLERANCE}', type=int, dest='MD_TOLERANCE',
                        help='Indicate this OPTION to generate domain permutations, & TOLERANCE of lat (or lon) degrees for generating domain permutations. E.g. "-md 2"')
    parser.add_argument('cluster_num', metavar='k', type=int, default=0, nargs='*',
                        help='Indicate selected cluster(s) of domain for evaluation & validation (Not required for training & validation).')
    parser.add_argument('-bb', '--bounding-box', action='store_true',
                        help='Indicate this OPTION if the domain indicated are the dimensions of your bounding box (LL, LR, UL, UR).')
    args = parser.parse_args()
    if args.period == 'all': periods = ['NE_mon', 'SW_mon', 'inter_mon']
    else: periods = [args.period]
    if args.objective in ['evaluation', 'visualization'] and args.cluster_num == 0: raise NameError('"Cluster_num" positional argument needed for evaluation/visualization.')
    args.version = version
    print(vars(args), periods, version)
    return args

def parse_cli_args(args):
    ## indicate to user if sure, ask for y/n
    # if domain/bb dimensions are too small, 
    # indicate how many domains will be generated given the MD_TOLERANCE,
    # indicate chosen period
    # indicate
    pass

def create_model_params_template(file='model_args.txt'):
    ## * == optional, else: required=True, or is positional argument
    # purpose ('objective', metavar='O', choices=['training', 'validation', 'evaluation', 'visualization', ]
    # domain ('domain', metavar='D', type=float, nargs=4, help='Domain/extent/region of model (default is South, North, West, East; 4 values separated by space).')
    # either S, N, W, E, 
    # or LL, LR, UL, UR
    # * bounding_box ('-bb', '--bounding_box', action='store_true', help='Domain of model via bounding-box (LL, LR, UL, UR)')
    # multi-domain (bool - store_true)
    # seasons/periods (choices) => if "all", then periods = ['NE_mon', 'SW_mon', 'inter_mon']
    # cluster_num (int)
    pass

# def update_version(cfgfile=default_cfgfile):
#     cfg = configparser.ConfigParser()
#     cfg.read(cfgfile)
#     v, rev = (cfg['LOG']['version']).split('.')
#     if int(rev) != 99:
#         cfg['LOG']['version'] = f'{v}.{(int(rev)+1):02d}'
#     else:
#         cfg['LOG']['version'] = f'{str(int(v)+1)}.00'
#     with open(cfgfile, 'w') as f: cfg.write(cfgfile)

create_cfgfile(True)
# make_copy_cfgfile()
# parse_cli_args(get_cli_args(get_version(scan_for_cfgfile())))
# update_version()
