"""
- search for pre-opened xarray pickles
- load datasets
- subset datasets by (1) domain
- (2) remove extraneous/missing dates from datasets
- subset datasets into train/test splits
- reshaping + stacking dataarrays
- standardization/scaling of dataarrays
"""
import utils
import xarray as xr
import numpy as np
from pathlib import Path
from timeit import default_timer as timer
from sklearn.preprocessing import minmax_scale, RobustScaler
import os, time, datetime, configparser, logging, sys, cdsapi, toolz, subprocess, functools, gc

logger = logging.getLogger()
print = logger.info


def get_files(directory, exts=['.nc', '.nc4']):
    ls = []
    for ext in exts:
        for i in sorted(Path(directory).glob(f'*{ext}')): ls.append(str(i))
    if ls:
        return ls
    else:
        raise 'Uh oh'
    
def is_NE_mon(month): return (month <=3) | (month >=11) 

def is_SW_mon(month): return (month >=6) & (month <=9)

def is_inter_mon(month): return (month == 4) | (month == 5) | (month == 10) | (month == 11)

def get_raw_input_data(model):
    # these names are arbitrary. RF is gotten from NASA's GPM mission, hence the folder name.
    # and the input parameters (RHUM, uwnd, vwnd) are from ECMWF's ERA dataset
    lat_min, lat_max, lon_min, lon_max = model.domain_limits
    raw_input_subdirs = [i for i in utils.raw_data_dir.glob('*downloadERA*')]
    if raw_input_subdirs:
        for subdir in raw_input_subdirs:
            ltn, ltx, lnn, lnx = [float(num) for num in subdir.stem.split('_')[-4:]]
            print(f'Required domain limits: {lat_min} {lat_max} {lon_min} {lon_max}\n'\
                f'Current input data domain limits: {ltn} {ltx} {lnn} {lnx}')
            if (ltn <= lat_min) & (ltx >= lat_max) & (lnn <= lon_min) & (lnx >= lon_max):
                if utils.find('*.nc', subdir):
                    return subdir
                else: 
                    print(f'{subdir}\nis empty, no netcdf files found inside.')
    print('FAULTY'); sys.exit()
    print(f'{utils.datetime_now()} - No suitable raw input data found, attempting to spawn process for "dask_download.py".')
    filen = Path(__file__).resolve().parents[0] / "dask_download.py"
    cmd = f"python {filen} {lat_min} {lat_max} {lon_min} {lon_max}"
    pipe = subprocess.Popen(cmd.split(' '))
    pipe.wait()
    print(f'{utils.datetime_now()} - Raw input data has been successfully acquired.')

    new_path = utils.raw_data_dir / f"downloadERA_{lat_min}_{lat_max}_{lon_min}_{lon_max}"
    os.makedirs(new_path, exist_ok=True)
    return new_path


def get_raw_target_data(model):
    """
    more information can be acquired here: https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGDF_06/summary
    wget -r -c  -nH -nc -np -nd --user=XXX@EMAIL.com --password=XXX --auth-no-challenge --content-disposition -A nc4,xml "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.06/2007/"
    generating a list of the above wget sentence, and then running them through console/terminal.
    """
    lat_min, lat_max, lon_min, lon_max = model.domain_limits
    raw_rf_subdirs = [i for i in utils.raw_data_dir.glob('*GPM*')]
    print(f'raw_rf_subdirs: {raw_rf_subdirs}')

    # return raw_rf_subdirs[0] # NOT implementing the below first (16Dec 340pm) as time-strapped.
    # # originally, the GPM folder is named "GPM_L3_-90_90_-180_180", now is it simply "GPM_L3"
    # # when the model is rerun on clean folder, the prepared dataset will reference such a folder
    """
    20 Dec: note to self, NE & inter_mon are both GPM_L3, SW_mon is GPM_L3_-90_90_-180_180
    """

    if raw_rf_subdirs:
        for subdir in raw_rf_subdirs:
            try: 
                stems = [int(num) for num in subdir.stem.split('_')[-4:]]
                ltn, ltx, lnn, lnx = stems
                if (ltn < lat_min) & (ltx > lat_max) & (lnn < lon_min) & (lnx > lon_max):
                    return subdir
            except FileNotFoundError:
                return subdir
    else:
        raise(f'NO RAINFALL DATA FOUND @ {utils.raw_data_dir}')
    # FIXME: secondary download procedure for GPM_3IMERG
    # return new_dir


def prepare_dataset(model, dest):
    """
    - xr.open_mfdataset() = loading
    - restricting to certain variables + "levels" of variables
    - combining variables xarrays into one
    - restricting to only between 1999 to 2019
    - slicing domain dimensions up to required specs (i.e. model.LON_S, model.LON_N, etc...)
    - slicing up to chosen period only
    - pickling the datasets (both input & rainfall) & returning them
    """
    # searching for raw data pickles
    preloaded_input_pickles = utils.find('*.pkl', model.raw_input_dir)
    if preloaded_input_pickles:
        print('Preloaded raw INPUT data pickles found...')
        ds_CHOSEN_VARS_renamed = utils.open_pickle(utils.find('*.pkl', model.raw_input_dir)[0])
    else: 
        print('Creating pickles of raw input data...')
        ds_CHOSEN_VARS_renamed = save_preloaded_raw_input_data(model)
    
    preloaded_input_pickles = utils.find('*.pkl', model.raw_rf_dir)
    if preloaded_input_pickles:
        print('Preloaded raw rainfall data pickles found...')
        ds_RAINFALL = utils.open_pickle(utils.find('*.pkl', model.raw_rf_dir)[0])
    else: 
        print('Creating pickles of raw rainfall data...')
        ds_RAINFALL = save_preloaded_raw_rf_data(model)

    print("Proceeding to do preliminary data cleaning...")
    ds_sliced = ds_CHOSEN_VARS_renamed.sel(
        level=slice(np.min(model.unique_pressure_lvls),np.max(model.unique_pressure_lvls)), 
        lat=slice(model.LAT_N,model.LAT_S), lon=slice(model.LON_W,model.LON_E),
        time=slice('1999', '2019'))
    ds_sliced_rhum = ds_sliced.rhum
    ds_sliced_rhum_no925 = ds_sliced_rhum.drop_sel({"level":925})
    ds_sliced_uwnd_only = ds_sliced.uwnd
    ds_sliced_vwnd_only = ds_sliced.vwnd
    ds_combined_sliced = xr.merge([ds_sliced_rhum_no925, ds_sliced_uwnd_only, ds_sliced_vwnd_only], compat='override')

    rf_ds_sliced = ds_RAINFALL.sel(lat=slice(model.LAT_S, model.LAT_N), lon=slice(model.LON_W,model.LON_E))
    print('Pickling domain- & feature-constrained input & RF datasets...')
    if model.period == "NE_mon":
        input_ds = ds_combined_sliced.sel(time=is_NE_mon(ds_combined_sliced['time.month']))
        rf_ds = rf_ds_sliced.sel(time=is_NE_mon(ds_RAINFALL['time.month']))
        input_ds_serialized_path = utils.to_pickle('raw_input_ds_NE_mon_serialized', input_ds, dest)
        rf_ds_serialized_path = utils.to_pickle('raw_rf_ds_NE_mon_serialized', rf_ds, dest)
        return input_ds_serialized_path, rf_ds_serialized_path
    elif model.period == "SW_mon":
        input_ds = ds_combined_sliced.sel(time=is_SW_mon(ds_combined_sliced['time.month']))
        rf_ds = ds_RAINFALL.sel(time=is_SW_mon(ds_RAINFALL['time.month']))
        input_ds_serialized_path = utils.to_pickle('raw_input_ds_SW_mon_serialized', input_ds, dest)
        rf_ds_serialized_path = utils.to_pickle('raw_rf_ds_SW_mon_serialized', rf_ds, dest)
        return input_ds_serialized_path, rf_ds_serialized_path
    elif model.period == "inter_mon":
        input_ds = ds_combined_sliced.sel(time=is_inter_mon(ds_combined_sliced['time.month']))
        rf_ds = ds_RAINFALL.sel(time=is_inter_mon(ds_RAINFALL['time.month']))
        input_ds_serialized_path = utils.to_pickle('raw_input_ds_inter_mon_serialized', input_ds, dest)
        rf_ds_serialized_path = utils.to_pickle('raw_rf_ds_inter_mon_serialized', rf_ds, dest)
        return input_ds_serialized_path, rf_ds_serialized_path


def save_preloaded_raw_input_data(model):
    input_fpaths = get_files(model.raw_input_dir)
    CHOSEN_VARS_ds = []
    for var in model.CHOSEN_VARS:
        CHOSEN_VARS_ds.append([rf"{_}" for _ in input_fpaths if f"{var}" in _])

    ds_CHOSEN_VARS_renamed = xr.open_mfdataset(CHOSEN_VARS_ds, chunks={'time':4}).rename({
        'latitude':'lat', 'longitude':'lon', 'r':'rhum', 'u':'uwnd', 'v':'vwnd'
    })

    utils.to_pickle('ds_CHOSEN_VARS_renamed', ds_CHOSEN_VARS_renamed, model.raw_input_dir)
    return ds_CHOSEN_VARS_renamed


def save_preloaded_raw_rf_data(model):
    rf_fpaths = get_files(model.raw_rf_dir)

    ds_RAINFALL = xr.open_mfdataset(rf_fpaths)

    utils.to_pickle('ds_RAINFALL', ds_RAINFALL, model.raw_rf_dir)
    return ds_RAINFALL


def preprocess_time_series(model, dest, nfold_ALPHA=None, desired_res=0.75):   
    """
    Preparing datasets for use in training algorithms
    - dropping missing values
    - ensuring both target & input datasets have same dates 
    - coarsening spatial resolution of rainfall(target) dataset to desired resolution
    - pickling these "preprocessed" datasets
    """
    target_ds = utils.open_pickle(model.input_ds_serialized_path)
    rf_target_ds = utils.open_pickle(model.rf_ds_serialized_path)

    # removing NA rows, supraneous dates, & coarsening dates accordingly
    print(f'{utils.time_now()} - Preprocessing data now.')
    
    try:
        rf_target_ds['time'] =  rf_target_ds.indexes['time'].to_datetimeindex() #converting CFTimeIndex -> DateTime Index 
    except AttributeError:
        print('AttributeError: \'DatetimeIndex\' object has no attribute \'to_datetimeindex\', continuing regardless...')
        pass

    earliest_rf_reading, latest_rf_reading =  rf_target_ds.isel(time=0).time.values,  rf_target_ds.isel(time=-1).time.values
    earliest_target_ds_reading, latest_target_ds_reading = target_ds.isel(time=0).time.values, target_ds.isel(time=-1).time.values
    earliest_date = earliest_target_ds_reading if earliest_target_ds_reading > earliest_rf_reading else earliest_rf_reading
    latest_date = latest_target_ds_reading if latest_target_ds_reading < latest_rf_reading else latest_rf_reading

    rf_ds_preprocessed =  rf_target_ds.sel(time=slice(earliest_date, latest_date))
    target_ds = target_ds.sel(time=slice(earliest_date, latest_date))

    more_time_gaps = [i for i in target_ds.time.data if i not in rf_ds_preprocessed.time.data]
    more_time_gaps = more_time_gaps+[i for i in rf_ds_preprocessed.time.data if i not in target_ds.time.data]
    valid_dates = [date for date in target_ds.time.data if date not in more_time_gaps]
    target_ds = target_ds.sel(time = valid_dates)
    coarsen_magnitude = int(desired_res/np.ediff1d(target_ds.isel(lon=slice(0,2)).lon.data)[0])
    print(f'Coarsen magnitude set at: {coarsen_magnitude} toward desired spatial resolu. of {desired_res}')
    target_ds_preprocessed = target_ds.coarsen(lat=coarsen_magnitude, lon=coarsen_magnitude, boundary='trim').mean()
        
    target_ds_preprocessed_path = utils.to_pickle('target_ds_preprocessed', target_ds_preprocessed, dest)
    rf_ds_preprocessed_path = utils.to_pickle('rf_ds_preprocessed', rf_ds_preprocessed, dest)

    target_ds_preprocessed = utils.remove_expver(target_ds_preprocessed)

    if nfold_ALPHA:
        for alpha in range(nfold_ALPHA):
            pass

    return target_ds_preprocessed_path, rf_ds_preprocessed_path

def cut_dataset(model, alpha, dest, dataset_path, ds_name):
    dataset = utils.open_pickle(dataset_path)
    try: 
        dataset = dataset.sel(
            level=slice(np.min(model.tl_model.unique_pressure_lvls),np.max(model.tl_model.unique_pressure_lvls)), 
            lat=slice(model.tl_model.LAT_N, model.tl_model.LAT_S), lon=slice(model.tl_model.LON_W, model.tl_model.LON_E),
            time=slice('1999', '2019'))
    except ValueError:
        dataset = dataset.sel(
            lat=slice(model.tl_model.LAT_S, model.tl_model.LAT_N), lon=slice(model.tl_model.LON_W, model.tl_model.LON_E),
            time=slice('1999', '2019'))
    if model.tl_model.period == "NE_mon":
        dataset = dataset.sel(time=is_NE_mon(dataset['time.month']))
    elif model.tl_model.period == "SW_mon":
        dataset = dataset.sel(time=is_SW_mon(dataset['time.month']))
    elif model.tl_model.period == "inter_mon":
        dataset = dataset.sel(time=is_inter_mon(dataset['time.month']))

    if alpha != model.ALPHAs:
        gt_years = model.tl_model.years[(alpha-1)*model.PSI : alpha*model.PSI]
        train_years = np.delete(model.tl_model.years, np.arange((alpha-1) * model.PSI, alpha * model.PSI))
        test = utils.cut_year(dataset, np.min(gt_years), np.max(gt_years))
        train = utils.cut_year(dataset, np.min(train_years), np.max(train_years))
    else:
        gt_years = model.tl_model.years[(alpha-1)*model.PSI : alpha*model.PSI+model.runoff_years] 
        train_years = np.delete(model.tl_model.years, np.arange((alpha-1)*model.PSI, alpha*model.PSI+model.runoff_years)) 
        test = utils.cut_year(dataset, np.min(gt_years), np.max(gt_years))
        train = utils.cut_year(dataset, np.min(train_years), np.max(train_years))
    time.sleep(1); gc.collect()

    utils.to_pickle(f'{ds_name}_test_alpha_{alpha}_preprocessed', test, dest)
    utils.to_pickle(f'{ds_name}_train_alpha_{alpha}_preprocessed', train, dest)

def cut_target_dataset(model, alpha, dest):
    cut_dataset(model, alpha, dest, model.tl_model.target_ds_preprocessed_path, 'target_ds')
    print(f'Input dataset split into train/test sets for alpha-{alpha}')

def cut_rf_dataset(model, alpha, dest):
    cut_dataset(model, alpha, dest, model.tl_model.rf_ds_preprocessed_path, 'rf_ds')
    print(f'Rainfall dataset split into train/test sets for alpha-{alpha}')

def flatten_and_standardize_dataset(model, dest):

    target_ds_preprocessed = utils.open_pickle(model.target_ds_preprocessed_path)
    target_ds_preprocessed = utils.remove_expver(target_ds_preprocessed)
    
    # reshaping
    reshapestarttime = timer(); print(f"{utils.time_now()} - Reshaping data now...")
    print(f"\n{utils.time_now()} - reshaping rhum dataarrays now, total levels to loop: {model.rhum_pressure_levels}.")

    reshaped_unnorma_darrays = {}
    reshaped_unnorma_darrays['rhum'], reshaped_unnorma_darrays['uwnd'], reshaped_unnorma_darrays['vwnd'] = {}, {}, {}

    for level in model.rhum_pressure_levels:
        print(f'@{level}... ')
        reshaped_unnorma_darrays['rhum'][level] = np.reshape(
            target_ds_preprocessed.rhum.sel(level=level).values, (model.n_datapoints, model.lat_size*model.lon_size ))

    print(f"\n{utils.time_now()} - reshaping uwnd/vwnd dataarrays now, total levels to loop: {model.uwnd_vwnd_pressure_lvls}.")

    for level in model.uwnd_vwnd_pressure_lvls:
        print(f'@{level}... ')
        reshaped_unnorma_darrays['uwnd'][level] = np.reshape(
            target_ds_preprocessed.uwnd.sel(level=level).values, (model.n_datapoints, model.lat_size*model.lon_size ))
        reshaped_unnorma_darrays['vwnd'][level] = np.reshape(
            target_ds_preprocessed.vwnd.sel(level=level).values, (model.n_datapoints, model.lat_size*model.lon_size ))

    reshapetime = timer()-reshapestarttime; reshapetime = str(datetime.timedelta(seconds=reshapetime)).split(".")[0]; print(f'Time taken: {reshapetime}s.\n')

    # stacking unstandardized dataarrays
    stackingstarttime = timer(); print("Stacking unstandardized dataarrays now...")
    stacked_unstandardized_ds = np.hstack([reshaped_unnorma_darrays[var][lvl] for var in reshaped_unnorma_darrays for lvl in reshaped_unnorma_darrays[var]])

    stackingtime = timer()-stackingstarttime; stackingtime = str(datetime.timedelta(seconds=stackingtime)).split(".")[0]; print(f'Time taken: {stackingtime}s.\n')

    # standardizing the stacked dataarrays
    standardizestarttime = timer(); print("standardizing stacked dataarrays now...")
    print(f'"stacked_unstandardized_ds.shape" is {stacked_unstandardized_ds.shape}')
    transformer = RobustScaler(quantile_range=(25, 75))
    standardized_stacked_arr = transformer.fit_transform(stacked_unstandardized_ds) # som & kmeans training
    transformer.get_params()
    standardizetime = timer()-standardizestarttime; standardizetime = str(datetime.timedelta(seconds=standardizetime)).split(".")[0]; print(f'That took {standardizetime}s to complete.\n')

    standardized_stacked_arr_path = utils.to_pickle('standardized_stacked_arr', standardized_stacked_arr, dest)

    return standardized_stacked_arr_path
