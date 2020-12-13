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
import time
import datetime
import xarray as xr
import numpy as np
from pathlib import Path
from timeit import default_timer as timer
from sklearn.preprocessing import minmax_scale, RobustScaler
import os, configparser


def get_files(directory, exts=['.nc', '.nc4']):
    ls = []
    for ext in exts:
        for i in sorted(Path(directory).glob(f'*{ext}')): ls.append(str(i))
    if ls:
        return ls
    else:
        raise 'Uh oh'
    
def is_NE_mon(month): return (month <=3) | (month==11) 

def is_SW_mon(month): return (month >=6) & (month <=9)

def is_inter_mon(month): return (month == 4) | (month == 5) | (month == 10) | (month == 11)

def save_preloaded_raw_data(model, raw_inputs_dir, raw_rf_dir):
    input_fpaths = get_files(raw_inputs_dir)
    rf_fpaths = get_files(raw_rf_dir)
    CHOSEN_VARS_ds = []
    for var in model.CHOSEN_VARS:
        CHOSEN_VARS_ds.append([rf"{_}" for _ in input_fpaths if f"{var}" in _])

    ds_CHOSEN_VARS_renamed = xr.open_mfdataset(CHOSEN_VARS_ds, chunks={'time':4}).rename({
        'latitude':'lat', 'longitude':'lon', 'r':'rhum', 'u':'uwnd', 'v':'vwnd'
    })
    ds_RAINFALL = xr.open_mfdataset(rf_fpaths)

    utils.to_pickle('ds_CHOSEN_VARS_renamed', ds_CHOSEN_VARS_renamed, raw_inputs_dir)
    utils.to_pickle('ds_RAINFALL', ds_RAINFALL, raw_rf_dir)
    return ds_CHOSEN_VARS_renamed, ds_RAINFALL

def prepare_dataset(model, dest):
    cfg = utils.scan_for_cfgfile()
    raw_inputs_dir = cfg.get('Paths', 'raw_inputs_dir')
    raw_rf_dir = cfg.get('Paths', 'raw_rf_dir')
    preloaded_pickles = utils.find('*.pkl', raw_inputs_dir)
    if preloaded_pickles:
        print('Preloaded raw data pickles found...', end=' ')
        ds_CHOSEN_VARS_renamed = utils.open_pickle(utils.find('*.pkl', raw_inputs_dir)[0])
        ds_RAINFALL = utils.open_pickle(utils.find('*.pkl', raw_rf_dir)[0])
    else: 
        print('Creating pickles of raw data...', end=' ')
        ds_CHOSEN_VARS_renamed, ds_RAINFALL = save_preloaded_raw_data(model, raw_inputs_dir, raw_rf_dir)
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
    print(ds_combined_sliced)
    print('\n\n\n\n', ds_RAINFALL)
    print('Pickling domain- & feature-constrained input & RF datasets...')
    if model.period == "NE_mon":
        input_ds = ds_combined_sliced.sel(time=is_NE_mon(ds_combined_sliced['time.month']))
        rf_ds = rf_ds_sliced.sel(time=is_NE_mon(ds_RAINFALL['time.month']))
        input_ds_serialized_path = utils.to_pickle('input_ds_NE_mon_serialized', input_ds, dest)
        rf_ds_serialized_path = utils.to_pickle('rf_ds_NE_mon_serialized', rf_ds, dest)
        return input_ds_serialized_path, rf_ds_serialized_path
    elif model.period == "SW_mon":
        input_ds = ds_combined_sliced.sel(time=is_SW_mon(ds_combined_sliced['time.month']))
        rf_ds = ds_RAINFALL.sel(time=is_SW_mon(ds_RAINFALL['time.month']))
        input_ds_serialized_path = utils.to_pickle('input_ds_SW_mon_serialized', input_ds, dest)
        rf_ds_serialized_path = utils.to_pickle('rf_ds_SW_mon_serialized', rf_ds, dest)
        return input_ds_serialized_path, rf_ds_serialized_path
    elif model.period == "inter_mon":
        input_ds = ds_combined_sliced.sel(time=is_inter_mon(ds_combined_sliced['time.month']))
        rf_ds = ds_RAINFALL.sel(time=is_inter_mon(ds_RAINFALL['time.month']))
        input_ds_serialized_path = utils.to_pickle('input_ds_inter_mon_serialized', input_ds, dest)
        rf_ds_serialized_path = utils.to_pickle('rf_ds_inter_mon_serialized', rf_ds, dest)
        return input_ds_serialized_path, rf_ds_serialized_path


def preprocess_time_series(model, dest, desired_res=0.75):   
    """
    Preparing datasets for use in training algorithms
    """
    target_ds = utils.open_pickle(model.input_ds_serialized_path)
    rf_target_ds = utils.open_pickle(model.rf_ds_serialized_path)

    # removing NA rows, supraneous dates, & coarsening dates accordingly
    print(f'{utils.time_now()} - Preprocessing data now.\n')
    
    # rf_target_ds['time'] =  rf_target_ds.indexes['time'].to_datetimeindex() #converting CFTimeIndex -> DateTime Index 

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
    print(f'Coarsen magnitude set at: {coarsen_magnitude}')
    target_ds_preprocessed = target_ds.coarsen(lat=coarsen_magnitude, lon=coarsen_magnitude, boundary='trim').mean()
        
    target_ds_preprocessed_path = utils.to_pickle('target_ds_preprocessed', target_ds_preprocessed, dest)
    rf_ds_preprocessed_path = utils.to_pickle('rf_ds_preprocessed', rf_ds_preprocessed, dest)
    return target_ds_preprocessed_path, rf_ds_preprocessed_path

def flatten_and_standardize_dataset(model, dest):

    target_ds_preprocessed = utils.open_pickle(model.target_ds_preprocessed_path)

    # reshaping
    reshapestarttime = timer(); print(f"{utils.time_now()} - Reshaping data now...")
    print(f"\n{utils.time_now()} - reshaping rhum dataarrays now, total levels to loop: {model.rhum_pressure_levels}.", end=' ')

    reshaped_unnorma_darrays = {}
    reshaped_unnorma_darrays['rhum'], reshaped_unnorma_darrays['uwnd'], reshaped_unnorma_darrays['vwnd'] = {}, {}, {}

    for level in model.rhum_pressure_levels:
        print(f'@{level}... ', end=' ')
        reshaped_unnorma_darrays['rhum'][level] = np.reshape(
            target_ds_preprocessed.rhum.sel(level=level).values, (model.n_datapoints, model.lat_size*model.lon_size ))

    print(f"\n{utils.time_now()} - reshaping uwnd/vwnd dataarrays now, total levels to loop: {model.uwnd_vwnd_pressure_lvls}.", end=' ')

    for level in model.uwnd_vwnd_pressure_lvls:
        print(f'@{level}... ', end=' ')
        reshaped_unnorma_darrays['uwnd'][level] = np.reshape(
            target_ds_preprocessed.uwnd.sel(level=level).values, (model.n_datapoints, model.lat_size*model.lon_size ))
        reshaped_unnorma_darrays['vwnd'][level] = np.reshape(
            target_ds_preprocessed.vwnd.sel(level=level).values, (model.n_datapoints, model.lat_size*model.lon_size ))

    reshapetime = timer()-reshapestarttime; reshapetime = str(datetime.timedelta(seconds=reshapetime)).split(".")[0]; print(f'Time taken: {reshapetime}s.\n')

    # stacking unstandardized dataarrays
    stackingstarttime = timer(); print("Stacking unstandardized dataarrays now...", end=' ')
    stacked_unstandardized_ds = np.hstack([reshaped_unnorma_darrays[var][lvl] for var in reshaped_unnorma_darrays for lvl in reshaped_unnorma_darrays[var]])

    stackingtime = timer()-stackingstarttime; stackingtime = str(datetime.timedelta(seconds=stackingtime)).split(".")[0]; print(f'Time taken: {stackingtime}s.\n')

    # standardizing the stacked dataarrays
    standardizestarttime = timer(); print("standardizing stacked dataarrays now...", end=' ')
    print(f'"stacked_unstandardized_ds.shape" is {stacked_unstandardized_ds.shape}')
    transformer = RobustScaler(quantile_range=(25, 75))
    standardized_stacked_arr = transformer.fit_transform(stacked_unstandardized_ds) # som & kmeans training
    transformer.get_params()
    standardizetime = timer()-standardizestarttime; standardizetime = str(datetime.timedelta(seconds=standardizetime)).split(".")[0]; print(f'That took {standardizetime}s to complete.\n')

    standardized_stacked_arr_path = utils.to_pickle('standardized_stacked_arr', standardized_stacked_arr, dest)

    return standardized_stacked_arr_path
