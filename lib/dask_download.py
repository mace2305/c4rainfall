import utils
from dask.distributed import wait, Client, LocalCluster
import os, sys, cdsapi, toolz, logging

logfmt = '\n# %(asctime)s %(filename)s|(%(module)s):\n%(message)s'
formatter = logging.Formatter(logfmt)
logging.basicConfig(level=logging.DEBUG, format=logfmt)
logger = logging.getLogger()
fh = logging.FileHandler(f'{utils.logs_dir}/DASK_DOWNLOAD_runtime_{utils.datetime_now()}.log', 'a')
fh.setFormatter(formatter)
logger.addHandler(fh)
print = logger.info
logging.getLogger('matplotlib.font_manager').disabled = True

def download_era(pe, c):
    print(f'Trying {pe}...')
    v, y = pe
    try:
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                "product_type": product_type,
                "format": fformat,
                "variable": v,
                "pressure_level": ['700', '850', '925'],
                "year": y,
                "month": ['01', '02', '03', '04', '05', '06', '07', '08', '09',
                          '10', '11', '12'],
                "day": ['01', '02', '03', '04', '05', '06', '07', '08', '09',
                        '10', '11', '12', '13', '14', '15', '16', '17', '18',
                        '19', '20', '21', '22', '23', '24', '25', '26', '27',
                        '28', '29', '30', '31'],
                "time": time,
                "area": area
            },
            f'{savedir}/{y}_{time_hr}-{v}.nc') 
    except:
        print('Passing, invalid!')
        pass

if __name__ == "__main__":
    """
    Only for the domain limits specified
    """
    lat_min, lat_max, lon_min, lon_max = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    print(f'Printing data for {lat_min} {lat_max} {lon_min} {lon_max}')
    
    variables = ['relative_humidity', 'u_component_of_wind', 'v_component_of_wind']
    years = ['1999', '2000', '2001','2002', '2003', '2005','2006', '2007', '2008','2009', 
    '2010', '2011','2012', '2013', '2014','2015', '2016', '2017','2018', '2019', '2020']
    time = '00:00'
    time_hr = time.split(':')[0]
    area = [lat_min, lat_max, lon_min, lon_max]
    fformat = 'netcdf'
    product_type = 'reanalysis'
    permutations = [(v,y) for v in variables for y in years]

    savedir = utils.raw_data_dir / f"downloadERA_{lat_min}_{lat_max}_{lon_min}_{lon_max}"
    os.makedirs(savedir, exist_ok=True)
    print(f'You can find the new raw input data @:\n{savedir}')

    local_cluster = LocalCluster(n_workers=7, threads_per_worker=1, memory_limit='4GB')
    client = Client(local_cluster)
    c = cdsapi.Client()
    partitions = toolz.partition_all(12, permutations)
    for i in partitions:
        futures = [client.submit(download_era, permutation, c) for permutation in i]
        wait(futures)
        client.gather(futures)
        client.restart()
    