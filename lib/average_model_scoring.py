import utils, run_test
import numpy as np
import sqlite3 as lite
from pathlib import Path
import io, os, logging

logger = logging.getLogger()
print = logger.info

def db_connect(db_path):
    conn = lite.connect(db_path)
    return conn


def initialize_tables(conn):
    
    sqlite_create_table_query1 = """
    CREATE TABLE IF NOT EXISTS actual_rf(
    sn integer NOT NULL,
    period text NOT NULL,
    domain text NOT NULL,
    cluster text NOT NULL,
    test_date text NOT NULL, 
    SG_only_gt1mm_actual binary,
    whole_grid_gt1mm_actual binary
    )
    """

    sqlite_create_table_query2 = """
    CREATE TABLE IF NOT EXISTS predicted_rf(
    period text NOT NULL,
    domain text NOT NULL,
    cluster text NOT NULL,
    SG_only_gt1mm_pred binary,
    whole_grid_gt1mm_pred binary
    )
    """
    
    cur = conn.cursor()
    try: 
        cur.execute(sqlite_create_table_query1)
        cur.execute(sqlite_create_table_query2)
        conn.commit()
    except:
        conn.rollback()
        raise RuntimeError("Uh oh, an error occurred ...")


def row_exists_actual_rf(conn, sn, period, domain, cluster, test_date):
    actual_sql = """
    SELECT sn FROM actual_rf where sn = ? AND period = ? AND domain = ? AND cluster = ? AND test_date = ?
    """
    cur = conn.cursor()
    cur.execute(actual_sql, (sn, period, domain, cluster, test_date))
    if cur.fetchone(): row_actual_exists = True
    else: row_actual_exists = False

    pred_sql = """
    SELECT period FROM predicted_rf where period = ? AND domain = ? AND cluster = ?
    """
    cur.execute(pred_sql, (period, domain, cluster))
    if cur.fetchone(): row_prediction_exists = True
    else: row_prediction_exists = False

    return row_actual_exists, row_prediction_exists


def insert_actual_rf_array(conn, sn, period, domain, cluster, test_date, SG_only_gt1mm_actual, whole_grid_gt1mm_actual):
    sql = """
    INSERT INTO actual_rf (sn, period, domain, cluster, test_date, SG_only_gt1mm_actual, whole_grid_gt1mm_actual)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    try: 
        cur = conn.cursor()
        cur.execute(sql, (sn, period, domain, cluster, test_date, SG_only_gt1mm_actual.dumps(), whole_grid_gt1mm_actual.dumps()))
        conn.commit()
        return cur.lastrowid
    except:
        # rollback all database actions since last commit
        conn.rollback()
        raise RuntimeError("Uh oh, an error occurred ...")


def insert_predicted_rf_array(conn, period, domain, cluster, SG_only_gt1mm_pred, whole_grid_gt1mm_pred):
    sql = """
    INSERT INTO predicted_rf (period, domain, cluster, SG_only_gt1mm_pred, whole_grid_gt1mm_pred)
    VALUES (?, ?, ?, ?, ?)
    """
    try: 
        cur = conn.cursor()
        cur.execute(sql, (period, domain, cluster, SG_only_gt1mm_pred.dumps(), whole_grid_gt1mm_pred.dumps()))
        conn.commit()
        return cur.lastrowid
    except:
        # rollback all database actions since last commit
        conn.rollback()
        raise RuntimeError("Uh oh, an error occurred ...")


def retrieve_and_insert_actual_RF_array(conn, all_test_prepared_data_dir, period, domain, test_date, sn, cluster, w_lim, e_lim, s_lim, n_lim):
    test_ds = utils.open_pickle(Path(all_test_prepared_data_dir) / f'{period}_mon_{domain}_prepared/RFprec_to_ClusterLabels_dataset.pkl')
    wholegrid_gt1mm_pred = (test_ds.sel(time=test_date).precipitationCal > 1).values
    SG_only_gt1mm_actual = (test_ds.precipitationCal.sel(lon=slice(w_lim, e_lim), lat=slice(s_lim, n_lim), time=test_date) > 1).values
    insert_actual_rf_array(conn, sn, period, domain, cluster, test_date, SG_only_gt1mm_actual, wholegrid_gt1mm_pred)


def retrieve_and_insert_predicted_RF_array(conn, period, domain, cluster, w_lim, e_lim, s_lim, n_lim):
    print(f'Inserting predicted_rf arr: {period}, {domain} - cluster: {cluster}')
    # pred_ds = utils.open_pickle(model.RFprec_to_ClusterLabels_dataset_path) ## NOT possible due to this only being for ONE domain-period
    pred_ds = utils.open_pickle([*utils.models_dir.glob(f'**/{domain}*/{period}*/k-*/*RFprec*.pkl')][0])
    data_pred_wholegrid = (pred_ds.where(pred_ds.cluster==int(cluster)-1, drop=True).precipitationCal > 1)
    data_pred_sgonly = (pred_ds.where(pred_ds.cluster==int(cluster)-1, drop=True).precipitationCal > 1).sel(lon=slice(w_lim, e_lim), lat=slice(s_lim, n_lim))
    whole_grid_gt1mm_pred = np.mean(data_pred_wholegrid, axis=0).values
    SG_only_gt1mm_pred = np.mean(data_pred_sgonly, axis=0).values
    insert_predicted_rf_array(conn, period, domain, cluster, SG_only_gt1mm_pred, whole_grid_gt1mm_pred)


def draw_SG_only_grids(conn, period, domain, dates_to_test, dest):
    with conn:
        cur = conn.cursor()
        sql = f"""
        SELECT A.SG_only_gt1mm_actual, B.SG_only_gt1mm_pred FROM actual_rf A INNER JOIN predicted_rf B
        ON (A.period = B.period and A.domain = B.domain and A.cluster = B.cluster)
        WHERE A.period = ? and A.domain = ?
        """
        cur.execute(sql, (period, domain))
        arrs = cur.fetchall()
    run_test.print_SG_only_brier_vs_MAE_plots(arrs, period, domain, dates_to_test, dest)
    


def main(successful_evals):

    db_path = Path(__file__).resolve().parents[1] / 'test/2021_Jan_28_testing2020randomdates/4Feb_comparingmodels/db.sqlite3'
    all_test_prepared_data_dir = Path(__file__).resolve().parents[1] / 'data/external/casestudytesting_29_Jan'

    # coordinates for SG grid
    w_lim = 103.5
    e_lim = 104.055
    s_lim = 1.1
    n_lim = 1.55

    conn = db_connect(db_path) # connect to the database

    initialize_tables(conn)

    test_pngs_dir = Path(os.getcwd()) / "test/2021_Jan_28_testing2020randomdates"
    test_pngs = [os.path.split(i)[-1] for i in test_pngs_dir.glob("*_mon_*clus_*_test_zscore_against_fullmodel*.png/")]

    for i, png  in enumerate(test_pngs):
        print(f'{utils.time_now()} - Searching {i}: {png}')
        test_date = png.split('\'')[1]
        str_split_underscore = png.split('_')
        domain = '_'.join((str_split_underscore[2], str_split_underscore[3], str_split_underscore[4], str_split_underscore[5]))
        period = str_split_underscore[0]
        cluster = str_split_underscore[7]
        sn = str_split_underscore[-2].split('sn')[-1]
        # print(f'{sn} {domain} {period} {cluster} ')
        # str_split_underscore = test_pngs[i-1].split('_')
        # domain = '_'.join((str_split_underscore[2], str_split_underscore[3], str_split_underscore[4], str_split_underscore[5]))
        # period = str_split_underscore[0]
        # cluster = str_split_underscore[7]
        # sn = str_split_underscore[-2].split('sn')[-1]

        row_actual_exists, row_prediction_exists = row_exists_actual_rf(conn, sn, period, domain, cluster, test_date)
        if not row_actual_exists:
            retrieve_and_insert_actual_RF_array(conn, all_test_prepared_data_dir, period, domain, test_date, sn, cluster, w_lim, e_lim, s_lim, n_lim)
        if not row_prediction_exists:
            retrieve_and_insert_predicted_RF_array(conn, period, domain, cluster, w_lim, e_lim, s_lim, n_lim)
    
    dates_to_test=len(successful_evals)
    brier_MAE_pngs = [os.path.split(i)[-1] for i in test_pngs_dir.glob(f"*_testdatesused_{dates_to_test}_brier_MAE_over_SG*.png/")]
    if len(brier_MAE_pngs) != dates_to_test:
        # successful_evals = period_domain
        for string in successful_evals:
            if string in brier_MAE_pngs: continue
            print(f'Generating brier_MAE for {string}')
            period = string.split('_')[0]
            domain = '_'.join(string.split('_mon_')[-1][1:-1].split(',')).replace(' ', '')
            draw_SG_only_grids(conn, period, domain, dates_to_test=dates_to_test, dest=test_pngs_dir)
    


