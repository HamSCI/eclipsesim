#!/usr/bin/env python3
import os
import datetime
import pandas as pd
import numpy as np
import multiprocessing as mp

log_file    = 'logs/run_sim.txt'
JOB_PATH    = "./jobs/"
SAMI3_PATH  = "./sami3/"
OUT_PATH    = "./traces/"
PLOT_PATH   = "./plots/"
PLOTS       = 1
ECLIPSE     = 1

def job_to_jobid(job_fpath):
    sDate       = datetime.datetime(2017,8,21,16)

    job_name    = os.path.split(job_fpath)[-1]
    job_time    = datetime.datetime.strptime(job_name,'%Y-%m-%d %H:%M:%S.csv')

    diff_min    = (job_time - sDate).total_seconds()/60.
    job_id      = int(diff_min/3.)
    return job_id

def create_job_files():
    src_files   = []
    src_files.append(( 7.030,'naf_files/7_rxTxPairs.csv'))
    src_files.append((14.030,'naf_files/14_rxTxPairs.csv'))


    sDate   = datetime.datetime(2017,8,21,16)
    eDate   = datetime.datetime(2017,8,21,22)
    dt      = datetime.timedelta(minutes=3)

#    sDate   = datetime.datetime(2017,8,21,18)
#    eDate   = datetime.datetime(2017,8,21,18,6)
#    dt      = datetime.timedelta(minutes=3)

    dates       = []
    this_date   = sDate
    files       = []
    while this_date < eDate:
        fname   = this_date.strftime('%Y-%m-%d %H:%M:%S.csv')
        fpath   = os.path.join(JOB_PATH,fname)
        files.append(fpath)
        with open(fpath,'w') as fl:
            line    = 'RX_CALL,TX_CALL,MHZ,RX_LAT,RX_LON,TX_LAT,TX_LON'
            fl.write(line)

        for freq,src in src_files:
            src_df      = pd.read_csv(src)
            for rinx,row in src_df.iterrows():
                rx_call = row['call_0']
                rx_lat  = row['lat_0']
                rx_lon  = row['lon_0']

                tx_call = row['call_1']
                tx_lat  = row['lat_1']
                tx_lon  = row['lon_1']

                with open(fpath,'a') as fl:
                    fl.write('\n')
                    #line   = 'WE9V,AA2MF,14.030,42.5625,-88.0417,27.8125,-82.7917'
                    line    = ','.join([rx_call,tx_call,str(freq),
                                str(rx_lat),str(rx_lon),str(tx_lat),str(tx_lon)])
                    fl.write(line)

        this_date   += dt

    return files

def run_job(job):
    with open(log_file,'a') as fl:
        line = "{}: {!s}\n".format(job,datetime.datetime.now())
        fl.write(line)


    job_id  = job_to_jobid(job)
    if job_id < 0 or job_id > 159:
        return

    for ECLIPSE in [0,1]:
        matlab_cmd = "eclipse({!s},{!s},{!s},'{!s}','{!s}','{!s}','{!s}'); exit;".format(
                        job_id, PLOTS, ECLIPSE, JOB_PATH, OUT_PATH, PLOT_PATH, SAMI3_PATH)
        cmd = 'matlab -nodisplay -r "{!s}"'.format(matlab_cmd)
        print(cmd)
        os.system(cmd)

jobs    = create_job_files()

#for job in jobs:
#    run_job(job)

with open(log_file,'w') as fl:
    line = "Job started: {!s}\n".format(datetime.datetime.now())
    fl.write(line)

with mp.Pool(4) as pool:
    pool.map(run_job,jobs)
