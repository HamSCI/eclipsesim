#!/usr/bin/env python3
import os
import datetime
import multiprocessing as mp

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
    files   = []

    sDate   = datetime.datetime(2017,8,21,16)
    eDate   = datetime.datetime(2017,8,21,22)
    dt      = datetime.timedelta(minutes=3)

#    sDate   = datetime.datetime(2017,8,21,18)
#    eDate   = datetime.datetime(2017,8,21,18,6)
#    dt      = datetime.timedelta(minutes=3)

    dates       = []
    this_date   = sDate
    while this_date < eDate:
        fname   = this_date.strftime('%Y-%m-%d %H:%M:%S.csv')
        fpath   = os.path.join(JOB_PATH,fname)

        with open(fpath,'w') as fl:
            line    = 'RX_CALL,TX_CALL,MHZ,RX_LAT,RX_LON,TX_LAT,TX_LON'
            fl.write(line+'\n')
            line    = 'WE9V,AA2MF,14.030,42.5625,-88.0417,27.8125,-82.7917'
            fl.write(line)

        files.append(fpath)
        this_date   += dt

    return files

def run_job(job):
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

with mp.Pool(4) as pool:
    pool.map(run_job,jobs)
