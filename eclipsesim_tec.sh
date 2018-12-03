#!/bin/bash

# Setup PHaRLAP
CWD=$(pwd)
PHARLAP_DIR="${CWD}/pharlap_4.1.3/"
export DIR_MODELS_REF_DAT="${PHARLAP_DIR}/dat"
export MATLABPATH=`find ${PHARLAP_DIR} -type d | sort | sed ':a;N;$!ba;s/\n/:/g'`

# Environment variables
MAKE_PLOTS=1
USE_TEC=1

# Directories
JOB_PATH="./jobs/"
OUT_PATH="./traces/"
PLOT_PATH="./plots/"
TEC_PATH="./gps181007.001.hdf5"

# Find our jobs
jobs=`ls -1 ${JOB_PATH}`

## Sort the jobs
IFS=$'\n'
for this_job in ${jobs}; do
    job_id=`./job_to_jobid "${this_job}"`

    # Run the program
    matlab -nodisplay -r "eclipse(${job_id}, ${MAKE_PLOTS}, ${USE_TEC}, '${JOB_PATH}', '${OUT_PATH}', '${PLOT_PATH}', '${TEC_PATH}'); exit;"
done

unset IFS
