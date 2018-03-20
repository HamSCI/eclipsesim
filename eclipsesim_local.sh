#!/bin/bash

# Setup PHaRLAP
CWD=$(pwd)
PHARLAP_DIR="${CWD}/pharlap_4.1.3/"
export DIR_MODELS_REF_DAT="${PHARLAP_DIR}/dat"
export MATLABPATH=`find ${PHARLAP_DIR} -type d | sort | sed ':a;N;$!ba;s/\n/:/g'`

# Environment variables
PLOTS=1
ECLIPSE=1

# Directories
JOB_PATH="./jobs/"
SAMI3_PATH="./sami3/"
OUT_PATH="./traces/"
PLOT_PATH="./plots/"

# Find our jobs
jobs=`ls -1 ${JOB_PATH}`
## Sort the jobs
IFS=$'\n'
for this_job in ${jobs}; do
    job_id=`./job_to_jobid "${this_job}"`

    # Run the program
#    matlab -nodisplay -r "dbstop in eclipse at 198; eclipse(${job_id}, ${PLOTS}, ${ECLIPSE}, '${JOB_PATH}', '${OUT_PATH}', '${PLOT_PATH}', '${SAMI3_PATH}'); exit;"
    matlab -nodisplay -r "eclipse(${job_id}, ${PLOTS}, ${ECLIPSE}, '${JOB_PATH}', '${OUT_PATH}', '${PLOT_PATH}', '${SAMI3_PATH}'); exit;"
done
unset IFS
