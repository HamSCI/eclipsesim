#!/bin/bash

# Setup PHaRLAP
CWD=$(pwd)
PHARLAP_DIR="${CWD}/pharlap_4.1.3/"
export DIR_MODELS_REF_DAT="${PHARLAP_DIR}/dat"
export MATLABPATH=`find ${PHARLAP_DIR} -type d | sort | sed ':a;N;$!ba;s/\n/:/g'`

# Find our jobs
jobs=()
for entry in "./jobs"/*; do jobs+=("$entry"); done

# Sort the jobs
IFS=$'\n'
jobs=($(sort <<<"${jobs[*]}"))
unset IFS

# Environment variables
PLOTS=1
ECLIPSE=1

# Directories
JOB_PATH="./jobs/"
SAMI3_PATH="./sami3/"
OUT_PATH="./traces/"
PLOT_PATH="./plots/"

# Run the program
matlab -nodisplay -r "dbstop in eclipse at 92; eclipse(${jobs[0]}, ${PLOTS}, ${ECLIPSE}, '${JOB_PATH}', '${OUT_PATH}', '${PLOT_PATH}', '${SAMI3_PATH}'); exit;"
#matlab -nodisplay -r "eclipse(${jobs[0]}, ${PLOTS}, ${ECLIPSE}, '${JOB_PATH}', '${OUT_PATH}', '${PLOT_PATH}', '${SAMI3_PATH}'); exit;"
