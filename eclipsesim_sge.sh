#!/bin/bash

# SGE Configuration
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -t 1-160

# Load modules
module load matlab/2016a        # Load MATLAB 2016a

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

i=$(expr $SGE_TASK_ID - 1)

# Environment variables
PLOTS=0
ECLIPSE=0
SAMI3_PATH="./sami3/"
OUT_PATH="./traces/"
PLOT_PATH="./plots/"

# Run the program
matlab -nodisplay -r "eclipse('${jobs[${i}]}', ${i}, ${PLOTS}, ${ECLIPSE}, '${OUT_PATH}', '${PLOT_PATH}', '${SAMI3_PATH}'); exit;"
