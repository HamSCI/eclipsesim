#!/bin/bash

# SGE Configuration
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -t 1-121

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

# Directories
JOB_PATH="./jobs/"
SAMI3_PATH="/afs/cad.njit.edu/research/physics/frissell/1/wb2jsv/sami3_mat/"
OUT_PATH="/afs/cad.njit.edu/research/physics/frissell/1/wb2jsv/eclipsesim_dcc_output/traces/"
PLOT_PATH="/afs/cad.njit.edu/research/physics/frissell/1/wb2jsv/eclipsesim_dcc_output/plots/"

# Run the program (only execute at 15 minute intervals)
matlab -nodisplay -r "eclipse(${i}, ${PLOTS}, ${ECLIPSE}, '${JOB_PATH}', '${OUT_PATH}', '${PLOT_PATH}', '${SAMI3_PATH}'); exit;"
