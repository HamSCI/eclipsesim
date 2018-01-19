#!/bin/bash

# SGE Configuration
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -t 1-160

# Load modules
module load matlab/2016a

# Setup PHaRLAP
CWD=$(pwd)
PHARLAP_DIR="${CWD}/pharlap_4.1.3/"
export DIR_MODELS_REF_DAT="${PHARLAP_DIR}/dat"
export MATLABPATH=`find ${PHARLAP_DIR} -type d | sort | sed ':a;N;$!ba;s/\n/:/g'`

JOB_ID=$(expr $SGE_TASK_ID - 1)

JOB_PATH="./jobs/"
SAMI3_PATH="/afs/cad.njit.edu/research/physics/frissell/1/wb2jsv/sami3_mat/"
OUT_PATH="/afs/cad.njit.edu/research/physics/frissell/1/wb2jsv/eclipsesim_10hop/traces/"
PLOT_PATH="/afs/cad.njit.edu/research/physics/frissell/1/wb2jsv/eclipsesim_10hop/plots/"

./run_sim_sge.py ${JOB_ID} ${JOB_PATH} ${SAMI3_PATH} ${OUT_PATH} ${PLOT_PATH}
