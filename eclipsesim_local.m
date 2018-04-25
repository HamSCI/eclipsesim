clear
% Environment variables
PLOTS   = 1;

% Directories
JOB_ID      = 46;
JOB_PATH    = "./jobs/";
SAMI3_PATH  = "./sami3/";
OUT_PATH    = "./traces/";
PLOT_PATH   = "./plots/";

% ECLIPSE = 0;
% eclipse(JOB_ID, PLOTS, ECLIPSE, JOB_PATH, OUT_PATH, PLOT_PATH, SAMI3_PATH);

ECLIPSE = 1;
eclipse(JOB_ID, PLOTS, ECLIPSE, JOB_PATH, OUT_PATH, PLOT_PATH, SAMI3_PATH);
