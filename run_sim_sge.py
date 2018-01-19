#!/usr/bin/env python3

import os
import sys
import datetime

import pandas as pd

JOB_ID     = int(sys.argv[1])
PLOTS      = 1
JOB_PATH   = sys.argv[2]
SAMI3_PATH = sys.argv[3]
OUT_PATH   = sys.argv[4]
PLOT_PATH  = sys.argv[5]

src_files = []
src_files.append(( 7.030, 'naf_files/7_rxTxPairs.csv'))
src_files.append((14.030, 'naf_files/14_rxTxPairs.csv'))

sDate = datetime.datetime(2017, 8, 21, 16)

thisDate = sDate + datetime.timedelta(minutes=(3 * JOB_ID))
dateStr  = thisDate.strftime('%Y-%m-%d %H:%M:%S.csv')

#
# Create the job file for the current job
#
with open(os.path.join(JOB_PATH, dateStr), 'w') as f:
	f.write('RX_CALL,TX_CALL,MHZ,RX_LAT,RX_LON,TX_LAT,TX_LON')

	for freq, src in src_files:
		src_df = pd.read_csv(src)

		for ridx, row in src_df.iterrows():
			rx_call = row['call_0']
			rx_lat  = row['lat_0']
			rx_lon  = row['lon_0']
			tx_call = row['call_1']
			tx_lat  = row['lat_1']
			tx_lon  = row['lon_1']

			f.write('\n')
			f.write(','.join([
				rx_call, tx_call, str(freq), str(rx_lat), str(rx_lon),
				str(tx_lat), str(tx_lon),
			]))

#
# Execute the MATLAB code
#
for ECLIPSE in [0,1]:
        matlab_cmd = "eclipse({!s}, {!s}, {!s}, '{!s}', '{!s}', '{!s}', '{!s}'); exit;".format(
                JOB_ID, PLOTS, ECLIPSE, JOB_PATH, OUT_PATH, PLOT_PATH, SAMI3_PATH)
	cmd = 'matlab -nodisplay -r "{!s}"'.format(matlab_cmd)
	print(cmd)
	os.system(cmd)

