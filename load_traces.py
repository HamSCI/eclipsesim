#!/usr/bin/env python3
import os
import datetime
import glob
from collections import OrderedDict
import pickle
import multiprocessing as mp

import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

import mysql.connector

import seqp
import eclipse_calc

class MySqlEclipse(object):
    def __init__(self,user='hamsci',password='hamsci',host='localhost',database='seqp_analysis'):
        db          = mysql.connector.connect(user=user, password=password,host=host, database=database)
        crsr        = db.cursor()

        qry         = '''
                      CREATE TABLE IF NOT EXISTS eclipse_obscuration (
                      lat DECIMAL(10,4),
                      lon DECIMAL(10,4),
                      height INT,
                      datetime DATETIME,
                      obscuration FLOAT
                      );
                      '''
        crsr.execute(qry)
        db.commit()

        self.db     = db
mysql_ecl = MySqlEclipse()

def bin_inner_loop(run_dct):
    df_date     = run_dct.get('df_date')
    tx_call     = run_dct.get('tx_call')
    rx_call     = run_dct.get('rx_call')
    tx_lat      = run_dct.get('tx_lat')
    tx_lon      = run_dct.get('tx_lon')
    azm         = run_dct.get('azm')
    date        = run_dct.get('date')
    freq        = run_dct.get('freq')

    result_list = []
    for plt_inx,ionosphere in enumerate(['base','eclipse']):
        tf      = df_date['ionosphere'] == ionosphere
        df_tmp  = df_date[tf]

        vals    = df_tmp['rdall_ground_range']
        bin_0   = 0
        bin_1   = 8000
        bin_stp = 100
        bins    = np.arange(bin_0,bin_1,bin_stp)
        weights = 1/(vals**3)
        hist,bin_edges  = np.histogram(vals,bins=bins,weights=weights)

        for hist_val,bin_edge in zip(hist,bin_edges[:-1]):
            result              = seqp.geopack.greatCircleMove(
                                    tx_lat,tx_lon,bin_edge/2.,azm)
            mid_lat             = float(result[0])
            mid_lon             = float(result[1])
            
#                        mid_obsc_300km      = float(eclipse_calc.calculate_obscuration(
#                                                date,mid_lat,mid_lon,height=300e3))

            mid_obsc_300km      = get_eclipse_obscuration(mid_lat,mid_lon,date,height=300e3)

            dct = OrderedDict()
            dct['tx_call']      	= tx_call
            dct['rx_call']      	= rx_call
            dct['tx_lat']       	= tx_lat
            dct['tx_lon']       	= tx_lon
            dct['azm']          	= azm
            dct['datetime']     	= date
            dct['freq']         	= freq
            dct['ionosphere']   	= ionosphere
            dct['range_km']     	= bin_edge
            dct['hist']         	= hist_val
            dct['mid_lat']      	= mid_lat
            dct['mid_lon']      	= mid_lon
            dct['mid_obsc_300km']   = mid_obsc_300km
            result_list.append(dct)
    return result_list

def get_eclipse_obscuration(lat,lon,date,height=300e3):
    user        = 'hamsci'
    password    = 'hamsci'
    host        = 'localhost'
    database    = 'seqp_analysis'
    db          = mysql.connector.connect(user=user,password=password,host=host,database=database)
    
    slat    = '{:.4F}'.format(lat)
    slon    = '{:.4F}'.format(lon)
    sheight = '{:.0F}'.format(height/1000.)
    sdate   = date.strftime('%Y-%m-%d %H:%M:%S')

    qry     = ('SELECT obscuration FROM eclipse_obscuration '
               'WHERE lat={} AND lon={} and height={} and datetime="{}"'.format(slat,slon,sheight,sdate))
    crsr    = db.cursor()
    crsr.execute(qry)
    result  = crsr.fetchone()
    crsr.close()

    if result is None:
        ob          = float(eclipse_calc.calculate_obscuration(date,lat,lon,height))

        add_ecl     = ("INSERT INTO eclipse_obscuration "
                           "(lat,lon,height,datetime,obscuration)"
                           "VALUES (%s, %s, %s, %s, %s)")

        data_ecl    = (slat,slon,sheight,sdate,ob)

        crsr        = db.cursor()
        crsr.execute(add_ecl,data_ecl)
        db.commit()
        crsr.close()
    else:
        ob  = result[0]
    db.close()
    return ob

def load_traces():
    trace_dirs  = {}
    trace_dirs['base']      = 'traces/base'
    trace_dirs['eclipse']   = 'traces/eclipse'

    df  = pd.DataFrame()
    for key,path in trace_dirs.items():
        files   = glob.glob(os.path.join(path,'all_*.csv'))

        for fpath in files:
            date_str    = os.path.split(fpath)[-1][-23:-4]
            date        = datetime.datetime.strptime(date_str,'%Y-%m-%d %H:%M:%S')

            df_tmp      = pd.read_csv(fpath)
            df_tmp['datetime']      = date
            df_tmp['ionosphere']    = key

            df  = df.append(df_tmp,ignore_index=True)

    keys = []
    keys.append('datetime')
    keys.append('tx_call')
    keys.append('rx_call')
    keys.append('tx_lat')
    keys.append('tx_lon')
    keys.append('rx_lat')
    keys.append('rx_lon')
    keys.append('freq')
    keys.append('ionosphere')
    keys.append('rdall_lat')
    keys.append('rdall_lon')
    keys.append('rdall_ground_range')
    keys.append('rdall_group_range')
    #keys.append('rdall_phase_path')
    #keys.append('rdall_geometric_path_length')
    #keys.append('rdall_initial_elev')
    #keys.append('rdall_final_elev')
    #keys.append('rdall_apogee')
    keys.append('rdall_gnd_rng_to_apogee')
    #keys.append('rdall_plasma_freq_at_apogee')
    #keys.append('rdall_virtual_height')
    #keys.append('rdall_effective_range')
    #keys.append('rdall_deviative_absorption')
    #keys.append('rdall_TEC_path')
    #keys.append('rdall_Doppler_shift')
    #keys.append('rdall_Doppler_spread')
    #keys.append('rdall_frequency')
    keys.append('rdall_nhops_attempted')
    keys.append('rdall_hop_idx')
    keys.append('rdall_apogee_lat')
    keys.append('rdall_apogee_lon')

    df  = df[keys]
    df  = df.dropna()
    return df

def compute_ray_density(df):
    pwr_df_list = []
    # Frequencies
    freqs   = df['freq'].unique()
    for freq in freqs:
        df_freq = df[df['freq'] == freq]

        # Idenitify TX/RX pairs
        pairs   = set(tuple(zip(df_freq.tx_call.tolist(),df_freq.rx_call.tolist())))
        for pair in pairs:
            tx_call = pair[0]
            rx_call = pair[1]

            tf      = np.logical_and(df_freq.tx_call == tx_call, df_freq.rx_call == rx_call)
            df_pair = df_freq[tf]

            # Get TX/RX lat/lons and azimuths
            tx_lat  = df_pair['tx_lat'].iloc[0]
            tx_lon  = df_pair['tx_lon'].iloc[0]
            rx_lat  = df_pair['rx_lat'].iloc[0]
            rx_lon  = df_pair['rx_lon'].iloc[0]

            azm     = seqp.geopack.greatCircleAzm(tx_lat,tx_lon,rx_lat,rx_lon) % 360.

            # Identify unique times
            dates   = [x.to_pydatetime() for x in df_pair['datetime']]
            dates   = list(set(dates))
            dates.sort()
            df_date_lst = []
            for date in dates:
                tf      = df_pair['datetime'] == date
                df_date = df_pair[tf]

                rdct    = {}
                rdct['df_date']         = df_date
                rdct['tx_call']      	= tx_call
                rdct['rx_call']      	= rx_call
                rdct['tx_lat']       	= tx_lat
                rdct['tx_lon']       	= tx_lon
                rdct['azm']          	= azm
                rdct['date']         	= date
                rdct['freq']         	= freq
                df_date_lst.append(rdct)

#            for run_dct in df_date_lst:
#                result      = bin_inner_loop(run_dct)
#                pwr_df_list += result
            
            with mp.Pool() as pool:
                results  = pool.map(bin_inner_loop,df_date_lst)
            for result in results:
                pwr_df_list += result


    df_pwr  = pd.DataFrame(pwr_df_list)
    return df_pwr

def plot_power_histograms(df_pwr):
    out_dir = 'plots/histograms'
    seqp.gen_lib.prep_output({0:out_dir},clear=True)

    freqs   = df_pwr['freq'].unique()
    for freq in freqs:
        df_freq = df_pwr[df_pwr['freq'] == freq]

        # Idenitify TX/RX pairs
        pairs   = set(tuple(zip(df_freq.tx_call.tolist(),df_freq.rx_call.tolist())))
        for pair in pairs:
            tx_call = pair[0]
            rx_call = pair[1]

            tf      = np.logical_and(df_freq.tx_call == tx_call, df_freq.rx_call == rx_call)
            df_pair = df_freq[tf]

            # Identify unique times
            dates   = [x.to_pydatetime() for x in df_pair['datetime']]
            dates   = list(set(dates))
            dates.sort()
            for date in dates:
                fig         = plt.figure(figsize=(10,8))

                this_dir    = os.path.join(out_dir,
                                '{:.3f}_{}_{}'.format(freq,rx_call,tx_call))
                seqp.gen_lib.prep_output({0:this_dir})

                
                for plt_inx,ionosphere in enumerate(['base','eclipse']):
                    tf      = np.logical_and(df_pair['datetime'] == date,df_pair['ionosphere'] == ionosphere)
                    df_tmp  = df_pair[tf]

                    # Histogram plotting...
                    xx      = df_tmp['range_km'].tolist()
                    yy      = df_tmp['hist'].tolist()
                    dx      = xx[1]-xx[0]

                    ax      = fig.add_subplot(2,1,plt_inx+1)
                    ax.bar(xx,yy,dx,align='edge')
                    ax.set_xlim(xx[0],xx[-1]+dx)
                    ax.set_ylim(0,6e-9)
                    ax.set_xlabel('Ground Range [km]')
                    ax.set_ylabel('Number')

                    title   = []
                    tmp     = 'TX: {} RX: {} {:.3f} MHz'.format(tx_call,rx_call,freq)
                    title.append(tmp)

                    date_str    = date.strftime('%Y %b %d %H%M UT')
                    tmp         = '{} - {}'.format(ionosphere.title(),date_str)
                    title.append(tmp)
                    ax.set_title('\n'.join(title))

                fdate_str   = date.strftime('%Y%m%d.%H%M')
                fname       = '{}_histogram.png'.format(fdate_str)
                fpath       = os.path.join(this_dir,fname)

                fig.tight_layout()
                fig.savefig(fpath,bbox_inches='tight')
                plt.close(fig)

if __name__ == '__main__':
    use_cache   = True

    cache_file  = 'df_pwr.p'
    if not use_cache:
        df      = load_traces()
        df_pwr  = compute_ray_density(df)
        with open(cache_file,'wb') as fl:
            pickle.dump(df_pwr,fl)
    else:
        with open(cache_file,'rb') as fl:
            df_pwr  = pickle.load(fl)

    print('Plotting histograms...')
    plot_power_histograms(df_pwr)
    import ipdb; ipdb.set_trace()
