#!/usr/bin/env python3
import os
import datetime
import glob
from collections import OrderedDict

import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

import seqp

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
    #keys.append('tx_lat')
    #keys.append('tx_lon')
    #keys.append('rx_lat')
    #keys.append('rx_lon')
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

            # Identify unique times
            dates   = [x.to_pydatetime() for x in df_pair['datetime']]
            dates   = list(set(dates))
            dates.sort()
            for date in dates:
                for plt_inx,ionosphere in enumerate(['base','eclipse']):
                    tf      = np.logical_and(df_pair['datetime'] == date,df_pair['ionosphere'] == ionosphere)
                    df_tmp  = df_pair[tf]

                    vals    = df_tmp['rdall_ground_range']
                    bin_0   = 0
                    bin_1   = 8000
                    bin_stp = 100
                    bins    = np.arange(bin_0,bin_1,bin_stp)
                    weights = 1/(vals**3)
                    hist,bin_edges  = np.histogram(vals,bins=bins,weights=weights)

                    for hist_val,bin_edge in zip(hist,bin_edges[:-1]):
                        dct = OrderedDict()
                        dct['tx_call']      = tx_call
                        dct['rx_call']      = rx_call
                        dct['datetime']     = date
                        dct['freq']         = freq
                        dct['ionosphere']   = ionosphere
                        dct['range_km']     = bin_edge
                        dct['hist']         = hist_val
                        pwr_df_list.append(dct)

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
                fig = plt.figure(figsize=(10,8))

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

                date_str    = date.strftime('%Y%m%d.%H%M')
                fname       = '{}_histogram.png'.format(date_str)
                fpath       = os.path.join(out_dir,fname)

                fig.tight_layout()
                fig.savefig(fpath,bbox_inches='tight')
                plt.close(fig)

if __name__ == '__main__':
    df      = load_traces()
    df_pwr  = compute_ray_density(df)
    plot_power_histograms(df_pwr)
    import ipdb; ipdb.set_trace()
