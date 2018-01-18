#!/usr/bin/env python3
import os
import datetime
import glob
from collections import OrderedDict
import multiprocessing as mp

import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import mysql.connector

import seqp
import eclipse_calc

import seaborn as sns

bandObj     = seqp.maps.BandData()
rcp = mpl.rcParams
rcp['figure.titlesize']     = 'xx-large'
rcp['axes.titlesize']       = 'xx-large'
rcp['axes.labelsize']       = 'xx-large'
rcp['xtick.labelsize']      = 'xx-large'
rcp['ytick.labelsize']      = 'xx-large'
rcp['legend.fontsize']      = 'large'

rcp['figure.titleweight']   = 'bold'
rcp['axes.titleweight']     = 'bold'
rcp['axes.labelweight']     = 'bold'

Re  = 6371.
hgt = 300.

# Parameter Dictionary
prmd = {}

tmp = {}
tmp['label']            = 'Midpoint Obscuration at 300 km Altitude'
tmp['lim']              = (0,1.05)
prmd['obs_mid_300km']   = tmp

tmp = {}
tmp['label']            = 'Mean Great Circle Hop Length [km]'
tmp['lim']              = (0.,4500.)
prmd['R_gc_mean']       = tmp

tmp = {}
tmp['label']            = 'Great Circle Distance [km]'
#tmp['lim']              = (0.,17500.)
tmp['lim']              = (0.,8000.)
#tmp['lim']              = (0.,5000.)
tmp['vmin']             = 0.
tmp['vmax']             = 10000.
prmd['R_gc']       = tmp

tmp = {}
tmp['label']            = 'Frequency [MHz]'
tmp['vmin']             = 0.
tmp['vmax']             = 30.
tmp['cmap']             = mpl.cm.jet
prmd['frequency']       = tmp

tmp = {}
tmp['label']            = 'SNR [dB]'
tmp['lim']              = (0.,100.)
tmp['vmin']             = 0.
tmp['vmax']             = 50.
tmp['cmap']             = mpl.cm.viridis
prmd['srpt_0']          = tmp

#tmp = {}
#tmp['label']            = 'Weighted Ray Denisty'
#tmp['lim']              = (0.,6e-9)
#tmp['vmin']             = 0.
#tmp['vmax']             = 4e-9
#tmp['cmap']             = mpl.cm.viridis
#prmd['hist']          = tmp

tmp = {}
tmp['label']            = 'Weighted Ray Denisty'
tmp['lim']              = (-250,0)
tmp['vmin']             = -110
tmp['vmax']             = -75
tmp['cmap']             = mpl.cm.viridis
prmd['hist']         = tmp

tmp = {}
tmp['label']            = 'N Hops'
tmp['vmin']             = 0.
tmp['vmax']             = 5.
tmp['cmap']             = mpl.cm.jet
prmd['N_hops']          = tmp

class SeabornFig2Grid():
    """From https://stackoverflow.com/questions/35042255/how-to-plot-multiple-seaborn-jointplot-in-subplot"""
    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())

def count_vals(x,y):
    assert x.size == y.size
    return x.size

def roundup100(x):
    result  = int(np.ceil(x/100.) * 100)
    return result

def seaborn_scatter(df,x_key,y_key,band,c_key=None,ckey_ascending=True,kind='scatter',data_set=None,alpha=None,**kwargs):
    date_0      = df.datetime.min()
    date_1      = df.datetime.max()

    xx      = df[x_key]
    xpd     = prmd.get(x_key,{})
    x_label = xpd.get('label',x_key)
    xlim    = xpd.get('lim',None)

    yy      = df[y_key]
    ypd     = prmd.get(y_key,{})
    y_label = ypd.get('label',y_key)
    ylim    = ypd.get('lim',None)

    dft     = df.copy()
    dft.dropna(subset=[x_key,y_key],inplace=True)
    dft.sort_values(c_key,inplace=True,ascending=ckey_ascending)

    joint_kws = {}
    if kind == 'scatter' and c_key is not None:
        c           = dft[c_key]
        cpd         = prmd.get(c_key,{})
        cmap        = cpd.get('cmap',mpl.cm.jet)
        vmin        = cpd.get('vmin',min(c))
        vmax        = cpd.get('vmax',max(c))
        cbar_label  = cpd.get('label',c_key)
        joint_kws   = dict(c=c,vmin=vmin,vmax=vmax,cmap=cmap,color=None)

    g       = sns.jointplot(x=x_key,y=y_key,data=dft,kind=kind,size=8,xlim=xlim,ylim=ylim,
                    stat_func=None,joint_kws=joint_kws)
                
    g.annotate(count_vals,stat='N',loc='upper right',template='{stat}: {val:g}')
    g.set_axis_labels(x_label,y_label)

    if data_set == 'eclipse':
        title = []
        date_0_str  = date_0.strftime('%d %b %Y %H%M')
        date_1_str  = date_1.strftime('%H%M UT')
        date_str    = '{}-{}'.format(date_0_str,date_1_str)

        title.append('{} {}'.format(bandObj.band_dict[band]['freq_name'],data_set.title()))
        title.append(date_str)
    else:
        title = []
        date_str  = ( date_0.strftime('%d %b  - ') + date_1.strftime('%d %b %Y')
                    + date_0.strftime(' %H%M-')   + date_1.strftime('%H%M UT') )

        title.append('{} {}'.format(bandObj.band_dict[band]['freq_name'],data_set.title()))
        title.append(date_str)

    sax = g.ax_marg_x
#    g.ax_marg_x.set_title('\n'.join(title),loc='right')
    if alpha is not None:
#        g.ax_marg_x.set_title('({})'.format(alpha),loc='left',fontsize=36)
        sax.text(-0.3,0.5,'({})'.format(alpha),transform=sax.transAxes,fontweight='bold',fontsize=36,va='center')


    # Histogram Ticklabels
    ssize   = 'medium'

    sax = g.ax_marg_x
    sylim   = ( 0, roundup100(sax.get_ylim()[1]) )
    sax.set_ylim(sylim)
    sax.set_yticks(sylim)
    for tl in sax.get_yticklabels():
        tl.set_visible(True)
        tl.set_fontsize(ssize)

    sax = g.ax_marg_y
    sxlim   = ( 0, roundup100(sax.get_xlim()[1]) )
    sax.set_xlim(sxlim)
    sax.set_xticks(sxlim)
    for tl in sax.get_xticklabels():
        tl.set_visible(True)
        tl.set_fontsize(ssize)

    return g

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
        hist,bin_edges  = np.histogram(vals,bins=bins)

#        weights = 1/(vals**3)
#        hist,bin_edges  = np.histogram(vals,bins=bins,weights=weights)
#        hist    = 10*np.log10(hist)

        for hist_val,bin_edge in zip(hist,bin_edges[:-1]):
            result              = seqp.geopack.greatCircleMove(
                                    tx_lat,tx_lon,bin_edge/2.,azm)
            mid_lat             = float(result[0])
            mid_lon             = float(result[1])
            
#                        obs_mid_300km      = float(eclipse_calc.calculate_obscuration(
#                                                date,mid_lat,mid_lon,height=300e3))

            obs_mid_300km      = get_eclipse_obscuration(mid_lat,mid_lon,date,height=300e3)

            dct = OrderedDict()
            dct['tx_call']      	= tx_call
            dct['rx_call']      	= rx_call
            dct['tx_lat']       	= tx_lat
            dct['tx_lon']       	= tx_lon
            dct['azm']          	= azm
            dct['datetime']     	= date
            dct['freq']         	= freq
            dct['ionosphere']   	= ionosphere
            dct['R_gc']     	        = bin_edge
            dct['hist']         	= hist_val
            dct['mid_lat']      	= mid_lat
            dct['mid_lon']      	= mid_lon
            dct['obs_mid_300km']        = obs_mid_300km
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
                    xx      = df_tmp['R_gc'].tolist()
                    yy      = df_tmp['hist'].tolist()
                    dx      = xx[1]-xx[0]

                    ydct    = prmd['hist']
                    ylim    = ydct['lim']

                    ax      = fig.add_subplot(2,1,plt_inx+1)
#                    ax.bar(xx,yy,dx,align='edge')
                    ax.bar(xx,np.abs(yy),dx,ylim[0],align='edge')

                    ax.set_xlim(xx[0],xx[-1]+dx)
                    ax.set_ylim(ylim)
                    ax.set_xlabel('Ground Range [km]')
                    ax.set_ylabel(ydct['label'])

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

def plot_scatterplots(df_pwr):
    df_pwr          = df_pwr.rename(columns={'freq':'frequency'})
    df_pwr['band']  = np.floor(df_pwr['frequency'])
    df_pwr['band']  = df_pwr['band'].map(int)

    # Define Bands
    bands = [7, 14]

    # Define Plot Keys
    x_key           = 'obs_mid_300km'
    y_key           = 'R_gc'
    c_key           = 'hist'
    ckey_ascending  = True

    kind            = 'scatter'

    # Define and load in Eclipse and Control datasets.
    print('Loading data from CSV files...')
    data_sets   = OrderedDict()
    dsd = {}
    data_sets['eclipse']    = dsd
    
    dsd = {}
    data_sets['control']    = dsd

    for data_set,dsd in data_sets.items():
        if data_set == 'eclipse':
            ionosphere = 'eclipse'
        else:
            ionosphere = 'base'

        tf          = df_pwr['ionosphere'] == ionosphere
        df          = df_pwr[tf]

        tf          = df['obs_mid_300km'] > 0.
        df          = df[tf].copy()

        df          = df.sort_values('frequency')
        dsd['df']   = df


    alphas   = ['a','b','c','d']
    # Create run list/dictionaries.
    sgs = []
    plot_nr = 0
    for inx,band in enumerate(bands):
        for data_set,dsd in data_sets.items():
            alpha   = alphas[plot_nr]
            df      = dsd['df']

            this_df     = df[df.band == band]
            band_tag    = bandObj.band_dict[band]['name'].replace(' ','')

            dct = {}
            dct['data_set']         = data_set
            dct['df']               = this_df
            dct['x_key']            = x_key
            dct['y_key']            = y_key
            dct['c_key']            = c_key
            dct['ckey_ascending']   = ckey_ascending 
            dct['kind']             = kind
            dct['band']             = band
            dct['alpha']            = alpha

            if len(this_df) == 0:
                sg = None
            else:
                sg  = seaborn_scatter(**dct)
            sgs.append(sg)
            plot_nr     += 1

    fig     = plt.figure(figsize=(20,20))
    gspecs  = gridspec.GridSpec(2,2)
    for sg,gspec in zip(sgs,gspecs):
        if sg is not None:
            mg  = SeabornFig2Grid(sg,fig,gspec)

    left    = 0.30
    width   = 0.40
    bottom  = 0.005
    height  = 0.025

    cpd         = prmd.get(c_key,{})
    cmap        = cpd.get('cmap',mpl.cm.jet)
    vmin        = cpd.get('vmin')
    vmax        = cpd.get('vmax')
    cbar_label  = cpd.get('label',c_key)
    rect    = (left,bottom,width,height)
    cax     = fig.add_axes(rect)
    norm    = mpl.colors.Normalize(vmin,vmax)
    cbar    = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal')
    cbar.set_label(cbar_label)

    fontsize = 36
    text    = 'Eclipse'
    xpos    = 0.25
    ypos    = 0.865
    fig.text(xpos,ypos,text,fontsize=fontsize,fontweight='bold',ha='center')

    text    = 'Control'
    xpos    = 0.70
    fig.text(xpos,ypos,text,fontsize=fontsize,fontweight='bold',ha='center')

    xpos    = 0.0200
    ypos    = 0.65
    text    = '7 MHz'
    fig.text(xpos,ypos,text,fontsize=fontsize,fontweight='bold',va='center',rotation=90.)

    ypos    = 0.225
    text    = '14 MHz'
    fig.text(xpos,ypos,text,fontsize=fontsize,fontweight='bold',va='center',rotation=90.)

    top     = 0.85
    lft     = 0.88
    fig.text(0.0,0.0,u"\u00B7")
    fig.text(0.0,top,u"\u00B7")
    fig.text(lft,top,u"\u00B7")
    fig.text(lft,0.0,u"\u00B7")

    fpath   = os.path.join('plots','sami3_seqp_scatter.png')
    fig.savefig(fpath,bbox_inches='tight')

def plot_sea(df_pwr,sea=True):
    df_pwr          = df_pwr.rename(columns={'freq':'frequency','rx_call':'call_0','tx_call':'call_1'})
    df_pwr['band']  = np.floor(df_pwr['frequency'])
    df_pwr['band']  = df_pwr['band'].map(int)

    map_left    = 0.
    map_width   = 0.50

    ts_left     = 0.55
    ts_width    = 0.45

    h_pad       = 0.05

    height      = 0.4
    v_pad       = 0.1

    ts_left     = map_width + h_pad
    fig = plt.figure(figsize=(20.5,12))

    pdct    = {}
    tmp     = {}
    tmp['csv']  = 'naf_files/7_rxTxPairs.csv'
    pdct[7]     = tmp
    
    tmp     = {}
    tmp['csv']  = 'naf_files/14_rxTxPairs.csv'
    pdct[14]    = tmp

    alphabet    = ['(c)','(d)','(a)','(b)']
    plot_nr     = 0
    for inx,band in enumerate([14,7]):
        df_pair = pd.read_csv(pdct[band]['csv'],parse_dates=['ecl_maxtime','partial_end','partial_start'])

        this_bottom = inx*(height+v_pad) + v_pad
        map_ax      = fig.add_axes([map_left,this_bottom,map_width,height])
        ax          = map_ax

        row_0       = df_pair.iloc[0]
        call_0      = row_0['call_0']
        lat_0       = row_0['lat_0']
        lon_0       = row_0['lon_0']

        sDate       = datetime.datetime(2017,8,21,14)
        eDate       = datetime.datetime(2017,8,21,22)

        map_prm = {}
        map_prm['llcrnrlon'] = -130.
        map_prm['llcrnrlat'] = 20
        map_prm['urcrnrlon'] = -60.
        map_prm['urcrnrlat'] = 55.

        hmap    = seqp.maps.HamMap(sDate,eDate,ax,show_title=False,**map_prm)
        hmap.overlay_gridsquares(label_precision=0,major_style={'color':'0.8','dashes':[1,1]})

        m_zorder    = 100
        tx_size     = 50
        rx_size     = 50
        lw          =  3

#        c0_lbl      = '{!s} ({!s})'.format(call_0,df_pair.n_spots.sum())
        c0_lbl      = call_0
        hmap.m.scatter(lon_0,lat_0,marker='^',s=rx_size,label=c0_lbl,color='k',zorder=m_zorder,lw=lw)

        for inx,row in df_pair.iterrows():
            c1          = row['call_1']

            n_spots     = row['n_spots']
            lat_1       = row['lat_1']
            lon_1       = row['lon_1']
            color       = row['color']

#            c1_lbl      = '{!s} ({!s})'.format(c1,n_spots)
            c1_lbl      = c1
            hmap.m.scatter(lon_1,lat_1,marker='*',s=tx_size,color=color,zorder=m_zorder,label=c1_lbl,lw=lw)

            hmap.m.scatter(row['lon_mid'],row['lat_mid'],marker='o',s=20,color=color,zorder=m_zorder,lw=lw)
            hmap.m.drawgreatcircle(lon_0,lat_0,lon_1,lat_1,color=color,lw=2)

#        title       = ['{!s} RBN Pairs'.format(call_0)] + self.title
#        fontdict    = {'weight':'bold','size':'x-large'}
#        ax.text(0.5,1.025,'\n'.join(title),fontdict=fontdict,transform=ax.transAxes,ha='center')

        leg = ax.legend(loc='lower left',ncol=3)
        frame   = leg.get_frame()
        leg.set_zorder(100)
#        pair_obj.plot_pair_map(map_ax)
        letter  = alphabet[plot_nr]
        letter_dct = {'fontweight':'bold','fontsize':22,'zorder':500}
        let_xpos    = 0.01
        let_ypos    = 0.925
        map_ax.text(let_xpos,let_ypos,letter,transform=map_ax.transAxes,**letter_dct)
        plot_nr += 1

        ################################################################################ 
        ################################################################################ 

        ts_ax   = fig.add_axes([ts_left,this_bottom,ts_width,height])
#        pair_obj.plot_pair_ts(ts_ax)
        ax = ts_ax
        row_0       = df_pair.iloc[0]
        call_0      = row_0['call_0']

        spans_0     = []
        spans_1     = []

        df      = df_pwr[df_pwr.band == band]
        y_key   = 'hist'
        for inx,row in df_pair.iterrows():
            c1          = row['call_1']
            n_spots     = row['n_spots']
            color       = row['color']
            R_gc        = row['R_gc']

            c1_lbl      = '{!s} ({!s})'.format(c1,n_spots)

            tf          = np.logical_and(df['call_0']==call_0,df['call_1']==c1)
            df_tmp      = df[tf]

            drgc        = np.array(np.abs(df_tmp['R_gc']-R_gc).tolist())
            tf          = drgc < 50 
            df_tmp      = df_tmp[tf]

            df_tmp      = df_tmp.sort_values('datetime')

            xx          = np.array([x.to_pydatetime() for x in df_tmp['datetime']])

            if sea:
                xx  = xx - row['ecl_maxtime']
                xx  = [x.total_seconds()/3600. for x in xx]

                span_0  = (row['partial_start'] - row['ecl_maxtime']).total_seconds()/3600.
                span_1  = (row['partial_end']   - row['ecl_maxtime']).total_seconds()/3600.
                spans_0.append(span_0)
                spans_1.append(span_1)

            yy  = np.array(df_tmp[y_key].tolist())
#            yy  = (yy - np.nanmean(yy))/np.nanstd(yy)
#            yy  = (yy - np.nanmean(yy))

            ax.plot(xx,yy,ls=' ',marker='.',color=color,label=None)

        ylabel  = prmd[y_key].get('label',y_key)
        ax.set_ylabel(ylabel)

        if sea:
            span_0  = np.min(spans_0)
            span_1  = np.max(spans_1)
            ax.axvspan(span_0,span_1,color='0.7',label='Partial Eclipse')
            ax.axvline(0,lw=2,ls='--',color='k',label='Maximum Eclipse')
            ax.set_xlabel('Eclipse Epoch Time [hr]')
            ax.set_xlim(-4,4)
        else:
            ax.set_xlabel('Time [UT]')
            ax.set_xlim(datetime.datetime(2017,8,21,14),datetime.datetime(2017,8,21,22))
            xts     = [mpl.dates.num2date(x) for x in ax.get_xticks()]
            xtls    = []
            for inx,xt in enumerate(xts):
                xtl = xt.strftime('%H%M')
                if inx == 0:
                    xtl = '{}\n{}'.format(xtl,xt.strftime('%d %b %Y'))
                xtls.append(xtl)
            ax.set_xticklabels(xtls)
            
            xtl = ax.get_xticklabels()[0]
            xtl.set_ha('left')
        
        ax.legend(loc='lower left',ncol=1)
        letter  = alphabet[plot_nr]
        ts_ax.text(let_xpos,let_ypos,letter,transform=ts_ax.transAxes,**letter_dct)
        plot_nr += 1

        title = '{} {} RBN Receiver Most Spotted Stations'.format(bandObj.band_dict[band]['freq_name'],call_0)
#        title = 'RBN Receiver Most Spotted Stations'
        fontdict = {'size':24,'weight':'bold'}
        xpos    = 0.5
        ypos    = this_bottom + height + 0.010
        fig.text(xpos,ypos,title,fontdict=fontdict,ha='center')

    fpath   = os.path.join('plots','sami3_seqp_sea.png')
    fig.savefig(fpath,bbox_inches='tight')
if __name__ == '__main__':
    use_cache   = True

    csv_file    = 'df_pwr.csv.bz2'
    if not use_cache:
        df      = load_traces()
        df_pwr  = compute_ray_density(df)
        df_pwr.to_csv(csv_file,index=False,compression='bz2')
    else:
        df_pwr  = pd.read_csv(csv_file,parse_dates=['datetime'])

#    print('Plotting histograms...')
#    plot_power_histograms(df_pwr)

    plot_scatterplots(df_pwr)

    plot_sea(df_pwr)
    import ipdb; ipdb.set_trace()
