#!/usr/bin/env python3
import os,glob
import datetime
from collections import OrderedDict

import bz2

import cartopy.crs as ccrs

import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import scipy.io

import seqp
prep_dir = seqp.gen_lib.prep_output

out_dir = 'output'
prep_dir({0:out_dir},clear=True)

def load_raw_grid():
    nx,ny,nz    = (100,360,100)
    lats    = np.ndarray((nx,ny,nz))*np.nan
    lons    = lats.copy()
    heights = lats.copy()

    grid_fl = 'sami3_raw/sami3_grid.dat.bz2'

    with bz2.BZ2File(grid_fl,'r') as fl:
        result = fl.readlines()

    import ipdb; ipdb.set_trace()

def load_ml_grid():
    ml_fl_grid  = 'sami3/grid.mat'
    ml_grid     = scipy.io.loadmat(ml_fl_grid)

    grid            = {}
    grid['heights'] = ml_grid['grid_heights']
    grid['lats']    = ml_grid['grid_lats']
    grid['lons']    = ml_grid['grid_lons']
    return grid

def plot_grid(grids):
    nx  = len(grids)
    ny  = 3

    import ipdb; ipdb.set_trace()
    marker_sz = 100
    for alt_inx,alt in enumerate(heights[0,0,:]):
        for xinx,(src,grid) in enumerate(grids.items()):
            lats    = grid['lats']
            lons    = grid['lons']
            heights = grid['heights']

            ax_inx  = 0 
            fig     = plt.figure(figsize=(10,12))

            ax_inx  += 1
            ax      = fig.add_subplot(ny,nx,ax_inx,projection=ccrs.PlateCarree())
            ax.coastlines()
            xx      = lons[:,:,alt_inx]
            yy      = lats[:,:,alt_inx]
            pcoll   = ax.scatter(xx,yy,c=xx,vmin=-180,vmax=180,s=marker_sz,edgecolor='face',marker='s')
            cbar    = fig.colorbar(pcoll,label='Longitude')

            ax_inx  += 1
            ax      = fig.add_subplot(ny,nx,ax_inx,projection=ccrs.PlateCarree())
            ax.coastlines()
            xx      = lons[:,:,alt_inx]
            yy      = lats[:,:,alt_inx]
            pcoll   = ax.scatter(xx,yy,c=yy,vmin=-90,vmax=90,s=marker_sz,edgecolor='face',marker='s')
            cbar    = fig.colorbar(pcoll,label='Latitude')

            ax_inx  += 1
            ax      = fig.add_subplot(ny,nx,ax_inx,projection=ccrs.PlateCarree())
            ax.coastlines()
            xx      = lons[:,:,alt_inx]
            yy      = lats[:,:,alt_inx]
            zz      = heights[:,:,alt_inx]
            pcoll   = ax.scatter(xx,yy,c=zz,vmin=0,vmax=600,s=marker_sz,edgecolor='face',marker='s')
            cbar    = fig.colorbar(pcoll,label='Altitude [km]')

            txt     = '{:.1f} km Altitude'.format(alt)
            fig.text(0.5,1,txt,fontdict={'weight':'bold','size':'x-large'})
            fig.tight_layout()
            fname   = "{:03d}km_alt.png".format(int(alt))
            fpath   = os.path.join(out_dir,fname)
            fig.savefig(fpath,bbox_inches='tight')

            plt.close(fig)

#ml_fls_data = glob.glob(os.path.join('sami3','data_*.mat'))
#lat =   40.
#lon = -100.
#lat_inx     = np.argmin(np.abs(ml_lats[:,0,0]-lat))
#lon_inx     = np.argmin(np.abs(ml_lons[:,0,0]-lon))
#
## 0000: 1600 UT, 0159: 2357 UT (21 August 2017)
#keys = [x for x in range(160)]

if __name__ == '__main__':
    raw_grid    = load_raw_grid()
    ml_grid     = load_ml_grid()

    grids       = OrderedDict()
    grids['ml'] = ml_grid

    plot_grid(grids)
    import ipdb; ipdb.set_trace()
