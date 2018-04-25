#!/usr/bin/env python3
import os,glob
import datetime

from collections import OrderedDict
from collections import defaultdict
nested_dict = lambda: defaultdict(nested_dict)

import bz2

import cartopy.crs as ccrs

import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

import scipy.io

import seqp
prep_dir = seqp.gen_lib.prep_output

out_dir = 'output'
prep_dir({0:out_dir},clear=True)

xlim    = (-130, -60)
ylim    = (  20,  55)

def load_raw_grid():
    nx,ny,nz    = (100,360,100)
    lats    = np.ndarray((nx,ny,nz))*np.nan
    lons    = lats.copy()
    heights = lats.copy()

    grid_fl = 'sami3_raw/sami3_grid.dat.bz2'

    with bz2.BZ2File(grid_fl,'r') as fl:
        result = fl.readlines()

#    (17x100)x360 + 360 = 612360

    for pinx,arr in enumerate([lats,lons]):
        for yi in range(360):
            for zi in range(100):
                grp_start = (17*zi+1)+(yi*1700)+yi+pinx*612359-1
                grp = result[grp_start:grp_start+17]

                vals    = []
                for ln in grp:
                    vals += ln.split()
                vals            = [float(x) for x in vals]
                arr[:,yi,zi]    = vals
                
    # Adjust lons
    tf  = lons > 180.
    lons[tf] = lons[tf] - 360.

    # Heights
    grp_start   = 1224718
    grp         = result[grp_start:grp_start+17]
    vals        = []
    for ln in grp:
        vals += ln.split()
    vals        = [float(x) for x in vals]
    for xx in range(100):
        for yy  in range(360):
            heights[xx,yy,:]    = vals

    grid    = {}
    grid['heights'] = heights
    grid['lats']    = lats
    grid['lons']    = lons
    return grid

def load_ml_grid():
    ml_fl_grid  = 'sami3/grid.mat'
    ml_grid     = scipy.io.loadmat(ml_fl_grid)

    grid            = {}
    grid['heights'] = ml_grid['grid_heights']
    grid['lats']    = ml_grid['grid_lats']
    grid['lons']    = ml_grid['grid_lons']
    return grid

def get_closest(lat,lon,grid,hgt_inx=0):
    lats    = grid['lats'][:,:,hgt_inx]
    lons    = grid['lons'][:,:,hgt_inx]

    lat_0s  = lats*0 + lat
    lon_0s  = lons*0 + lon

    dists   = seqp.geopack.greatCircleDist(lat_0s,lon_0s,lats,lons)
    argmin  = np.argmin(dists)
    inxs    = np.unravel_index(argmin,dists.shape)
    inxs    = (inxs[0], inxs[1], hgt_inx)
    return inxs

def plot_grid(grids,ref_point=None):
    nx  = len(grids)
    ny  = 3

    inx = list(grids.keys())[0]
    ref_heights = grids[inx]['heights'][0,0,:]
    ref_heights = ref_heights[:1]

    marker_sz = 100
    for alt_inx,alt in enumerate(ref_heights):
        fig     = plt.figure(figsize=(15,10))
        axs     = []
        for xinx,(src,grid) in enumerate(grids.items()):
            lats    = grid['lats']
            lons    = grid['lons']
            heights = grid['heights']

            out_dir_0   = os.path.join(out_dir,'raw')

            ax_inx      = 1 + xinx
            ax          = fig.add_subplot(ny,nx,ax_inx,projection=ccrs.PlateCarree())
            xx          = lons[:,:,alt_inx]
            yy          = lats[:,:,alt_inx]
            vmin,vmax   = xlim
            pcoll       = ax.scatter(xx,yy,c=xx,vmin=vmin,vmax=vmax,s=marker_sz,edgecolor='face',marker='s')
            cbar        = fig.colorbar(pcoll,label='Longitude')
            axs.append(ax)

            ax_inx      = 3 + xinx
            ax          = fig.add_subplot(ny,nx,ax_inx,projection=ccrs.PlateCarree())
            xx          = lons[:,:,alt_inx]
            yy          = lats[:,:,alt_inx]
            vmin,vmax   = ylim
            pcoll       = ax.scatter(xx,yy,c=yy,vmin=vmin,vmax=vmax,s=marker_sz,edgecolor='face',marker='s')
            cbar        = fig.colorbar(pcoll,label='Latitude')
            axs.append(ax)

            ax_inx      = 5 + xinx
            ax          = fig.add_subplot(ny,nx,ax_inx,projection=ccrs.PlateCarree())
            xx          = lons[:,:,alt_inx]
            yy          = lats[:,:,alt_inx]
            zz          = heights[:,:,alt_inx]
            pcoll       = ax.scatter(xx,yy,c=zz,vmin=0,vmax=600,s=marker_sz,edgecolor='face',marker='s')
            cbar        = fig.colorbar(pcoll,label='Altitude [km]')
            axs.append(ax)

            for ax in axs:
                ax.coastlines()
                ax.set_title(src)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                if ref_point is not None:
                    xi  = ref_point[0]
                    yi  = ref_point[1]
                    zi  = alt_inx

                    xx  = lons[xi,yi,zi]
                    yy  = lats[xi,yi,zi]

                    ax.scatter(xx,yy,marker='*',s=500,color='r',zorder=100)


        txt     = '{:.1f} km Altitude'.format(alt)
        fig.text(0.5,1,txt,fontdict={'weight':'bold','size':'x-large'},ha='center')
        fig.tight_layout()

        prep_dir({0:out_dir_0},clear=False)
        fname   = "{:03d}km_alt.png".format(int(alt))
        fpath   = os.path.join(out_dir_0,fname)
        fig.savefig(fpath,bbox_inches='tight')
        plt.close(fig)
def get_vminmax(key,diff):
    prmd    = {}
    pd = nested_dict()
    pd['lats'][0]['pm'] = 2.5
    pd['lats'][1]['pm'] = 2.5
    pd['lons'][0]['pm'] = 2.5
    pd['lons'][1]['pm'] = 2.5
    pd['hgts'][0]['pm'] = 2.5
    pd['hgts'][1]['pm'] = 2.5

    pm = pd[key][diff]['pm']
    if pm is None:
        vmin,vmax   = (None,None)
    else:
        vmin,vmax   = (-pm,pm)

    return (vmin,vmax)

def plot_grid_diff(grids,src='raw'):
    nx  = len(grids)
    ny  = 3
    
    cmap        = 'Spectral'
    
    grid        = grids[src]
    ref_heights = grids[src]['heights'][0,0,:]

    marker_sz = 100
    for alt_inx,alt in enumerate(ref_heights):
        fig     = plt.figure(figsize=(15,10))
        for xinx,diff in enumerate([0,1]):
            lats    = grid['lats']
            lons    = grid['lons']
            heights = grid['heights']

            xx_0    = lons[:,:,alt_inx]
            yy_0    = lats[:,:,alt_inx]

            vmin,vmax   = get_vminmax('lons',diff)
            ax_inx  = 1 + xinx
            ax      = fig.add_subplot(ny,nx,ax_inx,projection=ccrs.PlateCarree())
            ax.coastlines()
            zz      = np.diff(xx_0,axis=diff)
            xx      = np.resize(xx_0,zz.shape)
            yy      = np.resize(yy_0,zz.shape)
            pcoll   = ax.scatter(xx,yy,c=zz,vmin=vmin,vmax=vmax,s=marker_sz,edgecolor='face',marker='s',cmap=cmap)
            cbar    = fig.colorbar(pcoll,label='Longitude')
            ax.set_title('{!s} - Diff {!s}'.format(src,diff))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            vmin,vmax   = get_vminmax('lats',diff)
            ax_inx  = 3 + xinx
            ax      = fig.add_subplot(ny,nx,ax_inx,projection=ccrs.PlateCarree())
            ax.coastlines()
            zz      = np.diff(yy_0,axis=diff)
            xx      = np.resize(xx_0,zz.shape)
            yy      = np.resize(yy_0,zz.shape)
            pcoll   = ax.scatter(xx,yy,c=zz,vmin=vmin,vmax=vmax,s=marker_sz,edgecolor='face',marker='s',cmap=cmap)
            cbar    = fig.colorbar(pcoll,label='Latitude')
            ax.set_title('{!s} - Diff {!s}'.format(src,diff))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            vmin,vmax   = get_vminmax('hgts',diff)
            ax_inx  = 5 + xinx
            ax      = fig.add_subplot(ny,nx,ax_inx,projection=ccrs.PlateCarree())
            ax.coastlines()
            zz      = np.diff(heights[:,:,alt_inx],axis=diff)
            xx      = np.resize(xx_0,zz.shape)
            yy      = np.resize(yy_0,zz.shape)
            pcoll   = ax.scatter(xx,yy,c=zz,vmin=vmin,vmax=vmax,s=marker_sz,edgecolor='face',marker='s',cmap=cmap)
            cbar    = fig.colorbar(pcoll,label='Altitude [km]')
            ax.set_title('{!s} - Diff {!s}'.format(src,diff))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        txt     = '{:.1f} km Altitude'.format(alt)
        fig.text(0.5,1,txt,fontdict={'weight':'bold','size':'x-large'},ha='center')
        fig.tight_layout()

        out_dir_0   = os.path.join(out_dir,'diff')
        prep_dir({0:out_dir_0},clear=False)
        fname   = "{:03d}km_alt.png".format(int(alt))
        fpath   = os.path.join(out_dir_0,fname)
        fig.savefig(fpath,bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    ml_grid         = load_ml_grid()
    raw_grid        = load_raw_grid()

    rg              = ml_grid   # Reference Grid
    lat,lon         = (40,-100)
    inxs            = get_closest(lat,lon,rg)

    grids           = OrderedDict()
    grids['ml']     = ml_grid
    grids['raw']    = raw_grid

    plot_grid(grids,ref_point=inxs)
#    plot_grid_diff(grids)

    xi      = inxs[0]
    yi      = inxs[1]
    zi      = inxs[2]
    hgts    = rg['heights'][xi,yi,:]
    lats    = rg['lats'][:,yi,zi]
    lons    = rg['lons'][xi,:,zi]

#    lat_tf  = np.logical_and(lats >= ylim[0], lats < ylim[1])
#    lon_tf  = np.logical_and(lons >= xlim[0], lons < xlim[1])
#    tf      = np.logical_and(lat_tf,lon_tf)

    import ipdb; ipdb.set_trace()
