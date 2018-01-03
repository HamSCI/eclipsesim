#!/usr/bin/env python3
import os
import glob
import datetime
from collections import OrderedDict
import seqp

eclipse_dir     = 'plots/eclipse'
base_dir        = 'plots/base'
combined_dir    = 'plots/combined'

seqp.gen_lib.prep_output({0:combined_dir},clear=True)

base_pngs       = glob.glob(os.path.join(base_dir,'*.png'))
eclipse_pngs    = glob.glob(os.path.join(eclipse_dir,'*.png'))

all_dct     = OrderedDict()
for fpath in base_pngs + eclipse_pngs:
    date_str    = os.path.split(fpath)[-1][:19]
    this_date   = datetime.datetime.strptime(date_str,'%Y-%m-%d %H:%M:%S')

    if this_date not in all_dct:
        all_dct[this_date] = {}

    if 'base' in fpath:
        all_dct[this_date]['base']       = fpath
    else:
        all_dct[this_date]['eclipse']    = fpath

for this_date,path_dct in all_dct.items():
    base_fpath      = path_dct.get('base')
    eclipse_fpath   = path_dct.get('eclipse')

    fname   = os.path.split(base_fpath)[-1].replace('_base','')
    combined_fpath  = os.path.join(combined_dir,fname)

    cmd = 'convert "{}" "{}" -append "{}"'.format(base_fpath,eclipse_fpath,combined_fpath)
    print(cmd)
    os.system(cmd)
