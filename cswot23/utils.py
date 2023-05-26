import numpy as np
import pandas as pd

from cswot_adcp import data_loader as loader

import pynsitu as pin
crs = pin.maps.crs

def load_adcp_sta(sta_files, start, end, fix_time=False):

    D = []
    for file in sta_files:
        ds = loader.read_data(file)
        # not sure this is wise but should help with selection
        ds = (ds
              .assign_coords(z=-ds["range"])
              .swap_dims(dict(range="z"))
        )
        ds["time_gps"] = ("time", ds["time_gps"].values)
        ds.attrs["sta_file"] = file
        D.append(ds)

    # year is 2020 in tethys time variable ! offset by 2 years
    # this may be coming from cswot_adcp loader ...
    if fix_time:
        for ds in D:
            _fix_time(ds)

    Dt = []
    for ds in D:
        dst = ds.sel(time=slice(start, end))
        # time_gps=slice(start, end)
        if dst.time.size>0:
            Dt.append(dst)
    
    return Dt

def _fix_time(ds):
    ds["time"] = ("time", 
                  np.array([(pd.Timestamp(t.values)+pd.DateOffset(years=3)).to_numpy() 
                            for t in ds.time])
                 )
    
def get_z_minmax(D):
    return min([float(ds.z.min()) for ds in D]), max([float(ds.z.max()) for ds in D])

def add_adcp(ax, adcp, 
             di, 
             depth, op="mean", 
             tstart=None, tend=None,
             reference=.5, keyposition=(0.8, 0.9), 
             **kwargs,
            ):
    """ plot adcp velocities """

    dkwargs = dict(color="k", pivot="tail", scale=5, width=2e-3, add_guide=False)
    dkwargs.update(**kwargs)
    
    x, y = "elongitude_gps", "elatitude_gps"
    u, v = "compensated_E", "compensated_N"
    q = None
    for a in adcp:
        _a = a.sel(time=slice(tstart, tend))
        if _a.time.size>0:
            _a = _a.isel(time=slice(0,None,di))
            if op=="mean":
                _a = _a.sel(z=slice(0, -depth)).mean("z")
            elif op=="sel":
                _a = _a.sel(z=-depth, method="nearest")
            q = _a.plot.quiver(x, y, u, v, transform=crs, **dkwargs)

    #add quiver key
    if reference and q is not None:
        qk = ax.quiverkey(q, *keyposition, reference, f'{reference} m/s', 
                     transform=crs, color=dkwargs["color"], labelcolor="k",
                     labelpos='N', coordinates='axes')
        qk.set_zorder(10)
        