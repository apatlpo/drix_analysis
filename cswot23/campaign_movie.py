import os
from glob import glob

import numpy as np
import pandas as pd

#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib
from contextlib import contextmanager

import pynsitu as pin
crs = pin.maps.crs

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# load campaign object
campaign = "cswot23"

kwargs_map = dict(coastline="h")
# coastline = ["c", "l", "i", "h", "f"]

# fig_dir = os.path.join(cp.pathp, 'figs/')
#fig_dir = "/Users/aponte/tmp/figs/"
fig_dir = "/home1/scratch/aponte/figs"



def load_data():

    cp = pin.Campaign(f"/home1/datahome/aponte/cswot_analysis/{campaign}.yaml")

    # tethys
    te = cp.load("tethys_underway_1T.nc").set_coords(["lon", "lat"])

    # atalante
    at = cp.load("atalante_underway_1T.nc").set_coords(["lon", "lat"])

    # tethys
    adcp_dir = os.path.join(cp["path_raw"], cp["tethys"]["raw_dir"], "VMADCP_75")
    sta_files = sorted(glob(os.path.join(adcp_dir, "*.STA")))
    start = cp["tethys"]["leg1"].start.time
    end = cp["tethys"]["leg4"].end.time

    adcp_te = load_adcp_sta(sta_files, start, end, fix_time=True)
    print("tethys adcp min/max:", get_z_minmax(adcp_te))
    # 8 meter bin sizes: [ -16.47,  -24.47,  -32.47, ...

    # atalante
    start = cp["atalante"]["leg0"].start.time
    end = cp["atalante"]["leg3"].end.time

    # OS38
    if False:
        adcp_dirs = sorted(glob(os.path.join(cp["path_raw"], cp["atalante"]["raw_dir"], "OS38/DONNEES/*")))
        sta_files = []
        for d in adcp_dirs:
            sta_files = sta_files + sorted(glob(os.path.join(d, "*.STA")))
        adcp_at_38 = load_adcp_sta(sta_files, start, end, fix_time=True)
        print("atalante 38 adcp min/max:", get_z_minmax(adcp_at_38))
        
    # OS150
    if True:
        adcp_dirs = sorted(glob(os.path.join(cp["path_raw"], cp["atalante"]["raw_dir"], "OS150/DONNEES/*")))
        sta_files = []
        for d in adcp_dirs:
            sta_files = sta_files + sorted(glob(os.path.join(d, "*.STA")))
        adcp_at_150 = load_adcp_sta(sta_files, start, end, fix_time=True)
        print("atalante 150 adcp min/max:", get_z_minmax(adcp_at_150))
    # 6 meter bin sizes: [ -14.26,  -20.26,  -26.26, ...    


    # various shortcuts
    col_te = cp["tethys"]["color"]
    col_at = cp["atalante"]["color"]

from cswot_adcp import data_loader as loader
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


# -------------------------------- plotting ------------------------------------

@contextmanager
def mpl_backend(backend):
    original = matplotlib.get_backend()
    yield matplotlib.use(backend)
    matplotlib.use(original)
    
# ref: 
#   - https://stackoverflow.com/questions/73322175/temporarily-set-matplotlib-backend-for-one-cell-reverting-afterwards
#   - https://www.jujens.eu/posts/en/2022/Sep/25/context-manager-decorators/#context-manager-decorators

@mpl_backend("agg")
def make_figures(
    time_range,
    istart=0,
    dt_trail="1H",  # trail behing ships
    geobuffer=0.1, # geographical buffer
    #tstart,
    #tend,
    #drifters=None,
    #ship=False,
    adcp=True,
    #wind=None,
    #ctd=None,
    #wind_arrow_scale=1,
    #wind_di=4,
    #wind_offset=0,
    overwrite=False,
):
    """Make a movie"""

    cp = pin.Campaign(f"/home1/datahome/aponte/cswot_analysis/{campaign}.yaml")
    #from tqdm import tqdm

    #t_range = pd.date_range(tstart, tend, freq=dt)
    dt_trail = pd.Timedelta(dt_trail)

    _kwargs_map = dict(figsize=(7,7))
    _kwargs_map.update(**kwargs_map)

    i = istart
    #for t in tqdm(time_range):
    for t in time_range:
        
        figname = os.path.join(fig_dir, "fig_t%05d" % (i) + ".png")
        if not overwrite and os.path.isfile(figname):
            continue

        # find geographical extent
        extent = find_extent(geobuffer,
                             te.sel(time=t, method="nearest"),
                             at.sel(time=t, method="nearest"),
                            )
        projection, _ = pin.maps.get_projection(extent)
        extent = adjust_extent_aspect_ratio(_kwargs_map["figsize"], extent, projection)

        #fig, ax, _ = pin.maps.plot_map(**_kwargs_map, bathy_levels=cp["bathy"]["levels"])
        fig, ax, _ = cp.map(extent=extent, **_kwargs_map)

        # ship
        #if ship:
        
        # atalante
        _ds = at.sel(time=slice(t - dt_trail, t))
        ax.plot(
            _ds["lon"], _ds["lat"], lw=3, color=col_at, alpha=1, transform=crs
        )
        ax.scatter(
            _ds["lon"][-1],
            _ds["lat"][-1],
            s=10,
            c="0.7",
            marker="o",
            edgecolors="k",
            linewidths=0.5,
            transform=crs,
            zorder=10,
        )
        # tethys
        _ds = te.sel(time=slice(t-dt_trail,t))
        ax.plot(
            _ds["lon"], _ds["lat"], lw=3, color=col_te, alpha=1, transform=crs
        )
        if _ds.time.size > 0:
            ax.scatter(
                _ds["lon"][-1],
                _ds["lat"][-1],
                s=10,
                c="0.7",
                marker="o",
                edgecolors="k",
                linewidths=0.5,
                transform=crs,
                zorder=10,
            )
            
        # adcp
        if adcp:
            add_adcp(ax, adcp_te, t, 1, 30, tstart=tstart)
            add_adcp(ax, adcp_at_150, t, 1, 30, tstart=tstart)


        # add pool of isolated events

        # cp.add_legend(ax, loc=4, colors={idx: c for idx, c in zip(ids, colors)})
        ax.set_title(cp.name + "  " + str(t))

        _ = fig.savefig(figname, dpi=150, facecolor="w")  # bbox_inches = 'tight'
        _ = fig.clf()
        
        i += 1

def find_extent(buffer, *args):
    """ find geographical bounds based on a various datasets and a buffer size """
    lon_min = min([float(a.lon.min()) for a in args])
    lon_max = max([float(a.lon.max()) for a in args])
    lat_min = min([float(a.lat.min()) for a in args])
    lat_max = max([float(a.lat.max()) for a in args])
    lon_scale = 1/np.cos(float(args[0].lat.mean()) * pin.deg2rad)
    extent = (lon_min-buffer*lon_scale, lon_max+buffer*lon_scale, 
              lat_min-buffer, lat_max+buffer,
             )
    return extent

def adjust_extent_aspect_ratio(aspect_ratio, extent, out_crs):
    """
    Generate extent from aspect ratio, target extent, and projection
    latitude bounds are adjusted to maintain the aspect ratio

    Parameters
    ----------
    aspect_ratio: tuple:
        Aspect ratio x/y
    extent: tuple/list
        longitude and latitude bounds
    out_crs: cartopy.crs
        Out crs for extent values.

    Returns:
        tuple: (lon_min, lon_max, lat_min, lat_max) or in projected coordinates
    """
    
    # central point
    _lon_central = (extent[0] + extent[1]) * 0.5
    _lat_central = (extent[2] + extent[3]) * 0.5
    center_point = (_lon_central, _lat_central)
    # Transform map center to specified crs
    c_mercator = out_crs.transform_point(*center_point, src_crs=crs)
            
    # minimum longitude (min_lon) and maximnum longitude (max_lon):
    lon_min, lon_max = extent[0], extent[1]    
    # Transform minimum longitude and maximum longitude to specified crs (default to Mercator)
    lon_min = out_crs.transform_point(lon_min, center_point[0], src_crs=crs)[0]
    lon_max = out_crs.transform_point(lon_max, center_point[0], src_crs=crs)[0]
    
    # calculates minimum latitude (min_lat) and maximum latitude (max_lat) 
    # using center point and distance between min_lon and max_lon
    # To achieve this we will use formula [(lon_distance/lat_distance) = (aspect_ratio[0]/aspect_ratio[1])]
     # Calculate distance between min_lon and max_lon
    lon_distance = (lon_max) - (lon_min)
    
    # To calculate lat_distance, we will proceed accordingly
    lat_distance = lon_distance * aspect_ratio[1] / aspect_ratio[0]
    
    # Now calculate max_lat and min_lan by adding/subtracting half of the distance from center latitude
    lat_max = c_mercator[1] + lat_distance/2
    lat_min = c_mercator[1] - lat_distance/2
    
    # We can return our result in any format (eg. in Mercator coordinates or in degrees)
    if out_crs != crs:
        lon_min, lat_min = crs.transform_point(lon_min, lat_min, src_crs=out_crs)
        lon_max, lat_max = crs.transform_point(lon_max, lat_max, src_crs=out_crs)
    
    if lat_min<extent[2] and lat_max>extent[3]:
        return lon_min, lon_max, lat_min, lat_max
    else:
        # recursively call method to make sure initial bounds are satisfied
        dl = (extent[1] - extent[0])/10 # may not be robust to date change
        print(f"increase input exent with dl={dl}")
        extent[0] = extent[0]-dl
        extent[1] = extent[1]+dl
        return adjust_extent_aspect_ratio(aspect_ratio, extent, out_crs)

def add_adcp(ax, adcp, t, di, depth, reference=.5, tstart=None):
    """ plot adcp velocities """

    x, y = "elongitude_gps", "elatitude_gps"
    u, v = "compensated_E", "compensated_N"
    q = None
    for a in adcp:
        _a = a.sel(time=slice(tstart, t))
        if _a.time.size>0:
            _a = _a.isel(time=slice(0,None,di))
            _a = _a.sel(z=slice(0, -depth)).mean("z")
            q = _a.plot.quiver(x, y, u, v, transform=crs, pivot="tail", 
                               scale=5, width=2e-3, add_guide=False)

    #add quiver key
    if reference and q is not None:
        ax.quiverkey(q, 0.1, 0.1, reference, f'{reference} m/s', 
                     transform=crs, color="k", labelcolor="k",
                     labelpos='N', coordinates='axes')    

if __name__=="__main__":

    cp = pin.Campaign(f"/home1/datahome/aponte/cswot_analysis/{campaign}.yaml")

    part = "dev"
    #part = "all"
    dt = "5T"

    if part == "all":
        tstart = cp["start"]
        # tstart = '2022-05-11 05:00:00'
        tend = cp["end"]
    elif part == "drifters":
        tstart = "2022-09-23 05:30:00"
        tend = "2022-09-23 16:10:00"
        dt = "30s"
    elif part == "dev":
        # dev
        tstart = "2023-03-28 10:00:00"
        tend = "2023-03-28 11:00:00"

    kwargs = dict(
        #drifters=dr,
        #ship=True,
        #wind=arome,
        #wind=arome,
        #bounds=bounds,
        #istart=istart,
        #dt=dt,
        dt_trail="1H",
        overwrite=True,
    )
    #kwargs.update(wind_arrow_scale=3, wind_di=2, wind_offset=0.02)
    
    t_range = pd.date_range(tstart, tend, freq=dt)

    serial = True

    if serial:
        make_figures(t_range, **kwargs)
    else:
        # distributes
        # distributed figure production
        from dask.distributed import Client
        from dask import delayed
        client = Client()

        delayed_make_figures = delayed(make_figures)

        # divide timeline into chunks
        Nb = len(client.nthreads())
        rg = range(0, t_range.size)
        i_rg = np.array_split(rg, Nb)

        # distribute figure production
        values = [delayed_make_figures(t_range[i], i_start=i[0], **kwargs) for i in i_rg]
        futures = client.compute(values)
        results = client.gather(futures)

    print(" ! done !")







