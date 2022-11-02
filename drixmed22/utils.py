import os
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import minimize, Bounds, shgo

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from datetime import datetime
import time

import pyproj

import urllib.request
import ipywidgets as widgets
from IPython.display import display, clear_output

from pynsitu.geo import projection

rad2deg = 180.0 / np.pi
knot = 0.514
nmile = 1852  # meters

one_second = pd.Timedelta("1s")
now = lambda: pd.to_datetime(datetime.utcnow())

# ----------------------------- reference locations ---------------------------------

_stations = {
    "PS0": dict(lon=-5.075, lat=42.8266),
    "PN0": dict(lon=4.975, lat=43.09),
    "PS2": dict(lon=5.2783, lat=43.8866),
    "Julio": dict(lon=5.235, lat=43.142),
    "MesuRho": dict(lon=4.8666, lat=43.32),
}
stations = pd.DataFrame(_stations).T

# ----------------------------- reference locations ---------------------------------

bathy = (
    "/Users/aponte/Data/bathy/med_gebco0/gebco_2022_n43.932_s41.3124_w2.0025_e7.6744.nc"
)

# ----------------------------- project, metrics ---------------------------------

lon_ref, lat_ref = 5, 43
proj = projection(lon_ref, lat_ref)

lonlat2xy = proj.lonlat2xy
xy2lonlat = proj.xy2lonlat


def metrics_cheatsheet(lon, lat):
    """print useful info for navigation"""

    proj = pyproj.Proj(proj="aeqd", lat_0=lat, lon_0=lon, datum="WGS84", units="m")

    def dl(dlon, dlat):
        x0, y0 = proj.transform(lon, lat)
        x1, y1 = proj.transform(lon + dlon, lat + dlat)
        return np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    print(
        f" lon: 1 deg = {dl(1,0)/1e3:.2f}km,  0.1 deg = {dl(.1,0)/1e3:.1f}km"
        + f",  0.01 deg = {dl(.01,0):.1f}m,  0.001 deg = {dl(.001,0):.1f}m"
    )
    print(
        f" lon: 1 deg = {dl(1,0)/1e3:.2f}km,  1 min = {dl(1/60,0):.1f}m"
        + f",  .1 min = {dl(.1/60,0):.1f}m,  0.01 min = {dl(.01/60,0):.1f}m"
    )
    print(
        f" lon: 1 deg = {dl(1,0)/1e3:.2f}km,  1 sec = {dl(1/3600,0):.1f}m"
        + f",  .1 sec = {dl(.1/3600,0):.1f}m"
    )
    print("-----------------------------------------------------------------")
    print(
        f" lat: 1 deg = {dl(0,1)/1e3:.2f}km,  0.1 deg = {dl(0,.1)/1e3:.1f}km"
        + f",  0.01 deg = {dl(0,.01):.1f}m,  0.001 deg = {dl(0,.001):.1f}m"
    )
    print(
        f" lat: 1 deg = {dl(0,1)/1e3:.2f}km,  1 min = {dl(0,1/60):.1f}m"
        + f",  .1 min = {dl(0,.1/60):.1f}m,  0.01 min = {dl(0,.01/60):.1f}m"
    )
    print(
        f" lat: 1 deg = {dl(0,1)/1e3:.2f}km,  1 sec = {dl(0,1/3600):.1f}m"
        + f",  .1 sec = {dl(0,.1/3600):.1f}m"
    )


def deg_mindec(v):
    deg = np.trunc(v)
    mind = abs(v - deg) * 60
    return deg, mind


def to_deg_mindec(df, lon, lat):
    """add lon/lat in deg/min decimals to a dataframe"""
    df[lon + "_deg"], df[lon + "_min"] = deg_mindec(df[lon])
    df[lat + "_deg"], df[lat + "_min"] = deg_mindec(df[lat])
    return df


# ---------------------------- route scheduling ----------------------------------


def _match_displacements(x, dX, Uo, speed, t_wait):
    theta, dt = x
    Us = speed * np.exp(1j * theta)
    return np.abs(dX + Uo * (dt + t_wait) - Us * dt)


def solve_route_heading(dX, speed, Uo, time_waiting, time_max=1, **kwargs):
    """Solve where we need to head given:

    Parameters
    ----------
    dX: complex
        distance relative to initial point
    speed: float
        ship speed (m/s)
    Ui: complex
        ocean current (m/s)
    time_waiting: float
        time we wish to wait at the deployment point (seconds)
    dt_max
    **kwargs: sent to optimizing method

    Returns
    -------
    theta: float
        Heading in radians
    dt_route: float
        Time to launch location
    dt_total: float
        Time to launch location + time_waiting
    X_launch: complex
        Launch position
    Us: complex
        Complex ship velocity (i.e. include calculated heading)
    """

    # local optimization
    # this turn out not to work well, don't know why
    # bounds = Bounds([-2*np.pi, 2*np.pi], [0., np.inf])
    # https://docs.scipy.org/doc/scipy/reference/optimize.html#local-multivariate-optimization
    # minopts = dict(method='trust-constr', bounds=bounds)
    # minopts = dict(bounds=bounds)
    # minopts.update(**kwargs)
    # res = minimize(_match, [np.angle(dX), time_waiting], args=(dX, Uo, speed, time_waiting), **minopts)

    # global optimization
    bounds = [(-1.1 * np.pi, np.pi), (1.0, 3600 * time_max)]
    minopts = dict(sampling_method="sobol")  # options=dict(f_tol=1),
    minopts.update(**kwargs)
    res = shgo(_match_displacements, bounds, (dX, Uo, speed, time_waiting), **minopts)

    theta, dt_route = res["x"]
    Us = speed * np.exp(1j * theta)
    X_launch = Us * dt_route

    return theta, dt_route, dt_route + time_waiting, X_launch, Us


def solve_route_heading_xr(label, *args):
    """wrapper around solve_route_heading that spits a dataset instead"""
    theta, dt_route, dt_total, X_launch, _ = solve_route_heading(*args)
    ds = xr.Dataset(
        dict(
            theta=(("route",), [theta * rad2deg]),
            dt_route=(("route",), [pd.Timedelta(seconds=dt_route)]),
            dt_total=(("route",), [pd.Timedelta(seconds=dt_total)]),
            X_launch=(("route",), [X_launch]),
        )
    )
    return ds


def plot_route_solution(dX, Us, Uo, dt, time_waiting):
    """plot route solution"""

    fig, ax = plt.subplots(1, 1)

    opt = {
        "head_width": 50,
        "head_length": 50,
        "width": 20,
        "length_includes_head": False,
    }

    ax.arrow(0, 0, dX.real, dX.imag, fc="k", **opt)
    ax.arrow(0, 0, (dt) * Uo.real, (dt) * Uo.imag, fc="b", **opt)
    ax.arrow(
        0,
        0,
        (dt + time_waiting) * Uo.real,
        (dt + time_waiting) * Uo.imag,
        fc="b",
        **opt,
    )
    ax.arrow(0, 0, dt * Us.real, dt * Us.imag, fc="orange", **opt)

    ax.set_aspect("equal")
    ax.grid()


def deployments_route_schedule(
    lon_ship,
    lat_ship,
    ship_speed,
    lon_vertices,
    lat_vertices,
    time_waiting,
    Uo,
    lon_a=None,
    lat_a=None,
    time_a=None,
    skip=0,
):
    """Compute and schedule a route for drifter deployments
    2 use cases:
    - initial deployment
    - underway deployment (lon_a, lat_a, time_a not None and skip>0)

    Parameters
    ----------
    lon_ship, lat_ship: float
        ship position
    ship_speed: float
        ship cruise speed [m/s]
    lon_vertices, lat_vertices: np.array
        vertex positions
    time_waiting: float
        time we wish to wait at each deployment locations
    Uo: complex
        ocean current in m/s
    lon_a, lat_a, time_a: float, optional
        anchor point (first vertex) position and corresponding time
        If lon_a and lat_a are provided but not time_a, this is the first
        deployment.
        Otherwise, lon_a, lat_a corresponds to a passed position fix and this
        implies skip>0
    skip: int, optional
        Number of vertex (deployments) to skip.
        Default is 0 i.e. first deployment
    """

    # positions (initial, anchor, vertices)
    x, y = lonlat2xy(lon_ship, lat_ship)
    #
    if lon_a is None:
        lon_a = lon_ship
    if lat_a is None:
        lat_a = lat_ship
    x_a, y_a = lonlat2xy(lon_a, lat_a)
    #
    x_v, y_v = lonlat2xy(lon_vertices, lat_vertices)
    X = x_v + 1j * y_v

    # solve for one segment
    def _route_core(dx, Uo, X_deployment, time_deployment):
        theta_route, dt_route, dt_total, dX_deployment, Us = solve_route_heading(
            dx, ship_speed, Uo, time_waiting
        )
        # update
        X_deployment = X_deployment + dX_deployment
        time_deployment = time_deployment + dt_total * one_second
        # store
        x_deployment, y_deployment = X_deployment.real, X_deployment.imag
        lon_deployment, lat_deployment = xy2lonlat(x_deployment, y_deployment)
        se = pd.Series(
            dict(
                time=time_deployment,
                # x=x_deployment, y=y_deployment,
                lon=lon_deployment,
                lat=lat_deployment,
                heading=theta_route,
                dt_route=dt_route / 60,
                dt_total=dt_total / 60,
            )
        )
        return se, X_deployment, time_deployment

    # "first" deployment is treated separately
    time_initial = now()  # time where last position was recorded, to be updated
    # skip points if need be
    if skip == 0:
        # deployment at anchor position
        _dX = x_a - x + 1j * (y_a - y)
        Uo_init = 0.0 * 1j
    else:
        X = X[skip:]
        _dX = X[0] - (x + 1j * y)  # from current position
        assert time_a is not None, "time_a (and lon_a, lat_a) are required when skip>0"
        _dX += Uo * (
            (now() - time_a) / one_second
        )  # correct anchor position to account for ocean flow displacement
        Uo_init = Uo
    # store initial position
    se = pd.Series(
        dict(
            time=time_initial,
            # x=x_deployment, y=y_deployment,
            lon=lon_ship,
            lat=lat_ship,
            heading=np.NaN,
            dt_route=np.NaN,
            dt_total=np.NaN,
        )
    )
    S = [se]
    # compute route to first deployment
    se, X_deployment, time_deployment = _route_core(
        _dX, Uo_init, x + 1j * y, time_initial
    )
    S.append(se)

    # following deployments
    dX = np.diff(X)
    for dx in dX:
        # could also update velocity depending on time_deployment
        se, X_deployment, time_deployment = _route_core(
            dx, Uo, X_deployment, time_deployment
        )
        # print(dx, X_deployment) # debug
        S.append(se)

    # concatenate results
    df = pd.DataFrame(
        S, index=pd.Index(np.arange(skip - 1, skip + X.size), name="deployment")
    )

    # add degrees and minute with decimals
    to_deg_mindec(df, "lon", "lat")

    return df


def plot_deployments_route(lon, lat, df, arrows=True, **kwargs):
    """Plot deployment routes as compute from deployments_route_schedule"""

    dkwargs = dict(
        bathy=True, zoom=[-0.32, -0.2, 49.29, 49.36], vmax=30, figsize=(10, 10)
    )
    dkwargs.update(kwargs)
    fac = plot_bs(**dkwargs)
    ax = fac["ax"]

    ax.scatter(lon, lat, 40, color="orange", transform=ccrs.PlateCarree())
    ax.scatter(df.lon, df.lat, 40, color="red", transform=ccrs.PlateCarree())

    # arrows
    if arrows:
        transform = ccrs.PlateCarree()._as_mpl_transform(ax)
        for i in df.index[:-1]:
            ax.annotate(
                "",
                xy=(df.loc[i + 1, "lon"], df.loc[i + 1, "lat"]),
                xytext=(df.loc[i, "lon"], df.loc[i, "lat"]),
                xycoords=transform,
                size=20,
                arrowprops=dict(
                    facecolor="0.5",
                    ec="none",
                    # arrowstyle="fancy",
                    connectionstyle="arc3,rad=-0.3",
                ),
            )

    # labels
    for i in range(lon.size):
        ax.text(lon[i] + 1e-3, lat[i], f"{i}", transform=ccrs.PlateCarree())

    return fac


class dashboard_route(object):
    def __init__(self, lon, lat, **kwargs):
        #
        self.lon_vertices = lon
        self.lat_vertices = lat
        #
        self.update_num = 0

        self.out = widgets.Output()
        display(self.out)

        self.build_dashboard(**kwargs)
        self.update(None)  # time, devices

    def build_dashboard(self, **kwargs):

        # dkwargs = dict(ship_speed=5., square_radius=1., square_theta=0., square_center=2, time_waiting=1.)
        dkwargs = dict(
            ship_speed=5.0,
            time_waiting=1.0,
            current_speed_x=0.02,
            current_speed_y=0.0,
        )
        dkwargs.update(**kwargs)

        w = dict()

        w["button"] = widgets.Button(
            description="Update",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Update",
            icon="check",  # (FontAwesome names without the `fa-` prefix)
        )
        w["button"].on_click(self.update)

        w["ship_speed"] = widgets.FloatSlider(
            value=dkwargs["ship_speed"],
            min=1.0,
            max=20.0,
            step=1.0,  # 0.1 sec = 2m
            description="Ship speed [knots]:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".1f",
        )

        w["current_speed_x"] = widgets.BoundedFloatText(
            value=dkwargs["current_speed_x"],
            min=-2.0,
            max=2.0,
            step=0.02,
            description="Current speed x [m/s]:",
            disabled=False,
        )

        # widgets.FloatSlider(
        #    value=0.,
        #    min=-2.,
        #    max=2.,
        #    step=0.02, # 0.1 sec = 2m
        #    description='Current speed x [m/s]:',
        #    disabled=False,
        #    continuous_update=False,
        #    orientation='horizontal',
        #    readout=True,
        #    readout_format='.2f',
        # )

        w["current_speed_y"] = widgets.BoundedFloatText(
            value=dkwargs["current_speed_y"],
            min=-2.0,
            max=2.0,
            step=0.02,
            description="Current speed y [m/s]:",
            disabled=False,
        )

        # widgets.FloatSlider(
        #    value=0.,
        #    min=-2.,
        #    max=2.,
        #    step=0.02, # 0.1 sec = 2m
        #    description='Current speed y [m/s]:',
        #    disabled=False,
        #    continuous_update=False,
        #    orientation='horizontal',
        #    readout=True,
        #    readout_format='.2f',
        # )

        _now = datetime.utcnow()
        self.now = _now

        _lon_deg, _lon_min = deg_mindec(self.lon_vertices[0])
        _lat_deg, _lat_min = deg_mindec(self.lat_vertices[0])
        for suff in ["ship", "anchor"]:

            w[suff + "_lon_deg"] = widgets.Dropdown(
                options=[
                    f"{_lon_deg-1}E",
                    f"{_lon_deg}E",
                    f"{_lon_deg+1}E",
                ],  # to be generalized
                value=f"{_lon_deg}E",
                description=suff + " lon - deg:",
                disabled=False,
            )

            w[suff + "_lon_min"] = widgets.BoundedFloatText(
                value=round(_lon_min, ndigits=3),
                min=0.0,
                max=60.0,
                step=0.001,
                description="lon min dec:",
                disabled=False,
            )

            # widgets.Dropdown(
            #    options=np.arange(60),
            #    value=15,
            #    description=suff+' lon [min]:',
            #    disabled=False,
            # )

            # w[suff+"_lon_sec"] = widgets.FloatSlider(
            #    value=0,
            #    min=0,
            #    max=60.0,
            #    step=0.1, # 0.1 sec = 2m
            #    description=suff+' lon [sec]:',
            #    disabled=False,
            #    continuous_update=False,
            #    orientation='horizontal',
            #    readout=True,
            #    readout_format='.1f',
            # )

            w[suff + "_lat_deg"] = widgets.Dropdown(
                options=["43N"],  # to be generalized
                value="43N",
                description=suff + " lat [deg]:",
                disabled=False,
            )

            w[suff + "_lat_min"] = widgets.BoundedFloatText(
                value=round(_lat_min, ndigits=3),
                min=0.0,
                max=60.0,
                step=0.001,
                description="lat min dec:",
                disabled=False,
            )

            # w[suff+"_lat_min"] = widgets.Dropdown(
            #    options=np.arange(60),
            #    value=19,
            #    description=suff+' lat [min]:',
            #    disabled=False,
            # )

            w[suff + "_lat_sec"] = widgets.FloatSlider(
                value=0.0,
                min=0,
                max=60.0,
                step=0.1,  # 0.1 sec = 2m
                description=suff + " lat - sec:",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format=".1f",
            )

            w[suff + "_hour"] = widgets.Dropdown(
                options=np.arange(24),
                description=suff + " hour:",
                disabled=False,
                value=_now.hour,
            )

            w[suff + "_minute"] = widgets.Dropdown(
                options=np.arange(60),
                description=suff + " minute:",
                disabled=False,
                value=_now.minute,
            )

            w[suff + "_second"] = widgets.Dropdown(
                options=np.arange(60),
                description=suff + " second:",
                disabled=False,
                value=_now.second,
            )

        # w["square_radius"] = widgets.FloatSlider(
        #     value=dkwargs["square_radius"],
        #     min=.1,
        #     max=10.0,
        #     step=0.1, # 0.1 sec = 2m
        #     description='Square radius [km]:',
        #     disabled=False,
        #     continuous_update=False,
        #     orientation='horizontal',
        #     readout=True,
        #     readout_format='.1f',
        # )
        #
        # w["square_theta"] = widgets.FloatSlider(
        #     value=dkwargs["square_theta"],
        #     min=-90.,
        #     max=90.0,
        #     step=2., # 0.1 sec = 2m
        #     description='Square orientation [deg]:',
        #     disabled=False,
        #     continuous_update=False,
        #     orientation='horizontal',
        #     readout=True,
        #     readout_format='.0f',
        # )
        #
        # w["square_center"] = widgets.Dropdown(
        #     options=np.arange(5),
        #     value=dkwargs["square_center"],
        #     description='Center index',
        #     disabled=False,
        # )

        w["time_waiting"] = widgets.FloatSlider(
            value=dkwargs["time_waiting"],
            min=1.0,
            max=30.0,
            step=1.0,  # 0.1 sec = 2m
            description="Time waiting [min]:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".0f",
        )

        w["skip"] = widgets.Dropdown(
            options=np.arange(5),
            value=0,
            description="Skip index",
            disabled=False,
        )

        #
        grid = widgets.GridspecLayout(10, 3, height="400px", align_items="center")

        i = 0
        grid[0, 0] = w["button"]
        i += 1

        #
        grid[i, 0] = w["ship_speed"]
        grid[i, 1] = w["current_speed_x"]
        grid[i, 2] = w["current_speed_y"]
        i += 1

        for suff in ["ship", "anchor"]:

            grid[i, 0] = w[suff + "_lon_deg"]
            grid[i, 1] = w[suff + "_lon_min"]
            # grid[i, 2] = w[suff+"_lon_sec"]
            i += 1

            grid[i, 0] = w[suff + "_lat_deg"]
            grid[i, 1] = w[suff + "_lat_min"]
            # grid[i, 2] = w[suff+"_lat_sec"]
            i += 1

            # time
            if suff != "ship":
                grid[i, 0] = w[suff + "_hour"]
                grid[i, 1] = w[suff + "_minute"]
                grid[i, 2] = w[suff + "_second"]
                i += 1

        # grid[i, 0] = w["square_radius"]
        # grid[i, 1] = w["square_theta"]
        # grid[i, 2] = w["square_center"]
        # i+=1

        grid[i, 0] = w["time_waiting"]
        grid[i, 1] = w["skip"]

        # store
        self.grid = grid
        self.w = w

    def update(self, button):
        # _now = datetime.utcnow()
        w = self.w

        # self.w["date"].value = _now
        # self.w["hour"].value = _now.hour
        # self.w["minute"].value = _now.minute
        # self.w["second"].value = _now.second
        ll = {}
        for suff in ["ship", "anchor"]:
            lond = w[suff + "_lon_deg"].value
            lon = float(lond[:-1])
            if lond[-1] == "W":
                sign = -1
            else:
                sign = 1
            # lon = sign * (lon + w[suff+"_lon_min"].value/60 + w[suff+"_lon_sec"].value/3600)
            lon = sign * (lon + w[suff + "_lon_min"].value / 60)
            #
            latd = w[suff + "_lat_deg"].value
            lat = float(latd[:-1])
            if latd[-1] == "N":
                sign = 1
            else:
                sign = -1
            # lat = sign * (lat + w[suff+"_lat_min"].value/60 + w[suff+"_lat_sec"].value/3600)
            lat = sign * (lat + w[suff + "_lat_min"].value / 60)
            ll[suff] = dict(lon=lon, lat=lat)
        ocean_current = w["current_speed_x"].value + 1j * w["current_speed_y"].value
        time_waiting = w["time_waiting"].value * 60.0  # converts to seconds
        date = self.now
        time_a = pd.Timestamp(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=w["anchor_hour"].value,
            minute=w["anchor_minute"].value,
            second=w["anchor_second"].value,
        )
        print(ll)
        df = deployments_route_schedule(
            ll["ship"]["lon"],
            ll["ship"]["lat"],
            w["ship_speed"].value * knot,
            self.lon_vertices,
            self.lat_vertices,
            time_waiting,
            ocean_current,
            ll["anchor"]["lon"],
            ll["anchor"]["lat"],
            time_a=time_a,
            skip=w["skip"].value,
        )

        with self.out:
            clear_output()
            print(
                f"New hello from each button click!. This hello from {self.update_num}"
            )
            for label, r in df.iterrows():
                _lon_deg, _lon_min = deg_mindec(r.lon)
                _lat_deg, _lat_min = deg_mindec(r.lat)
                if _lon_deg <= 0:
                    EW = "W"
                else:
                    EW = "E"
                time = r.time.strftime("%Y/%m/%d %H:%M:%S")
                print(
                    f" drifter {label}: {_lon_deg}{EW} {_lon_min:.3f}  {_lat_deg}N {_lat_min:.3f} "
                    f"at {time}"
                )
            print(df)
            self.update_num += 1

        self.df = df


# ---------------------------- deployment geometry -------------------------------


def build_polygon(L, theta, N, rotation=1):
    """build a polygon with radius L and rotation angle theta

    Parameters
    ----------
    L: float
        Radius
    theta: float
        rotation angle of the first vertex [radians]
    N: int
        Number of vertices
    rotation: -1,1
        sense of rotation: 1 = counter-clockwise, -1 = clockwise
    """
    X = (
        L
        * (np.exp(rotation * 1j * 2 * np.arange(N) * np.pi / N) - 1)
        * np.exp(1j * theta)
    )
    dX = np.diff(X)
    return X, dX


def build_square_with_center(L, theta, rotation=1, center_loc=1):
    """build a square with center

    Parameters
    ----------
    L: float
        Radius
    theta: float
        rotation angle of the first vertex [radians]
    N: int
        Number of vertices
    rotation: -1,1
        sense of rotation: 1 = counter-clockwise, -1 = clockwise
    center_loc: int
        location of the center
    """
    # build square without center
    X, _ = build_polygon(L, theta, 4, rotation=rotation)
    # add center
    X = np.insert(X, center_loc, X.mean())
    # put first vertex at origin
    X = X - X[0]
    return X, np.diff(X)


def plot_polygon(X, dX=None):
    """plot polygon, not geographic"""
    fig, ax = plt.subplots(1, 1)
    ax.scatter(X.real, X.imag, s=300, marker=".")

    if dX is not None:
        opt = {
            "head_width": 50,
            "head_length": 50,
            "width": 10,
            "length_includes_head": False,
        }
        for i in range(len(X) - 1):
            ax.arrow(X[i].real, X[i].imag, dX[i].real, dX[i].imag, fc="k", **opt)

    for i in range(X.size):
        ax.text(X[i].real + 100, X[i].imag, f"{i}")

    ax.set_aspect("equal")
    ax.grid()


def build_square_geo(lon_a, lat_a, L, theta, **kwargs):
    """build square geographic coordinates, along with other useful informations

    Parameters
    ----------
    lon_a, lat_a: floats
        Anchor point (first vertex) longitude and latitude
    L: float
        Radius
    theta: float
        rotation angle of the first vertex [radians]
    **kwargs: passed to build_square_with_center

    Returns
    -------
    lon, lat: np.array
        vertex coordinates
    X: np.array
        vertex positions as complex numbers
    """
    # convert lon/lat to local coordinates
    x_a, y_a = lonlat2xy(lon_a, lat_a)
    # build polygon
    X, dX = build_square_with_center(L, theta, **kwargs)
    # offset local coordinates and compute lon/lat
    x, y = x_a + X.real, y_a + X.imag
    lon, lat = xy2lonlat(x, y)
    return lon, lat, x + 1j * y


def radiator(R, N, theta, lonc, latc):
    """Build a radiator (geodetic version)

    Parameters
    ----------
    R: float
        Radius in meters
    N: int
        Number of segments (odd preferentially)
    theta: float
        Orientation angle in radians

    Returns
    -------
    lon, lat: np.array
        Vertex coordinates
    length: float
        Length of the radiators in meters
    dX: np.array
        vertex position in projected coordinates
    """
    dX, length = radiator_xy(R, N, theta)

    # convert to lon/lat
    xc, yc = lonlat2xy(lonc, latc)
    x = xc + dX.real
    y = yc + dX.imag
    lon, lat = xy2lonlat(x, y)

    return lon, lat, length, dX


def radiator_xy(R, N, theta):
    """Build a radiator:

    Parameters
    ----------
    R: float
        Radius in meters
    N: int
        Number of segments (odd preferentially)
    theta: float
        Orientation angle in radians

    Returns
    -------
    X: np.array
        Vertex positions as complex numbers
    length: float
        Length of the radiators in meters
    """
    ex, ey = 1, 1j
    s = np.sqrt(2)
    delta = s / (N - 1)
    x0 = s / 2 * (ex + ey)
    print(x0)
    X = [x0]
    dX = [-s * ex, -delta * ey, s * ex, -delta * ey]
    for i in range((N - 1) // 2):
        for dx in dX:
            X.append(X[-1] + dx)
    X.append(X[-1] - s * ex)
    X = R * np.array(X) * np.exp(1j * theta)
    length = sum(np.abs(np.diff(X)))
    print(f"Distance {length/nmile} NM")
    return X, length


# ---------------------------- drifter data -------------------------------


def fetch_drifter_data(timestamp=True, verbose=True, alldata=True):
    """fetch drifter data from pacific gyre website"""

    with open("pacific_url", "r") as f:
        # do things with your file
        url = f.read().strip()
    if alldata:
        url = url.replace("&maxRowsEach=2", "")
    if timestamp:
        t = now()
        tstamp = "_" + t.strftime("%Y%m%d_%H%M%S")
    else:
        tstamp = ""
    file = "drifter_data" + tstamp + ".csv"

    urllib.request.urlretrieve(url, file)

    if verbose:
        print(f" Drifter data downloaded as {file}")

    return file


def find_latest_drifter_file():
    """find latest drifter file"""
    local_dir = os.getcwd()
    drifter_files = glob(os.path.join(local_dir, "drifter_data*.csv"))
    mtimes = [os.path.getmtime(f) for f in drifter_files]
    latest_file = drifter_files[mtimes.index(max(mtimes))]
    return latest_file


def load_drifter_data(last=None, file=None):
    """Load drifter data into a dict of dataframes"""
    if last is None and file is None:
        # concatenate all files
        local_dir = os.getcwd()
        drifter_files = glob(os.path.join(local_dir, "drifter_data*.csv"))
        DR = [load_drifter_data(file=f) for f in drifter_files]
        drifter_ids = set(sum([list(dr) for dr in DR], []))
        dr = dict()
        for i in drifter_ids:
            dr[i] = (
                pd.concat([_dr[i] for _dr in DR if i in _dr])
                .drop_duplicates()
                .sort_index()
            )
        return dr
    # load just one file
    if last:
        # load last file
        file = find_latest_drifter_file()
    #
    df = pd.read_csv(file, parse_dates=["DeviceDateTime"])
    dr = {k: v.sort_values("DeviceDateTime") for k, v in df.groupby("CommId")}
    #
    for k, v in dr.items():
        add_xy_uv(v)
        dr[k] = v.rename(
            columns=dict(
                Longitude="longitude", Latitude="latitude", DeviceDateTime="time"
            )
        ).set_index("time")
    return dr


def add_xy_uv(df):
    """derive variables from drifter data"""
    # compute local coordinates x, y
    df.loc[:, ["x", "y"]] = df.apply(
        lambda r: pd.Series(lonlat2xy(r["Longitude"], r["Latitude"]), index=["x", "y"]),
        axis=1,
    )
    # compute velocity
    df.loc[:, "dt"] = df.DeviceDateTime.diff() / pd.Timedelta("1s")
    df.loc[:, "u"] = df.x.diff() / df.dt
    df.loc[:, "v"] = df.y.diff() / df.dt
    return df


def extrapolate_one(df, time):
    """Extrapolate positions for one drifter"""
    dfl = df.reset_index().iloc[-1]
    time_last = dfl.time
    dt = (time - time_last) / pd.Timedelta("1s")
    #
    dx, dy = dfl.u * dt, dfl.v * dt
    x, y = dfl.x + dx, dfl.y + dy
    #
    lon, lat = xy2lonlat(x, y)
    return pd.Series(dict(time=time, longitude=lon, latitude=lat, x=x, y=y))


def extrapolate(dr, time=None):
    """extrapolate drifter position to new one"""
    if time is None:
        time = now()
    return {k: extrapolate_one(df, time) for k, df in dr.items()}


# ---------------------------- drifter monitoring ---------------------------------------


def show_last_positions(dr, dr_now, offset="1h", **kwargs):
    """Show last positions

    Parameters
    ----------
    dr: dict
        Dictionary of drifter dataframes
    dr_now: dict
        Dictionary of extrapolated positions (not used at the moment)
    offset: str
        Time offset specifying last amount of data shown
    **kwargs: passed to plot_bs
    """
    from .utils import get_cmap_colors

    for key, d in dr.items():
        dl = d.iloc[-1]
        _lon_deg, _lon_min = deg_mindec(dl.longitude)
        _lat_deg, _lat_min = deg_mindec(dl.latitude)
        if _lon_deg <= 0:
            EW = "W"
        else:
            EW = "E"
        time = dl.name.strftime("%Y/%m/%d %H:%M:%S")
        print(
            f" drifter {key}: {_lon_deg}{EW} {_lon_min:.3f}  {_lat_deg}N {_lat_min:.3f} "
            f" speed=({dl.u:.2f}, {dl.v:.2f}) at {time}"
        )

    dkwargs = dict(
        bathy=False,
        zoom=[-0.4, -0.1, 49.27, 49.4],
        vmax=30,
        figsize=(10, 10),
        land=dict(scale="10m"),
        coast_resolution=None,
    )
    dkwargs.update(**kwargs)

    fac = plot_bs(**dkwargs)

    ax = fac["ax"]

    colors = {key: c for key, c in zip(dr, get_cmap_colors(len(dr)))}
    for key, d in dr.items():
        dl = d.last(offset)
        ax.scatter(
            dl.longitude,
            dl.latitude,
            30,
            color=colors[key],
            transform=ccrs.PlateCarree(),
        )
        ax.scatter(
            dl.longitude,
            dl.latitude,
            30,
            color=colors[key],
            transform=ccrs.PlateCarree(),
        )
        ax.text(
            dl.longitude[-1] + 1e-3,
            dl.latitude[-1],
            f"{key}",
            transform=ccrs.PlateCarree(),
        )


def monitor_drifters(refresh_time=5, **kwargs):
    """Continuous monitoring of drifter positions
    Stop with `Control-C`

    Parameters
    ----------
    refresh_time: float
        Minutes between refreshes
    **kwargs passed to plot_bs, e.g.:
        zoom=[-.35, -.1, 49.28, 49.45], ...
    """

    refresh_time = refresh_time * 60  # converts to seconds

    dkwargs = dict(
        bathy=True, zoom=[-0.35, -0.1, 49.28, 49.45], vmax=30, figsize=(10, 10)
    )
    dkwargs.update(**kwargs)

    # init figure
    fac = plot_bs(**dkwargs)
    fig, ax = fac["fig"], fac["ax"]

    first = True
    while True:

        clear_output(wait=True)

        # print useful data:
        _now = now().strftime("%Y/%m/%d %H:%M:%S")
        print(f" Update at {_now}:")

        # fetch data from PacificGyre website
        fetch_drifter_data(alldata=first)

        # load all files
        dr = load_drifter_data()

        # add fake drifters # dev
        if False:
            d = dr["0-4351896"]
            dt = (d.index - d.index[0]) / pd.Timedelta("1s")
            d.loc[:, "longitude"] = -0.2 + 2 * 1 * dt.values / 111e3
            d.loc[:, "latitude"] = 49.35 + 2 * 0.25 * dt.values / 111e3
            dr["fake"] = d

        # extrapolate
        dr_now = extrapolate(dr)
        # dr_now = extrapolate(dr, time=dr["fake"].index[-1]+pd.Timedelta("10min")) # dev

        for key, d in dr.items():
            _dl = d.iloc[-1]
            _tl = d.index[-1].strftime("%Y/%m/%d")
            _dn = dr_now[key]
            _tn = _dn.time.strftime("%Y/%m/%d")
            print(
                f" {key} last: t={_tl} lon={_dl.longitude:.4f}, lat={_dl.latitude:.4f}"
            )
            print(
                f"        now: t={_tl} lon={_dn.longitude:.4f}, lat={_dn.latitude:.4f}"
            )

        # plot data
        fac = plot_bs(fig=fig, **dkwargs)
        ax = fac["ax"]
        ax.set_title(f"{_now}")

        for key, d in dr.items():
            # if key=="fake": # dev
            ax.scatter(
                d.longitude,
                d.latitude,
                40,
                color="orange",
                transform=ccrs.PlateCarree(),
            )
            ax.plot(
                d.longitude,
                d.latitude,
                lw=2,
                color="orange",
                transform=ccrs.PlateCarree(),
            )
            _lons, _lats = d.iloc[0, :].loc[["longitude", "latitude"]]
            ax.text(_lons + 1e-3, _lats, f"{key}", transform=ccrs.PlateCarree())
            #
            transform = ccrs.PlateCarree()._as_mpl_transform(ax)
            dn = dr_now[key]
            ax.annotate(
                "",
                xy=(dn["longitude"], dn["latitude"]),
                xytext=(d.iloc[-1]["longitude"], d.iloc[-1]["latitude"]),
                xycoords=transform,
                size=20,
                arrowprops=dict(
                    facecolor="0.5",
                    ec="none",
                    # arrowstyle="fancy",
                    connectionstyle="arc3,rad=-0.3",
                ),
            )

        display(fig)
        fig.clf()

        # goes to sleep for a bit
        time.sleep(refresh_time)
        first = False


# ---------------------------- drifter data; manual logging -----------------------------


class dashboard_log(object):
    def __init__(self, devices=None, log_path=None):
        if devices is None:
            devices = []
        self.devices = devices  # devices
        #
        self.build_dashboard()
        self.update()  # time, devices
        if log_path is None:
            log_path = "dashboard.log"
        self.log = log_path

    def build_dashboard(self):

        w = dict()

        w["device"] = widgets.Dropdown(
            options=self.devices,
            value=None,
            description="Device:",
            disabled=False,
        )

        w["button"] = widgets.Button(
            description="Register",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Click me",
            icon="check",  # (FontAwesome names without the `fa-` prefix)
        )
        w["button"].on_click(self.register)

        w["date"] = widgets.DatePicker(description="Pick a Date", disabled=False)

        w["hour"] = widgets.Dropdown(
            options=np.arange(24),
            description="Hour:",
            disabled=False,
        )

        w["minute"] = widgets.Dropdown(
            options=np.arange(60),
            description="Minute:",
            disabled=False,
        )

        w["second"] = widgets.Dropdown(
            options=np.arange(60),
            description="Second:",
            disabled=False,
        )

        w["lon_deg"] = widgets.Dropdown(
            options=["0W"],  # to be generalized
            value="0W",
            description="lon - deg:",
            disabled=False,
        )

        w["lon_min"] = widgets.Dropdown(
            options=np.arange(60),
            value=15,
            description="lon [min]:",
            disabled=False,
        )

        w["lon_sec"] = widgets.FloatSlider(
            value=0.0,
            min=0,
            max=60.0,
            step=0.1,  # 0.1 sec = 2m
            description="lon [sec]:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".1f",
        )

        w["lat_deg"] = widgets.Dropdown(
            options=["49N"],  # to be generalized
            value="49N",
            description="lat [deg]:",
            disabled=False,
        )

        w["lat_min"] = widgets.Dropdown(
            options=np.arange(60),
            value=19,
            description="lat [min]:",
            disabled=False,
        )

        w["lat_sec"] = widgets.FloatSlider(
            value=0.0,
            min=0,
            max=60.0,
            step=0.1,  # 0.1 sec = 2m
            description="lat - sec:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".1f",
        )

        #
        grid = widgets.GridspecLayout(4, 3, height="200px", align_items="center")

        grid[0, 0] = w["device"]
        grid[0, 1] = w["button"]

        #
        grid[0, 2] = w["date"]
        grid[1, 2] = w["hour"]
        grid[2, 2] = w["minute"]
        grid[3, 2] = w["second"]

        #
        grid[1, 0] = w["lon_deg"]
        grid[2, 0] = w["lon_min"]
        grid[3, 0] = w["lon_sec"]

        #
        grid[1, 1] = w["lat_deg"]
        grid[2, 1] = w["lat_min"]
        grid[3, 1] = w["lat_sec"]

        # store
        self.grid = grid
        self.w = w

    def update(self):
        _now = datetime.utcnow()
        self.w["date"].value = _now
        self.w["hour"].value = _now.hour
        self.w["minute"].value = _now.minute
        self.w["second"].value = _now.second

    def register(self, button):
        w = self.w
        #
        date = w["date"].value
        time = pd.Timestamp(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=w["hour"].value,
            minute=w["minute"].value,
            second=w["second"].value,
        )
        #
        lond = w["lon_deg"].value
        lon = float(lond[:-1])
        if lond[-1] == "W":
            sign = -1
        else:
            sign = 1
        lon = sign * (lon + w["lon_min"].value / 60 + w["lon_sec"].value / 3600)
        #
        latd = w["lat_deg"].value
        lat = float(latd[:-1])
        if latd[-1] == "N":
            sign = 1
        else:
            sign = -1
        lat = sign * (lat + w["lat_min"].value / 60 + w["lat_sec"].value / 3600)
        #
        s = pd.Series(
            dict(
                device=w["device"].value,
                time=time,
                lon=lon,
                lat=lat,
            )
        )
        # store
        (
            s.to_frame()
            .T.rename_axis("event")
            .to_csv(self.log, mode="a", header=not os.path.exists(self.log))
        )

    def load_log(self):
        """load log file"""
        df = pd.read_csv(d.log, parse_dates=["time"])
        return df
