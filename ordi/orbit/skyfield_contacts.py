"""Skyfield/SGP4 contact graph backend."""
from __future__ import annotations
import datetime as dt
import math
import numpy as np
from skyfield.api import EarthSatellite, load, wgs84
from sgp4.api import Satrec, WGS84
from ordi.orbit._contact_types import ContactEvent, DEFAULT_GROUND_STATIONS, GS_MIN_ELEVATION_DEG, ISL_MAX_RANGE_KM, DOWNLINK_RATE_BPS, ISL_RATE_BPS, UPLINK_RATE_BPS

def build_synthetic_walker(n_planes=6, sats_per_plane=6, alt_km=550.0, inc_deg=53.0, epoch_str="2024-01-01"):
    if epoch_str != "2024-01-01": raise ValueError("only the deterministic epoch is supported")
    ts=load.timescale(); out=[]; total=n_planes*sats_per_plane
    n=math.sqrt(3.986004418e14/((6371+alt_km)*1e3)**3)*60
    for p in range(n_planes):
        for s in range(sats_per_plane):
            satrec=Satrec(); satrec.sgp4init(WGS84,'i',p*100+s,7306.0,0,0,0,0.001,0,math.radians(inc_deg),math.radians((360*s/sats_per_plane+360*p/total)%360),n,math.radians(360*p/n_planes))
            sat=EarthSatellite.from_satrec(satrec,ts); sat.name=f"SAT_{p:02d}_{s:02d}"; out.append(sat)
    return out

def _unix(tt): return 946727935.816+(tt-2451545.0)*86400.0

def compute_contact_windows(satellites,t_start_unix,t_end_unix,ground_stations=None,isl_max_range_km=ISL_MAX_RANGE_KM,dt_seconds=30.0,min_elevation_deg=GS_MIN_ELEVATION_DEG):
    stations=ground_stations or DEFAULT_GROUND_STATIONS; ts=load.timescale(); events=[]
    start=ts.from_datetime(dt.datetime.fromtimestamp(t_start_unix,dt.timezone.utc)); end=ts.from_datetime(dt.datetime.fromtimestamp(t_end_unix,dt.timezone.utc))
    for sat in satellites:
        for name,lat,lon in stations:
            times,codes=sat.find_events(wgs84.latlon(lat,lon),start,end,altitude_degrees=min_elevation_deg); rise=None
            for t,code in zip(times,codes):
                if code==0: rise=t.tt
                elif code==2:
                    t0=_unix(rise if rise is not None else start.tt); t1=_unix(t.tt); events += [ContactEvent(t0,t1,sat.name,name,DOWNLINK_RATE_BPS,'downlink'),ContactEvent(t0,t1,name,sat.name,UPLINK_RATE_BPS,'uplink')]; rise=None
    nsteps=int((t_end_unix-t_start_unix)/dt_seconds)+1; tt=np.linspace(start.tt,end.tt,nsteps); times=np.array([_unix(x) for x in tt]); pos=np.array([sat.at(ts.tt_jd(tt)).position.km.T for sat in satellites])
    for i in range(len(satellites)):
        for j in range(i+1,len(satellites)):
            active=np.linalg.norm(pos[i]-pos[j],axis=1)<=isl_max_range_km; begin=None
            for k,on in enumerate(np.r_[active,False]):
                if on and begin is None: begin=times[k]
                elif not on and begin is not None:
                    events += [ContactEvent(begin,times[k],satellites[i].name,satellites[j].name,ISL_RATE_BPS,'isl'),ContactEvent(begin,times[k],satellites[j].name,satellites[i].name,ISL_RATE_BPS,'isl')]; begin=None
    return sorted(events,key=lambda e:e.t_start)

def compute_sat_groundtracks(satellites,t_start_unix,t_end_unix,dt_seconds=60.0):
    ts=load.timescale(); out={}; times=np.arange(t_start_unix,t_end_unix+dt_seconds,dt_seconds)
    for sat in satellites:
        out[sat.name]=[]
        for t in times:
            sub=wgs84.subpoint_of(sat.at(ts.from_datetime(dt.datetime.fromtimestamp(float(t),dt.timezone.utc))))
            out[sat.name].append((float(t),sub.latitude.degrees,sub.longitude.degrees))
    return out
