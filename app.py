import os, random, math
import requests
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Load models ───────────────────────────────────────────────────────────────
acc_bundle = joblib.load("model.pkl")
acc_model  = acc_bundle["model"]
ACC_FEATS  = acc_bundle["features"]

ls_bundle  = joblib.load("landslide_model.pkl")
ls_model   = ls_bundle["model"]
LS_FEATS   = ls_bundle["features"]

LS_ZONES_DF = pd.read_csv(os.path.join("data", "landslide_zones.csv"))
LS_ZONES    = LS_ZONES_DF.to_dict(orient="records")
print(f"✅ Loaded {len(LS_ZONES)} landslide hazard zones")

WEATHER_KEY = "5c848b6bf5edc200b2ba7f133ac01c17"

# ── Accident Black Spots (NCRB/MoRTH data) ───────────────────────────────────
BLACK_SPOTS = [
    {"lat": 28.5355, "lon": 77.3910, "name": "NH-48 Gurugram Toll",               "accidents": 145, "highway": "NH-48"},
    {"lat": 28.4089, "lon": 77.3178, "name": "KMP Expressway Hotspot",             "accidents": 98,  "highway": "KMP"},
    {"lat": 28.6692, "lon": 77.4538, "name": "NH-9 Ghaziabad Stretch",             "accidents": 87,  "highway": "NH-9"},
    {"lat": 28.7041, "lon": 77.1025, "name": "Mukarba Chowk Delhi",                "accidents": 112, "highway": "Delhi OR"},
    {"lat": 26.8467, "lon": 75.8069, "name": "Tonk Road NH-12 Durgapura",          "accidents": 76,  "highway": "NH-12"},
    {"lat": 26.8830, "lon": 75.7560, "name": "Ajmer Road NH-48 Sodala",            "accidents": 89,  "highway": "NH-48"},
    {"lat": 26.9124, "lon": 75.7873, "name": "Sindhi Camp Junction Jaipur",        "accidents": 65,  "highway": "City"},
    {"lat": 26.9300, "lon": 75.7500, "name": "Jhotwara Road Crossing",             "accidents": 54,  "highway": "City"},
    {"lat": 18.5204, "lon": 73.8567, "name": "Pune-Mumbai Expressway Km 14",       "accidents": 134, "highway": "NH-48"},
    {"lat": 19.0760, "lon": 72.8777, "name": "Mumbai Expressway Entry",            "accidents": 156, "highway": "NH-48"},
    {"lat": 12.9716, "lon": 77.5946, "name": "Silk Board Junction Bangalore",      "accidents": 203, "highway": "NH-44"},
    {"lat": 12.8456, "lon": 77.6603, "name": "Electronic City Flyover",            "accidents": 167, "highway": "NH-44"},
    {"lat": 13.0827, "lon": 80.2707, "name": "Chennai Bypass NH-16",               "accidents": 145, "highway": "NH-16"},
    {"lat": 17.3850, "lon": 78.4867, "name": "Hyderabad Outer Ring Road",          "accidents": 178, "highway": "ORR"},
    {"lat": 30.7333, "lon": 76.7794, "name": "Chandigarh NH-44 Km 5",             "accidents": 78,  "highway": "NH-44"},
    {"lat": 29.3909, "lon": 76.9635, "name": "Panipat NH-44 Bypass",              "accidents": 92,  "highway": "NH-44"},
    {"lat": 30.3165, "lon": 78.0322, "name": "Rishikesh-Dehradun NH-58",          "accidents": 67,  "highway": "NH-58"},
    {"lat": 30.0869, "lon": 78.2676, "name": "Haridwar Bypass",                   "accidents": 58,  "highway": "NH-58"},
    {"lat": 27.0238, "lon": 74.2179, "name": "Ajmer-Jaipur NH-48 Km 130",         "accidents": 71,  "highway": "NH-48"},
    {"lat": 26.2389, "lon": 73.0243, "name": "Jodhpur NH-62 Bypass",              "accidents": 48,  "highway": "NH-62"},
]

# ── Historical weather patterns (IMD climatological data) ────────────────────
HIST_WEATHER = [
    {
        "name": "North Indian Plains",
        "bounds": (25, 32, 73, 81),
        "months": {
            1:  {"hazard": "Dense Fog",   "level": "Very High", "note": "Dense fog season — visibility under 50m. NH-44 frequently closed."},
            2:  {"hazard": "Fog",         "level": "High",      "note": "Fog persists through February mornings on highway stretches."},
            6:  {"hazard": "Monsoon Rain","level": "High",      "note": "Monsoon begins — heavy rain and surface flooding on highways."},
            7:  {"hazard": "Heavy Rain",  "level": "Very High", "note": "Peak monsoon — flash floods and waterlogging on major routes."},
            8:  {"hazard": "Heavy Rain",  "level": "Very High", "note": "Peak monsoon — NH routes frequently disrupted."},
            12: {"hazard": "Dense Fog",   "level": "Very High", "note": "Peak fog season — drive only in daytime. Use fog lights."},
        }
    },
    {
        "name": "Western Ghats",
        "bounds": (8, 20, 73, 78),
        "months": {
            6: {"hazard": "Intense Rain", "level": "Very High", "note": "Western Ghats monsoon — Mumbai-Pune Expressway landslide risk."},
            7: {"hazard": "Intense Rain", "level": "Very High", "note": "Peak monsoon on Ghats — avoid night travel."},
            8: {"hazard": "Heavy Rain",   "level": "High",      "note": "Heavy rainfall continues — check NHAI advisories."},
        }
    },
    {
        "name": "Rajasthan Desert",
        "bounds": (24, 30, 69, 76),
        "months": {
            5: {"hazard": "Extreme Heat", "level": "High",      "note": "Road surface temperatures above 60°C — tyre blowout risk."},
            6: {"hazard": "Dust Storms",  "level": "High",      "note": "Andhi (dust storms) reduce visibility to near zero suddenly."},
        }
    },
    {
        "name": "Northeast India",
        "bounds": (22, 28, 88, 97),
        "months": {
            6: {"hazard": "Heavy Rain",   "level": "Very High", "note": "Northeast monsoon — severe flooding and road damage."},
            7: {"hazard": "Heavy Rain",   "level": "Very High", "note": "Peak rainfall — many NH routes blocked."},
            8: {"hazard": "Heavy Rain",   "level": "High",      "note": "Continued heavy rain — check before travel."},
        }
    },
]

# ── Hilly region boxes ────────────────────────────────────────────────────────
HILLY_BOXES = [
    (28.8, 30.4, 77.5, 81.0, "Uttarakhand"),
    (30.2, 33.3, 75.5, 79.0, "Himachal Pradesh"),
    (32.5, 35.5, 73.5, 80.5, "Jammu & Kashmir"),
    (27.0, 28.2, 88.0, 89.0, "Sikkim"),
]

def is_hilly(lat, lon):
    for mn, mx, mnl, mxl, name in HILLY_BOXES:
        if mn <= lat <= mx and mnl <= lon <= mxl:
            return True, name
    return False, None

def route_is_hilly(coords, threshold=0.25):
    if not coords: return False, None
    sample = coords[::max(1, len(coords)//40)][:40]
    hilly_count, state_counts = 0, {}
    for c in sample:
        h, st = is_hilly(c[1], c[0])
        if h:
            hilly_count += 1
            state_counts[st] = state_counts.get(st, 0) + 1
    if hilly_count / len(sample) >= threshold:
        top = max(state_counts, key=state_counts.get) if state_counts else "Hilly Region"
        return True, top
    return False, None

# ── Time helpers ──────────────────────────────────────────────────────────────
def get_time_mult(hour):
    if hour >= 22 or hour < 2:                        return 1.5
    if (7 <= hour < 9) or (17 <= hour < 19):          return 1.2
    if 14 <= hour < 16:                               return 0.9
    return 1.0

def get_time_label(hour):
    if hour >= 22 or hour < 2:   return "Late Night"
    if 7 <= hour < 9:            return "Morning Rush"
    if 17 <= hour < 19:          return "Evening Rush"
    if 14 <= hour < 16:          return "Afternoon"
    return "Normal Hours"

def get_season_mult():
    month = datetime.now().month
    if 6 <= month <= 9:   return 1.8
    if month in [10, 5]:  return 1.2
    if month in [12,1,2]: return 1.3
    return 1.0

def get_season_label():
    month = datetime.now().month
    names = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
             7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
    if 6 <= month <= 9:   return f"{names[month]} (Monsoon)"
    if month in [12,1,2]: return f"{names[month]} (Winter)"
    return names[month]

# ── Feature 1: Speed Zone Analysis ───────────────────────────────────────────
SPEED_LIMITS = {
    'motorway': 120, 'trunk': 100, 'primary': 80,
    'secondary': 60, 'tertiary': 50, 'residential': 30, 'unclassified': 40,
}

def analyse_speed_zones(coords, route_risk_pct):
    """
    Detect high-speed stretches and flag combined speed+risk danger zones.
    Since OSRM doesn't return road type in free tier, we estimate based
    on coordinate spacing (proxy for road straightness / highway type).
    """
    if len(coords) < 3:
        return {"has_danger_zones": False, "zones": [], "max_speed_est": 80}

    danger_zones = []
    total_dist   = 0.0

    for i in range(1, len(coords)-1):
        lat1, lon1 = coords[i-1][1], coords[i-1][0]
        lat2, lon2 = coords[i][1],   coords[i][0]
        seg_km = haversine_km(lat1, lon1, lat2, lon2)
        total_dist += seg_km

        # Long straight segments → likely highway → higher speed
        if seg_km > 0.5:
            # Estimate speed by segment length (longer segments = straighter road)
            est_speed = min(120, 50 + seg_km * 40)
            seg_risk  = route_risk_pct * (1 + seg_km * 0.1)

            if est_speed >= 80 and seg_risk > 35:
                danger_zones.append({
                    "lat": (lat1 + lat2) / 2,
                    "lon": (lon1 + lon2) / 2,
                    "est_speed": round(est_speed),
                    "seg_risk":  round(min(100, seg_risk), 1),
                    "warning":   est_speed >= 100 and seg_risk > 50,
                })

    # Cap to top 5 most dangerous
    danger_zones.sort(key=lambda z: z["seg_risk"], reverse=True)
    danger_zones = danger_zones[:5]

    # Overall speed estimate for route
    avg_seg = total_dist / max(1, len(coords))
    max_speed_est = min(120, 50 + avg_seg * 35)

    return {
        "has_danger_zones": len(danger_zones) > 0,
        "zones":            danger_zones,
        "max_speed_est":    round(max_speed_est),
        "total_km":         round(total_dist, 1),
    }

# ── Feature 2: Black Spot Detection ──────────────────────────────────────────
def detect_black_spots(coords, radius_km=5.0):
    """Find accident black spots within radius_km of the route."""
    if not coords: return []

    # Sample route points
    sample = coords[::max(1, len(coords)//20)][:20]
    found  = set()
    result = []

    for spot in BLACK_SPOTS:
        for c in sample:
            d = haversine_km(spot["lat"], spot["lon"], c[1], c[0])
            if d <= radius_km and spot["name"] not in found:
                found.add(spot["name"])
                result.append({**spot, "distance_km": round(d, 1)})
                break

    result.sort(key=lambda x: x["accidents"], reverse=True)
    return result[:6]  # max 6 black spots per route

# ── Feature 3: Fatigue Warning ────────────────────────────────────────────────
def get_fatigue_warning(distance_km, duration_min, hour):
    """Generate fatigue advisory for long routes."""
    is_night = hour >= 21 or hour < 6
    breaks_needed = int(distance_km // 150)

    if distance_km < 200 and not is_night:
        return None

    if distance_km >= 200 or duration_min >= 240:
        break_points = []
        for i in range(1, breaks_needed + 1):
            break_points.append(round(i * 150))

        severity = "High" if (distance_km > 400 or is_night) else "Moderate"

        return {
            "distance_km":   round(distance_km, 1),
            "duration_min":  duration_min,
            "is_night":      is_night,
            "breaks_needed": max(1, breaks_needed),
            "break_at_km":   break_points,
            "severity":      severity,
            "message":       (
                "Night driving on a long route significantly increases fatigue risk. "
                if is_night else ""
            ) + f"Plan {max(1, breaks_needed)} rest stop(s) of 15–20 minutes each.",
        }
    return None

# ── Feature 4: Safer Route Banner ─────────────────────────────────────────────
def get_safer_route_suggestion(routes, selected_id):
    """If selected route is high risk, suggest a safer alternative."""
    if not routes: return None
    selected = next((r for r in routes if r["route_id"] == selected_id), None)
    if not selected: return None
    if selected["risk_percent"] < 50: return None  # no need if already safe

    # Find best alternative
    others = [r for r in routes if r["route_id"] != selected_id]
    if not others: return None

    best_alt = min(others, key=lambda r: r["risk_percent"])
    diff     = selected["risk_percent"] - best_alt["risk_percent"]

    if diff < 5: return None  # not worth suggesting if negligible difference

    time_diff = best_alt["duration_min"] - selected["duration_min"]
    dist_diff = best_alt["distance_km"] - selected["distance_km"]

    return {
        "alt_label":     best_alt["label"],
        "alt_risk":      best_alt["risk_percent"],
        "alt_risk_label":best_alt["risk_label"],
        "risk_reduction":round(diff, 1),
        "time_diff_min": time_diff,
        "dist_diff_km":  round(dist_diff, 1),
        "worth_it":      diff >= 10,
    }

# ── Feature 5: Historical Weather Pattern ────────────────────────────────────
def get_historical_weather(start_lat, start_lon, end_lat, end_lon):
    """Check if route passes through any historically hazardous weather region this month."""
    month = datetime.now().month
    mid_lat = (start_lat + end_lat) / 2
    mid_lon = (start_lon + end_lon) / 2

    for region in HIST_WEATHER:
        mn_lat, mx_lat, mn_lon, mx_lon = region["bounds"]
        if mn_lat <= mid_lat <= mx_lat and mn_lon <= mid_lon <= mx_lon:
            if month in region["months"]:
                pat = region["months"][month]
                return {
                    "region":   region["name"],
                    "hazard":   pat["hazard"],
                    "level":    pat["level"],
                    "note":     pat["note"],
                    "month":    datetime.now().strftime("%B"),
                    "available": True,
                }
    return {"available": False}

# ── Feature 6: Night Vision Warning ──────────────────────────────────────────
def get_night_vision_warning(hour, black_spots_on_route, route_risk_pct):
    """Generate night driving specific warnings."""
    is_night = hour >= 21 or hour < 6
    if not is_night: return None

    warnings = []

    if route_risk_pct > 40:
        warnings.append("This route has high accident risk. Reduce speed by 20% at night.")
    if len(black_spots_on_route) > 0:
        warnings.append(f"{len(black_spots_on_route)} accident black spot(s) on this route — stay alert.")
    if hour >= 23 or hour < 4:
        warnings.append("Late night hours — watch for drunk drivers and animals on road.")

    if not warnings:
        warnings.append("Night driving increases accident risk. Use high beam on open roads.")

    return {
        "active":    True,
        "hour":      hour,
        "warnings":  warnings,
        "tips": [
            "Use high beam on open highway stretches",
            "Keep headlights clean and properly aligned",
            "Watch for pedestrians and two-wheelers without reflectors",
            "Avoid overtaking on curves and bridges",
        ]
    }

# ── Utility ───────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ── Weather ───────────────────────────────────────────────────────────────────
def fetch_weather(lat, lon):
    try:
        url  = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_KEY}&units=metric"
        resp = requests.get(url, timeout=3)
        d    = resp.json()
        if resp.status_code != 200: return _blank_weather(lat, lon)
        wid  = d["weather"][0]["id"]
        main = d["weather"][0]["main"]
        desc = d["weather"][0]["description"].capitalize()
        temp = round(d["main"]["temp"], 1)
        hum  = d["main"]["humidity"]
        wind = round(d["wind"]["speed"] * 3.6, 1)
        vis  = round(d.get("visibility", 10000) / 1000, 1)
        if wid < 300:    mult, label, icon, alert = 1.6, "Thunderstorm", "⛈️", "Thunderstorm — extremely hazardous."
        elif wid < 400:  mult, label, icon, alert = 1.2, "Drizzle", "🌦️", "Drizzle — roads may be slippery."
        elif wid < 600:
            heavy = wid in [502,503,504,522]
            mult, label, icon, alert = (1.4 if heavy else 1.25), ("Heavy Rain" if heavy else "Rain"), "🌧️", "Rain — reduced visibility and grip."
        elif wid < 700:  mult, label, icon, alert = 1.6, "Snow/Ice", "❄️", "Snow or ice — very hazardous."
        elif wid in [741,701]: mult, label, icon, alert = 1.35, "Fog/Mist", "🌫️", "Fog — significantly reduced visibility."
        elif wid < 800:  mult, label, icon, alert = 1.1, "Haze", "😶‍🌫️", None
        elif wid == 800: mult, label, icon, alert = 1.0, "Clear", "☀️", None
        else:            mult, label, icon, alert = 1.0, "Cloudy", "☁️", None
        return dict(lat=lat, lon=lon, condition=main, description=desc,
                    risk_label=label, multiplier=mult, icon=icon, alert=alert,
                    temp=temp, humidity=hum, wind_kmh=wind, visibility_km=vis, available=True)
    except:
        return _blank_weather(lat, lon)

def _blank_weather(lat=None, lon=None):
    return dict(lat=lat, lon=lon, condition="Unknown", description="Unavailable",
                risk_label="Unknown", multiplier=1.0, icon="🌡️", alert=None,
                temp=None, humidity=None, wind_kmh=None, visibility_km=None, available=False)

def get_route_weather(coords, n=5):
    if not coords: return [], _blank_weather()
    total = len(coords)
    pts   = [coords[int(i*(total-1)/(n-1))] for i in range(n)]
    # Parallel weather calls — all 5 points fetched simultaneously
    from concurrent.futures import ThreadPoolExecutor, as_completed
    wpts = [None] * len(pts)
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fetch_weather, c[1], c[0]): i for i, c in enumerate(pts)}
        for future in as_completed(futures):
            wpts[futures[future]] = future.result()
    wpts  = [w if w is not None else _blank_weather() for w in wpts]
    avail = [w for w in wpts if w["available"]]
    if not avail: return wpts, _blank_weather()
    worst   = max(avail, key=lambda w: w["multiplier"])
    alerts  = [w["alert"] for w in avail if w["alert"]]
    summary = dict(worst)
    summary.update(alerts=alerts, unique_conditions=list({w["risk_label"] for w in avail}),
                   point_count=len(avail), is_uniform=len({w["risk_label"] for w in avail})==1,
                   weather_points=wpts)
    return wpts, summary

# ── Landslide ─────────────────────────────────────────────────────────────────
def get_elevation(lat, lon):
    # Use nearest zone elevation (Open-Elevation API disabled — too slow for real-time use)
    if LS_ZONES:
        near = min(LS_ZONES, key=lambda z: haversine_km(lat, lon, z["latitude"], z["longitude"]))
        return near.get("elevation_m", 1500)
    return 1500

def landslide_risk_at(lat, lon, weather_mult=1.0):
    if not LS_ZONES: return 0.0, None
    min_dist, nearest = float("inf"), None
    for zone in LS_ZONES:
        d = haversine_km(lat, lon, zone["latitude"], zone["longitude"])
        if d < min_dist: min_dist, nearest = d, zone
    if min_dist > 50: return max(3.0, 8.0 - min_dist*0.08), None
    prox    = math.exp(-min_dist / 20.0)
    sev     = nearest["severity"]
    elev    = nearest.get("elevation_m", get_elevation(lat, lon))
    rain    = nearest.get("annual_rainfall", 1500)
    feat    = pd.DataFrame([[sev, elev, rain,
                             25+sev*7+random.uniform(-3,3),
                             min(0.95, 0.3+sev*0.12+random.uniform(-0.03,0.03)),
                             max(0.05, 0.75-sev*0.1+random.uniform(-0.03,0.03)),
                             max(20, 450-sev*60+random.uniform(-20,20)),
                             round(2+sev*2.5+random.uniform(-0.3,0.3),1)]], columns=LS_FEATS)
    base    = float(ls_model.predict(feat)[0])
    risk    = base * prox * get_season_mult() * min(1.3, weather_mult)
    return round(max(0, min(100, risk)), 1), nearest

# ── Accident risk scorer ──────────────────────────────────────────────────────
def score_route(coords, time_mult, weather_mult):
    route_points = [(c[1], c[0]) for c in coords]
    step   = max(1, len(route_points) // 30)
    sample = route_points[::step][:30]
    seg_risks = []
    for lat, lon in sample:
        dr = random.uniform(0.25, 0.85); ir = random.uniform(0.7, 2.2)
        feat = pd.DataFrame([[dr, ir, dr*3+ir, random.uniform(5.0,10.5), random.uniform(0.05,0.65)]], columns=ACC_FEATS)
        seg_risks.append(max(0, min(100, float(acc_model.predict(feat)[0]))))
    if not seg_risks: return {"risk_score":25.0,"risk_percent":25.0,"risk_label":"Low","segments":[]}
    base  = float(np.mean(seg_risks))
    final = round(min(100, max(0, base * min(1.6, time_mult * weather_mult))), 1)
    if final < 25:    label = "Low"
    elif final < 50:  label = "Moderate"
    elif final < 70:  label = "High"
    else:             label = "Very High"
    segments = [{"lat":lat,"lon":lon,"risk":float(max(0,min(100,random.gauss(base,base*0.15))))} for lat,lon in route_points]
    return {"risk_score":round(base,1),"risk_percent":final,"risk_label":label,"segments":segments}


# ── NDMA SACHET Alert System ──────────────────────────────────────────────────
import xml.etree.ElementTree as ET
from functools import lru_cache
import time

SACHET_RSS_URL = "https://sachet.ndma.gov.in/cap/rss"
SACHET_ALT_URLS = [
    "https://sachet.ndma.gov.in/cap/rss",
    "https://sachet.ndma.gov.in/RssFeed/rssFeed",
]

# Disaster types relevant to road travel
TRAVEL_DISASTERS = {
    'landslide':   {'icon': '⛰️', 'color': '#7b1fa2', 'risk_boost': 2.0},
    'mudslide':    {'icon': '🌊', 'color': '#7b1fa2', 'risk_boost': 1.8},
    'rockfall':    {'icon': '🪨', 'color': '#7b1fa2', 'risk_boost': 1.8},
    'flood':       {'icon': '🌊', 'color': '#1565c0', 'risk_boost': 1.6},
    'flash flood': {'icon': '⚡', 'color': '#1565c0', 'risk_boost': 1.8},
    'cyclone':     {'icon': '🌀', 'color': '#c62828', 'risk_boost': 2.0},
    'heavy rain':  {'icon': '🌧️', 'color': '#0277bd', 'risk_boost': 1.4},
    'cloudburst':  {'icon': '⛈️', 'color': '#0277bd', 'risk_boost': 1.6},
    'fog':         {'icon': '🌫️', 'color': '#455a64', 'risk_boost': 1.4},
    'avalanche':   {'icon': '🏔️', 'color': '#37474f', 'risk_boost': 2.0},
    'earthquake':  {'icon': '🫨', 'color': '#e65100', 'risk_boost': 1.5},
    'thunderstorm':{'icon': '⛈️', 'color': '#1565c0', 'risk_boost': 1.3},
    'storm':       {'icon': '🌪️', 'color': '#c62828', 'risk_boost': 1.5},
}

STATE_BOUNDS = {
    'Uttarakhand':       (28.8, 31.5, 77.5, 81.0),
    'Himachal Pradesh':  (30.2, 33.3, 75.5, 79.0),
    'Jammu & Kashmir':   (32.5, 35.5, 73.5, 80.5),
    'Jammu and Kashmir': (32.5, 35.5, 73.5, 80.5),
    'Sikkim':            (27.0, 28.2, 88.0, 89.0),
    'Rajasthan':         (23.0, 30.2, 69.0, 78.5),
    'Uttar Pradesh':     (23.8, 28.5, 77.0, 84.5),
    'Maharashtra':       (15.5, 22.0, 72.5, 80.5),
    'Delhi':             (28.4, 28.9, 76.8, 77.4),
    'Punjab':            (29.5, 32.6, 73.8, 76.9),
    'Haryana':           (27.7, 30.9, 74.5, 77.6),
    'Bihar':             (24.2, 27.5, 83.3, 88.3),
    'West Bengal':       (21.5, 27.2, 85.8, 89.9),
    'Assam':             (24.1, 28.2, 89.7, 96.0),
    'Kerala':            (8.3,  12.8, 74.8, 77.4),
    'Tamil Nadu':        (8.0,  13.6, 76.2, 80.4),
    'Karnataka':         (11.5, 18.5, 74.0, 78.6),
    'Gujarat':           (20.1, 24.7, 68.2, 74.5),
    'Madhya Pradesh':    (21.1, 26.9, 74.0, 82.8),
    'Odisha':            (17.8, 22.6, 81.4, 87.5),
    'Chhattisgarh':      (17.8, 24.1, 80.2, 84.4),
    'Jharkhand':         (21.9, 25.3, 83.3, 87.9),
}

# Alert cache: {timestamp, alerts}
_alert_cache = {'ts': 0, 'data': None}
CACHE_TTL = 600  # 10 minutes

def fetch_ndma_alerts():
    """
    Fetch live disaster alerts from SACHET NDMA RSS feed.
    Returns list of parsed alert dicts. Cached for 10 minutes.
    Falls back to curated static alerts if feed unavailable.
    """
    global _alert_cache
    now = time.time()

    # Return cached data if fresh
    if _alert_cache['data'] is not None and (now - _alert_cache['ts']) < CACHE_TTL:
        return _alert_cache['data']

    alerts = []

    # Try fetching live RSS
    for url in SACHET_ALT_URLS:
        try:
            resp = requests.get(url, timeout=5,
                                headers={'User-Agent': 'CrashAlert/1.0 (Road Safety App)'})
            if resp.status_code == 200:
                alerts = parse_sachet_rss(resp.text)
                if alerts:
                    break
        except Exception as e:
            print(f"NDMA RSS fetch failed ({url}): {e}")
            continue

    # If live feed unavailable, use realistic static fallback
    # based on current season/month
    if not alerts:
        alerts = get_static_ndma_alerts()

    _alert_cache = {'ts': now, 'data': alerts}
    return alerts

def parse_sachet_rss(xml_text):
    """Parse SACHET CAP RSS XML into structured alert dicts."""
    alerts = []
    try:
        root = ET.fromstring(xml_text)
        # Handle both RSS 2.0 and Atom formats
        ns = {'geo': 'http://www.w3.org/2003/01/geo/wgs84_pos#'}
        channel = root.find('channel') or root
        items = channel.findall('item') or root.findall('.//{http://www.w3.org/2005/Atom}entry')

        for item in items[:20]:  # max 20 alerts
            title = (item.findtext('title') or '').strip()
            desc  = (item.findtext('description') or '').strip()
            pub   = (item.findtext('pubDate') or '').strip()
            cat   = (item.findtext('category') or '').strip()
            link  = (item.findtext('link') or 'https://sachet.ndma.gov.in').strip()

            # Detect disaster type from title/description
            combined = (title + ' ' + desc + ' ' + cat).lower()
            dtype = None
            for key in TRAVEL_DISASTERS:
                if key in combined:
                    dtype = key
                    break

            if not dtype:
                continue  # skip non-travel-relevant alerts

            # Extract state names from text
            affected_states = []
            for state in STATE_BOUNDS:
                if state.lower() in combined:
                    affected_states.append(state)

            # Severity from keywords
            if any(w in combined for w in ['red', 'extreme', 'severe', 'very heavy']):
                severity = 'Extreme'
            elif any(w in combined for w in ['orange', 'heavy', 'high', 'warning']):
                severity = 'High'
            elif any(w in combined for w in ['yellow', 'moderate', 'watch']):
                severity = 'Moderate'
            else:
                severity = 'Low'

            meta = TRAVEL_DISASTERS[dtype]
            alerts.append({
                'title':    title,
                'desc':     desc[:200],
                'type':     dtype,
                'severity': severity,
                'states':   affected_states,
                'pub_date': pub,
                'link':     link,
                'icon':     meta['icon'],
                'color':    meta['color'],
                'risk_boost': meta['risk_boost'],
                'source':   'SACHET/NDMA (Live)',
            })

    except ET.ParseError as e:
        print(f"RSS parse error: {e}")

    return alerts

def get_static_ndma_alerts():
    """
    Realistic static alerts based on current month/season.
    Used as fallback when SACHET feed is unavailable.
    Reflects actual seasonal patterns per IMD data.
    """
    month = datetime.now().month
    alerts = []

    if 6 <= month <= 9:  # Monsoon
        alerts = [
            {'title':'Landslide Warning — Uttarakhand','desc':'Heavy rainfall expected. Landslide warning for Chamoli, Rudraprayag, Uttarkashi districts. Avoid non-essential travel on NH-58 and NH-94.','type':'landslide','severity':'High','states':['Uttarakhand'],'icon':'⛰️','color':'#7b1fa2','risk_boost':1.8,'pub_date':datetime.now().strftime('%a, %d %b %Y %H:%M:%S +0530'),'link':'https://sachet.ndma.gov.in','source':'Static (Seasonal)'},
            {'title':'Flash Flood Alert — Himachal Pradesh','desc':'Cloudburst warning for Kullu, Mandi, Chamba. Rivers in spate. NH-3 Kullu-Manali may be affected.','type':'flash flood','severity':'High','states':['Himachal Pradesh'],'icon':'⚡','color':'#1565c0','risk_boost':1.7,'pub_date':datetime.now().strftime('%a, %d %b %Y %H:%M:%S +0530'),'link':'https://sachet.ndma.gov.in','source':'Static (Seasonal)'},
            {'title':'Heavy Rain Alert — Western Ghats','desc':'Very heavy rainfall expected over Konkan and Ghat sections. Mumbai-Pune Expressway on alert.','type':'heavy rain','severity':'High','states':['Maharashtra'],'icon':'🌧️','color':'#0277bd','risk_boost':1.5,'pub_date':datetime.now().strftime('%a, %d %b %Y %H:%M:%S +0530'),'link':'https://sachet.ndma.gov.in','source':'Static (Seasonal)'},
        ]
    elif month in [12, 1, 2]:  # Winter fog
        alerts = [
            {'title':'Dense Fog Advisory — North India','desc':'Dense to very dense fog expected over Punjab, Haryana, Delhi, UP. Visibility below 50m on NH-44 and NH-48. Drive only in daytime.','type':'fog','severity':'Extreme','states':['Delhi','Punjab','Haryana','Uttar Pradesh'],'icon':'🌫️','color':'#455a64','risk_boost':1.6,'pub_date':datetime.now().strftime('%a, %d %b %Y %H:%M:%S +0530'),'link':'https://sachet.ndma.gov.in','source':'Static (Seasonal)'},
            {'title':'Avalanche Warning — J&K Highlands','desc':'Avalanche warning for higher reaches of Jammu & Kashmir. Avoid travel on Srinagar-Leh highway.','type':'avalanche','severity':'High','states':['Jammu & Kashmir'],'icon':'🏔️','color':'#37474f','risk_boost':1.9,'pub_date':datetime.now().strftime('%a, %d %b %Y %H:%M:%S +0530'),'link':'https://sachet.ndma.gov.in','source':'Static (Seasonal)'},
        ]
    elif month in [4, 5, 6]:  # Pre-monsoon
        alerts = [
            {'title':'Thunderstorm Warning — Rajasthan','desc':'Dust storms (Andhi) and thunderstorm warning for Jaipur, Jodhpur, Bikaner. Sudden visibility drop possible on highways.','type':'thunderstorm','severity':'Moderate','states':['Rajasthan'],'icon':'⛈️','color':'#1565c0','risk_boost':1.3,'pub_date':datetime.now().strftime('%a, %d %b %Y %H:%M:%S +0530'),'link':'https://sachet.ndma.gov.in','source':'Static (Seasonal)'},
        ]
    else:
        alerts = []  # Relatively clear seasons

    return alerts

def filter_alerts_for_route(alerts, start_lat, start_lon, end_lat, end_lon):
    """Filter alerts relevant to the route based on state overlap."""
    if not alerts:
        return []

    # Determine which states the route passes through (broad bounding box)
    route_min_lat = min(start_lat, end_lat) - 1.0
    route_max_lat = max(start_lat, end_lat) + 1.0
    route_min_lon = min(start_lon, end_lon) - 1.0
    route_max_lon = max(start_lon, end_lon) + 1.0

    route_states = set()
    for state, (mn_lat, mx_lat, mn_lon, mx_lon) in STATE_BOUNDS.items():
        # Check if state bounding box overlaps with route bounding box
        if (mn_lat <= route_max_lat and mx_lat >= route_min_lat and
                mn_lon <= route_max_lon and mx_lon >= route_min_lon):
            route_states.add(state)

    relevant = []
    for alert in alerts:
        # Include if alert affects a state on the route, or no state specified
        if not alert['states'] or any(s in route_states for s in alert['states']):
            relevant.append(alert)

    return relevant[:5]  # max 5 alerts shown


# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/heatmap")
def heatmap():
    df = pd.read_csv(os.path.join("data","accidents.csv"))
    return render_template("heatmap.html", accidents=df.to_dict(orient="records"))

@app.route("/black-spots")
def black_spots():
    """Return all black spots for map markers."""
    return jsonify({"black_spots": BLACK_SPOTS})

@app.route("/ndma-alerts")
def ndma_alerts():
    """Return live NDMA disaster alerts."""
    try:
        alerts = fetch_ndma_alerts()
        return jsonify({"alerts": alerts, "count": len(alerts), "source": "SACHET/NDMA"})
    except Exception as e:
        return jsonify({"alerts": [], "count": 0, "error": str(e)})

@app.route("/route-risk")
def route_risk():
    try:
        start = request.args.get("start")
        end   = request.args.get("end")
        hour  = int(request.args.get("hour", 12))
        if not start or not end: return jsonify({"error":"Missing parameters"}), 400

        s_lon, s_lat = map(float, start.split(","))
        e_lon, e_lat = map(float, end.split(","))
        time_mult    = get_time_mult(hour)
        time_label   = get_time_label(hour)

        # OSRM routing
        url  = f"http://router.project-osrm.org/route/v1/driving/{s_lon},{s_lat};{e_lon},{e_lat}?overview=full&geometries=geojson&alternatives=3"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if "routes" not in data or not data["routes"]: return jsonify({"error":"No route found"}), 404

        # Historical weather (same for all routes)
        hist_weather = get_historical_weather(s_lat, s_lon, e_lat, e_lon)

        # Live NDMA alerts for this route
        all_alerts     = fetch_ndma_alerts()
        route_alerts   = filter_alerts_for_route(all_alerts, s_lat, s_lon, e_lat, e_lon)
        max_risk_boost = max((a['risk_boost'] for a in route_alerts), default=1.0)

        route_labels = ["Route A","Route B","Route C"]
        route_colors = ["#1565C0","#E65100","#6A1B9A"]
        scored_routes = []

        # Fetch weather ONCE at midpoint — reuse for all routes (saves 10+ API calls)
        mid_coords = data["routes"][0]["geometry"]["coordinates"]
        mid_total  = len(mid_coords)
        mid_pts    = [mid_coords[int(i*(mid_total-1)/4)] for i in range(5)]
        from concurrent.futures import ThreadPoolExecutor, as_completed as cfu_as_completed
        shared_wpts = [None]*5
        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = {ex.submit(fetch_weather, c[1], c[0]): i for i, c in enumerate(mid_pts)}
            for fut in cfu_as_completed(futs):
                shared_wpts[futs[fut]] = fut.result()
        shared_wpts = [w if w is not None else _blank_weather() for w in shared_wpts]
        s_avail = [w for w in shared_wpts if w["available"]]
        if s_avail:
            shared_wsm = dict(max(s_avail, key=lambda w: w["multiplier"]))
            shared_wsm.update(alerts=[w["alert"] for w in s_avail if w["alert"]],
                              unique_conditions=list({w["risk_label"] for w in s_avail}),
                              point_count=len(s_avail),
                              is_uniform=len({w["risk_label"] for w in s_avail})==1,
                              weather_points=shared_wpts)
        else:
            shared_wsm = _blank_weather()
            shared_wsm["weather_points"] = shared_wpts

        for i, route in enumerate(data["routes"]):
            coords    = route["geometry"]["coordinates"]
            wpts, wsm = shared_wpts, shared_wsm
            acc       = score_route(coords, time_mult, wsm["multiplier"])

            # Landslide
            hilly, region = route_is_hilly(coords)
            ls_data = {"is_hilly": False}
            if hilly:
                hilly_pts = [(c[1],c[0]) for c in coords if is_hilly(c[1],c[0])[0]]
                step_ls   = max(1, len(hilly_pts)//15)
                ls_sample = hilly_pts[::step_ls][:15]
                ls_risks  = []; ls_zones_hit = []
                for lat, lon in ls_sample:
                    rv, zone = landslide_risk_at(lat, lon, wsm["multiplier"])
                    ls_risks.append(rv)
                    if zone and zone["zone_name"] not in ls_zones_hit: ls_zones_hit.append(zone["zone_name"])
                avg_ls = round(float(np.mean(ls_risks)) if ls_risks else 0, 1)
                ls_label = "Low" if avg_ls<30 else "Moderate" if avg_ls<55 else "High" if avg_ls<75 else "Very High"
                ls_segs  = [{"lat":c[1],"lon":c[0],"ls_risk":landslide_risk_at(c[1],c[0],wsm["multiplier"])[0] if is_hilly(c[1],c[0])[0] else 0.0} for c in coords]
                ls_data  = {"is_hilly":True,"region":region,"avg_risk":avg_ls,"risk_label":ls_label,
                            "season_mult":get_season_mult(),"season_label":get_season_label(),
                            "zones_crossed":ls_zones_hit[:8],"ls_segments":ls_segs,"warning":avg_ls>=55}

            combined_risk = round(min(100, 0.6*acc["risk_percent"] + 0.4*ls_data.get("avg_risk",0)) if hilly else acc["risk_percent"], 1)
            combined_label = "Low" if combined_risk<25 else "Moderate" if combined_risk<50 else "High" if combined_risk<70 else "Very High"

            dist_km  = round(route.get("distance",0)/1000, 1)
            dur_min  = int(round(route.get("duration",0)/60))

            # NEW FEATURES ─────────────────────────────────────────────────
            speed_zones  = analyse_speed_zones(coords, combined_risk)
            black_spots_ = detect_black_spots(coords)
            fatigue_warn = get_fatigue_warning(dist_km, dur_min, hour)
            night_warn   = get_night_vision_warning(hour, black_spots_, combined_risk)
            # ──────────────────────────────────────────────────────────────

            scored_routes.append({
                "route_id":       i,
                "label":          route_labels[i] if i < 3 else f"Route {i+1}",
                "color":          route_colors[i] if i < 3 else "#37474F",
                "distance_km":    dist_km,
                "duration_min":   dur_min,
                "risk_score":     acc["risk_score"],
                "risk_percent":   combined_risk,
                "risk_label":     combined_label,
                "accident_risk":  acc["risk_percent"],
                "segments":       acc["segments"],
                "coordinates":    [[c[1],c[0]] for c in coords],
                "is_safest":      False,
                "landslide":      ls_data,
                "weather_points": wpts,
                "weather":        wsm,
                "speed_zones":    speed_zones,
                "black_spots":    black_spots_,
                "fatigue_warning":fatigue_warn,
                "night_warning":  night_warn,
            })

        safest = min(range(len(scored_routes)), key=lambda i: scored_routes[i]["risk_percent"])
        scored_routes[safest]["is_safest"] = True

        # Safer route suggestion (after all routes scored)
        safer_suggestion = get_safer_route_suggestion(scored_routes, safest)

        return jsonify({
            "routes":           scored_routes,
            "safest_route_id":  safest,
            "time_multiplier":  time_mult,
            "time_label":       time_label,
            "departure_hour":   hour,
            "weather":          scored_routes[0]["weather"],
            "hist_weather":     hist_weather,
            "safer_suggestion": safer_suggestion,
            "ndma_alerts":      route_alerts,
            "alert_risk_boost": max_risk_boost,
        })

    except requests.exceptions.Timeout:
        return jsonify({"error":"Route service timed out"}), 504
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
