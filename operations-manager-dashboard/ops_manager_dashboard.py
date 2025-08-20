import os
import math
from datetime import datetime, date, time as dtime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# =============================
# CONFIG & ENV
# =============================
load_dotenv()
st.set_page_config(page_title="Operations Manager — Daily Execution", layout="wide")
TZ = os.getenv("APP_TIMEZONE", "Africa/Tunis")
NOW = datetime.now(ZoneInfo(TZ))
TODAY = NOW.date()

# DB creds
DB_DIALECT = os.getenv("DB_DIALECT") or st.secrets.get("DB_DIALECT", "postgresql")
DB_HOST = os.getenv("DB_HOST") or st.secrets.get("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT") or st.secrets.get("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME") or st.secrets.get("DB_NAME", "")
DB_USER = os.getenv("DB_USER") or st.secrets.get("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD") or st.secrets.get("DB_PASSWORD", "")

# =============================
# HELPERS
# =============================
@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    if DB_DIALECT.startswith("postgres"):
        url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    else:
        url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url, pool_pre_ping=True)

def day_bounds(day: date) -> Tuple[datetime, datetime]:
    start = datetime.combine(day, dtime.min).astimezone(ZoneInfo(TZ))
    end = start + timedelta(days=1)
    return start, end

DAY_START, DAY_END = day_bounds(TODAY)

# =============================
# SIDEBAR FILTERS
# =============================
st.title("🚌 Operations Manager Dashboard → Daily Execution")

with st.sidebar:
    st.header("⚙️ Controls")
    day = st.date_input("Day", value=TODAY)
    DAY_START, DAY_END = day_bounds(day)
    ontime_tol_min = st.number_input("On-time tolerance (min)", 0, 120, 10)
    cpi_weight_on_time = st.slider("CPI Weight: On-Time", 0.0, 1.0, 0.6, 0.05)
    cpi_weight_att = st.slider("CPI Weight: Attendance", 0.0, 1.0, 0.4, 0.05)
    mtbf_lookback_days = st.number_input("MTBF Lookback (days)", 7, 120, 30)
    shift_start_default = st.time_input("Default shift start", value=dtime(8, 0))
    autorefresh_sec = st.number_input("Auto-refresh (sec)", 0, 300, 30, help="0 = off")
    st.caption(f"Timezone: {TZ}")

if autorefresh_sec:
    st.cache_data.clear()   # no-op for fresh feel
    st.autorefresh = st.experimental_rerun  # compatibility alias
    st.experimental_set_query_params(ts=int(NOW.timestamp()))

engine = get_engine()

# Small SQL helpers (Postgres/MySQL compatible where possible)
q_between = lambda col: f"{col} >= :start AND {col} < :end"

# =============================
# DATA LOAD — EXACT QUERIES
# =============================
@st.cache_data(show_spinner=False)
def load_trip_pairs(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Join planned (trip_timings) to actuals (trip_durations) by trip_id for the selected day.
    Planned columns are TIME ONLY; we align them to the same date as actual start when available,
    otherwise to selected day."""
    sql = text(
        f"""
        WITH actuals AS (
            SELECT id, trip_id, start AS actual_departure, "end" AS actual_arrival,
                   status, expected_duration, real_duration, created_at
            FROM trip_durations
            WHERE {q_between('start')}
        ), planned AS (
            SELECT trip_id, "start" AS planned_dep_time, "end" AS planned_arr_time
            FROM trip_timings
        )
        SELECT a.*, p.planned_dep_time, p.planned_arr_time
        FROM actuals a
        LEFT JOIN planned p USING (trip_id)
        ORDER BY a.start
        """
    )
    df = pd.read_sql(sql, engine, params={"start": start_dt, "end": end_dt})
    # Build planned timestamps by combining date with time
    for col in ["planned_dep_time", "planned_arr_time"]:
        if col in df.columns:
            # Prefer the date of actual_departure; fallback to selected day
            base_dates = pd.to_datetime(df["actual_departure"]).dt.date.fillna(day)
            df[col] = pd.to_datetime(base_dates.astype(str) + " " + df[col].astype(str), errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_bus_trackings(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    sql = text(
        f"""
        SELECT id, created_at, updated_at, type, stop_id, time, trip
        FROM bus_trackings
        WHERE {q_between('time')}
        ORDER BY time DESC
        """
    )
    return pd.read_sql(sql, engine, params={"start": start_dt, "end": end_dt})

@st.cache_data(show_spinner=False)
def load_vehicle_events(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    sql = text(
        f"""
        SELECT id, created_at, updated_at, type, data, driver_id, trip, vehicle_id
        FROM vehicle_events
        WHERE {q_between('created_at')}
        ORDER BY created_at DESC
        """
    )
    return pd.read_sql(sql, engine, params={"start": start_dt, "end": end_dt})

@st.cache_data(show_spinner=False)
def load_scheduled_maint(start_dt: datetime, end_dt: datetime, lookback_days: int) -> pd.DataFrame:
    # For downtime (time_from/time_to) and MTBF proxy via engine_working_hours deltas
    lookback_start = start_dt - timedelta(days=lookback_days)
    sql = text(
        f"""
        SELECT id, date, time, vehicle_id, engine_working_hours, maintenance_location,
               manager_confirmation, technical_manager_confirmation, created_at, updated_at,
               name, time_from, time_to, bus_status, agent_id, duration
        FROM scheduled_maintenances
        WHERE (time_from IS NOT NULL AND time_to IS NOT NULL AND time_from >= :lb_start)
           OR (created_at >= :lb_start)
        ORDER BY COALESCE(time_from, created_at) DESC
        """
    )
    return pd.read_sql(sql, engine, params={"lb_start": lookback_start})

@st.cache_data(show_spinner=False)
def load_vehicles() -> pd.DataFrame:
    sql = text("SELECT id, registration_number, status, fleet_type_id FROM vehicles")
    return pd.read_sql(sql, engine)

@st.cache_data(show_spinner=False)
def load_attendances(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    sql = text(
        f"""
        SELECT id, user_id, date, start_time, end_time, created_at
        FROM attendances
        WHERE {q_between('created_at')}
        ORDER BY created_at
        """
    )
    return pd.read_sql(sql, engine, params={"start": start_dt, "end": end_dt})

@st.cache_data(show_spinner=False)
def load_users() -> pd.DataFrame:
    sql = text("SELECT id, firstname, lastname, position, status FROM users")
    return pd.read_sql(sql, engine)

# Load all
trips = load_trip_pairs(DAY_START, DAY_END)
track = load_bus_trackings(DAY_START, DAY_END)
veh_events = load_vehicle_events(DAY_START, DAY_END)
smaint = load_scheduled_maint(DAY_START, DAY_END, int(mtbf_lookback_days))
vehicles = load_vehicles()
att = load_attendances(DAY_START, DAY_END)
users = load_users()

# =============================
# KPI: On-Time Departure & Arrival %
# =============================
st.subheader("⏱️ On-Time Departure & Arrival %")

otp_dep = otp_arr = None
if not trips.empty:
    tol = pd.Timedelta(minutes=int(ontime_tol_min))
    # Parse times
    for c in ["actual_departure", "actual_arrival", "planned_dep_time", "planned_arr_time"]:
        if c in trips:
            trips[c] = pd.to_datetime(trips[c], errors="coerce")
    if {"planned_dep_time", "actual_departure"}.issubset(trips.columns):
        trips["dep_on_time"] = (trips["actual_departure"] - trips["planned_dep_time"]).abs() <= tol
        otp_dep = trips["dep_on_time"].mean()
    if {"planned_arr_time", "actual_arrival"}.issubset(trips.columns):
        trips["arr_on_time"] = (trips["actual_arrival"] - trips["planned_arr_time"]).abs() <= tol
        otp_arr = trips["arr_on_time"].mean()

k1, k2, k3 = st.columns([1,1,2])
with k1:
    st.metric("On-Time Departure %", f"{otp_dep*100:.1f}%" if otp_dep is not None else "N/A")
with k2:
    st.metric("On-Time Arrival %", f"{otp_arr*100:.1f}%" if otp_arr is not None else "N/A")
with k3:
    if not trips.empty and ("dep_on_time" in trips or "arr_on_time" in trips):
        dfh = pd.DataFrame({
            "metric": (['Departure']*len(trips)) + (['Arrival']*len(trips)) if "arr_on_time" in trips else ['Departure']*len(trips),
            "on_time": list(trips.get("dep_on_time", pd.Series(dtype=bool))) + (list(trips.get("arr_on_time", pd.Series(dtype=bool))) if "arr_on_time" in trips else [])
        })
        fig = px.histogram(dfh, x="on_time", color="metric", barmode="group", title="On-time vs Late")
        st.plotly_chart(fig, use_container_width=True)

# =============================
# KPI: Trip Completion & Cancellation Breakdown
# =============================
st.subheader("🧭 Trip Completion & Cancellation Breakdown")
status_break = pd.DataFrame()
if "status" in trips:
    status_break = trips["status"].fillna("unknown").str.lower().value_counts().reset_index()
    status_break.columns = ["status", "count"]
    fig = px.pie(status_break, names="status", values="count", title="Trip Outcomes Today")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No status field available in today's trips.")

# =============================
# KPI: Fleet Downtime & MTBF
# =============================
st.subheader("🧰 Fleet Downtime & MTBF")

# Downtime from scheduled_maintenances (time_from/time_to or duration minutes)
DT = pd.DataFrame()
if not smaint.empty:
    for c in ["time_from", "time_to", "created_at", "updated_at"]:
        if c in smaint:
            smaint[c] = pd.to_datetime(smaint[c], errors="coerce")
    DT = smaint.copy()
    if {"time_from", "time_to"}.issubset(DT.columns):
        DT["downtime_hours"] = (DT["time_to"] - DT["time_from"]).dt.total_seconds()/3600.0
    elif "duration" in DT:
        DT["downtime_hours"] = DT["duration"].astype(float)/60.0
    else:
        DT["downtime_hours"] = np.nan

    dt_agg = DT.groupby("vehicle_id", dropna=False)["downtime_hours"].sum().reset_index()
    fig = px.bar(dt_agg.sort_values("downtime_hours", ascending=False).head(20), x="vehicle_id", y="downtime_hours", title="Total Downtime (hrs) — Lookback")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No scheduled maintenance rows in the lookback window.")

# MTBF: failures / operating hours proxy
MTBF = pd.DataFrame()
if not veh_events.empty:
    fe = veh_events.copy()
    fe["is_failure"] = fe["type"].str.contains("breakdown|failure|fault", case=False, na=False)
    failures = fe.groupby("vehicle_id", dropna=False)["is_failure"].sum().astype(int).reset_index(name="failures")

    # Operating hours proxy: engine_working_hours delta per vehicle in lookback
    op = pd.DataFrame(columns=["vehicle_id","operating_hours"])
    if not smaint.empty and "engine_working_hours" in smaint:
        # take min and max per vehicle
        g = smaint.dropna(subset=["vehicle_id"]).sort_values(["vehicle_id","created_at"]).groupby("vehicle_id")
        op = pd.DataFrame({
            "vehicle_id": list(g.groups.keys()),
            "operating_hours": (g["engine_working_hours"].last() - g["engine_working_hours"].first()).fillna(0).values
        })
        # if hours are in integer hours already; if they are minutes, adjust here

    MTBF = pd.merge(op, failures, on="vehicle_id", how="outer").fillna({"operating_hours": 0, "failures": 0})
    MTBF["MTBF_hours"] = MTBF.apply(lambda r: (r["operating_hours"] / r["failures"]) if r["failures"]>0 else np.nan, axis=1)

    fig = px.scatter(MTBF, x="operating_hours", y="MTBF_hours", hover_data=["vehicle_id","failures"], title="MTBF vs Operating Hours (proxy)")
    st.plotly_chart(fig, use_container_width=True)

# =============================
# Staff Trip Punctuality & Readiness Score
# =============================
st.subheader("👷 Staff Trip Punctuality & Readiness")

staff_score = None
if not att.empty:
    # On-time if created_at <= shift_start_default + tol
    att["created_at"] = pd.to_datetime(att["created_at"], errors="coerce")
    tol = pd.Timedelta(minutes=int(ontime_tol_min))
    shift_dt = pd.to_datetime(att["created_at"].dt.date.astype(str) + " " + shift_start_default.strftime("%H:%M:%S"))
    att["on_time"] = att["created_at"] <= (shift_dt + tol)
    attendance_rate = att["user_id"].nunique() / max(users["id"].nunique(), 1)
    on_time_rate = att["on_time"].mean()
    staff_score = cpi_weight_on_time * on_time_rate + cpi_weight_att * attendance_rate

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Attendance Rate", f"{attendance_rate*100:.1f}%")
        st.metric("On-Time Check-in Rate", f"{on_time_rate*100:.1f}%")
    with col2:
        st.metric("Readiness Score", f"{staff_score*100:.1f}%")
        fig = px.histogram(att, x="on_time", title="On-Time vs Late Check-ins")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No attendances today.")

# =============================
# Route-level Journey Time Trends (by trip_id)
# =============================
st.subheader("🗺️ Journey Time Trends (by trip_id)")
if not trips.empty:
    jt = trips.copy()
    # Prefer real_duration; fallback to difference
    if "real_duration" in jt and jt["real_duration"].notna().any():
        jt["real_minutes"] = jt["real_duration"].astype(float)
    elif {"actual_departure","actual_arrival"}.issubset(jt.columns):
        jt["real_minutes"] = (pd.to_datetime(jt["actual_arrival"]) - pd.to_datetime(jt["actual_departure"])) .dt.total_seconds()/60.0
    else:
        jt["real_minutes"] = np.nan

    if "expected_duration" in jt:
        jt["expected_minutes"] = jt["expected_duration"].astype(float)

    fig = px.line(jt.sort_values("actual_departure"), x="actual_departure", y=[c for c in ["real_minutes","expected_minutes"] if c in jt], color_discrete_sequence=None, title="Expected vs Real Duration over time")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No trip records for the selected day.")

# =============================
# Crew Performance Index (attendance-based)
# =============================
st.subheader("🧑‍✈️ Crew Performance Index")
CPI = None
if not att.empty:
    CPI = staff_score  # Same computed above; rename for clarity
    st.metric("CPI", f"{(CPI or 0)*100:.1f}%")
else:
    st.info("CPI requires attendance data.")

# =============================
# Real-time Trip Dashboard (color-coded)
# =============================
st.subheader("📡 Real-time Trip Dashboard")
if not track.empty:
    latest = track.sort_values("time").groupby("trip").tail(1)
    # Derive delay flag: if we have planned_dep_time vs now
    dash = latest.merge(trips[["trip_id","planned_dep_time","planned_arr_time","actual_departure","actual_arrival","status"]], left_on="trip", right_on="trip_id", how="left")
    tol = pd.Timedelta(minutes=int(ontime_tol_min))
    dash["dep_delay"] = (NOW - pd.to_datetime(dash["planned_dep_time"])) if "planned_dep_time" in dash else pd.NaT
    def color_row(r):
        try:
            if pd.isna(r["planned_dep_time"]):
                return "gray"
            if pd.isna(r["actual_departure"]) and NOW > r["planned_dep_time"] + tol:
                return "red"   # late departure
            if pd.notna(r.get("actual_arrival")) and pd.notna(r.get("planned_arr_time")):
                return "green" if abs((r["actual_arrival"] - r["planned_arr_time"]).total_seconds()) <= tol.total_seconds() else "orange"
            return "blue"      # in-progress / unknown
        except Exception:
            return "gray"
    dash["status_color"] = dash.apply(color_row, axis=1)
    st.dataframe(dash[["trip","time","type","stop_id","status","status_color"]].rename(columns={"time":"last_event"}), use_container_width=True)
else:
    st.info("No bus tracking events for the selected day.")

# =============================
# Delay Root Cause Pie Chart
# =============================
st.subheader("🥧 Delay Root Cause")
if not veh_events.empty:
    causes = veh_events["type"].fillna("unknown").str.lower()
    # Map common buckets
    buckets = causes.replace({
        "breakdown":"vehicle issue",
        "failure":"vehicle issue",
        "fault":"vehicle issue",
    }, regex=True)
    pie = buckets.value_counts().reset_index()
    pie.columns = ["cause","count"]
    fig = px.pie(pie, names="cause", values="count", title="Event Types (proxy for delay causes)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No vehicle events today.")

# =============================
# Vehicle Reliability Scorecard
# =============================
st.subheader("🚍 Vehicle Reliability Scorecard")
if not MTBF.empty:
    score = MTBF.merge(vehicles, left_on="vehicle_id", right_on="id", how="left")
    score["reliability_score"] = np.clip((score["MTBF_hours"].fillna(0) / (score["MTBF_hours"].fillna(0).quantile(0.9) + 1e-6))*100, 0, 100)
    st.dataframe(score[["vehicle_id","registration_number","operating_hours","failures","MTBF_hours","reliability_score"]].sort_values("reliability_score", ascending=False), use_container_width=True)
else:
    st.info("MTBF not available to build a scorecard.")

# =============================
# Radar Chart: Driver performance by route (trip)
# =============================
st.subheader("📈 Driver Performance (Radar by trip)")
# Build a (driver_id, trip_id) mapping from vehicle_events (trip, driver_id)
radar_df = pd.DataFrame()
if not veh_events.empty and not trips.empty and {"driver_id","trip"}.issubset(veh_events.columns):
    m = veh_events.dropna(subset=["driver_id","trip"]).groupby(["driver_id","trip"]).size().reset_index(name="events")
    m = m.merge(trips, left_on="trip", right_on="trip_id", how="left")
    m["on_time"] = ((pd.to_datetime(m.get("actual_departure")) - pd.to_datetime(m.get("planned_dep_time"))).abs() <= pd.Timedelta(minutes=int(ontime_tol_min)))
    perf = m.groupby(["driver_id","trip"]).agg(on_time_rate=("on_time","mean")).reset_index()
    # Pivot for radar: driver rows, trip columns (limit top N trips for readability)
    top_trips = perf["trip"].value_counts().head(6).index
    radar = perf[perf["trip"].isin(top_trips)].pivot_table(index="driver_id", columns="trip", values="on_time_rate", aggfunc="mean").fillna(0)
    radar_df = radar.reset_index()
    if not radar.empty:
        # Plotly doesn't have native radar multi-series in express, use Graph Objects
        fig = go.Figure()
        categories = radar.columns.tolist()
        for _, row in radar.iterrows():
            fig.add_trace(go.Scatterpolar(r=row.values.tolist(), theta=categories, fill='toself', name=f"Driver {row.name}"))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True, title="Driver On-Time Rate by Trip")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Insufficient data to compute driver radar (need vehicle_events with driver_id & trip, and trips).")

# =============================
# Quick Actions / Shortcuts (DB writes)
# =============================
st.subheader("⚡ Quick Actions")

# Helper to run a write safely
def run_write(sql: str, params: dict) -> Optional[str]:
    try:
        with engine.begin() as conn:
            conn.execute(text(sql), params)
        return None
    except SQLAlchemyError as e:
        return str(e)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("**🚦 Flag delayed trip**")
    qa_trip = st.text_input("Trip ID", key="qa_trip_flag")
    if st.button("Flag for review") and qa_trip:
        err = run_write(
            "INSERT INTO vehicle_events (type, trip, created_at) VALUES (:t, :trip, NOW())",
            {"t": "delay_flag", "trip": qa_trip}
        )
        st.success("Flag inserted") if not err else st.error(err)

with c2:
    st.markdown("**🧑 Assign new driver**")
    qa_trip2 = st.text_input("Trip ID", key="qa_trip_assign")
    qa_driver = st.number_input("Driver User ID", min_value=1, step=1)
    if st.button("Assign") and qa_trip2 and qa_driver:
        err = run_write(
            "INSERT INTO vehicle_events (type, trip, driver_id, created_at) VALUES (:t, :trip, :driver, NOW())",
            {"t": "assign_driver", "trip": qa_trip2, "driver": int(qa_driver)}
        )
        st.success("Assignment recorded") if not err else st.error(err)

with c3:
    st.markdown("**🛠 Request maintenance**")
    qa_vehicle = st.number_input("Vehicle ID", min_value=1, step=1)
    qa_dur = st.number_input("Estimated duration (min)", min_value=0, step=15)
    if st.button("Create request") and qa_vehicle:
        err = run_write(
            """
            INSERT INTO scheduled_maintenances (vehicle_id, created_at, name, duration, bus_status)
            VALUES (:vid, NOW(), :name, :dur, :status)
            """,
            {"vid": int(qa_vehicle), "name":"Requested via dashboard", "dur": int(qa_dur), "status": "requested"}
        )
        st.success("Maintenance request inserted") if not err else st.error(err)

with c4:
    st.markdown("**🗓 Approve trip confirmation**")
    qa_trip3 = st.text_input("Trip ID", key="qa_trip_confirm")
    if st.button("Approve") and qa_trip3:
        err = run_write(
            "UPDATE trip_durations SET status = :st WHERE trip_id = :trip",
            {"st": "confirmed", "trip": qa_trip3}
        )
        st.success("Trip status updated") if not err else st.error(err)

st.divider()

# =============================
# Raw debug views
# =============================
st.subheader("🔎 Raw (today)")
preview = st.multiselect("Tables", ["trips","track","veh_events","smaint","vehicles","att","users"], [])
lookup = {"trips":trips, "track":track, "veh_events":veh_events, "smaint":smaint, "vehicles":vehicles, "att":att, "users":users}
for k in preview:
    st.write(f"**{k}** — {len(lookup[k]):,} rows")
    st.dataframe(lookup[k].head(500), use_container_width=True)

