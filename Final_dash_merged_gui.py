import os
from datetime import datetime

import numpy as np
import pandas as pd
import panel as pn
import hvplot.pandas  # noqa: F401
import requests

pn.extension('tabulator', notifications=True)


# CONFIG
DATA_PATH = os.environ.get('WEATHER_DATA_PATH', 'DATA2.csv')
MET_DATA_PATH = os.environ.get('MET_DATA_PATH', 'Met data.csv')
SHEET_NAME = os.environ.get('WEATHER_SHEET_NAME', 'in')
LAT = 53.7960
LON = -1.7594
WEATHER_REFRESH_SECONDS = 300


# DATA LOADING / CLEANSING
MISSING_MARKERS = ['---', '--', '—', 'nan', 'NaN', '']

MONTH_NAME_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path!r}. Put the file next to this script or set WEATHER_DATA_PATH."
        )

    if path.lower().endswith('.csv'):
        df = pd.read_csv(path)
    elif path.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(path, sheet_name=SHEET_NAME)
    else:
        raise ValueError('Supported file types: .csv, .xlsx, .xls')

    df.columns = df.columns.astype(str).str.strip()

    if 'Date' not in df.columns or 'Time' not in df.columns:
        raise ValueError("The dataset must contain 'Date' and 'Time' columns.")

    # Robust date parsing
    date_only = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce').dt.date.astype('string')

    # More flexible time parsing for CSV data
    time_parsed = pd.to_datetime(df['Time'].astype(str).str.strip(), errors='coerce')
    time_only = time_parsed.dt.strftime('%H:%M:%S')

    df['datetime'] = pd.to_datetime(date_only + ' ' + time_only, errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid datetime rows could be parsed from the CSV. Check the Date and Time columns.")

    # Convert object columns that look numeric
    for col in df.columns:
        if col in {'Date', 'Time', 'datetime'}:
            continue
        if df[col].dtype == 'object':
            converted = pd.to_numeric(df[col].replace(MISSING_MARKERS, np.nan), errors='coerce')
            if converted.notna().sum() > 0:
                df[col] = converted

    # Helper columns
    df['date'] = df['datetime'].dt.date
    df['month'] = df['datetime'].dt.to_period('M').astype(str)
    df['month_name'] = df['datetime'].dt.month_name()
    df['month_num'] = df['datetime'].dt.month
    df['weekday'] = df['datetime'].dt.day_name()
    df['year'] = df['datetime'].dt.year

    return df


def load_met_data(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path!r}. Put the file next to this script or set MET_DATA_PATH."
        )

    met_df = pd.read_csv(path)
    met_df.columns = met_df.columns.astype(str).str.strip()

    month_cols = [c for c in met_df.columns if c.startswith('tas ')]
    if not month_cols:
        raise ValueError(
            "MET dataset must contain monthly temperature columns like 'tas January', 'tas February', etc."
        )

    for col in month_cols:
        met_df[col] = pd.to_numeric(met_df[col], errors='coerce')

    met_long = met_df.melt(
        id_vars=[c for c in met_df.columns if c not in month_cols],
        value_vars=month_cols,
        var_name='met_month_raw',
        value_name='met_temp'
    )

    met_long['month_name'] = met_long['met_month_raw'].str.replace('tas ', '', regex=False).str.strip()
    met_long['month_num'] = met_long['month_name'].map(MONTH_NAME_MAP)

    met_monthly = (
        met_long.dropna(subset=['month_num', 'met_temp'])
        .groupby(['month_num', 'month_name'], as_index=False)['met_temp']
        .mean()
        .sort_values('month_num')
    )

    return met_df, met_long, met_monthly


def build_local_monthly_temperature(df: pd.DataFrame, temp_col: str):
    if temp_col not in df.columns:
        raise ValueError(f"Temperature column {temp_col!r} not found in local dataset.")

    local_monthly = (
        df[['month_num', 'month_name', temp_col]]
        .dropna()
        .groupby(['month_num', 'month_name'], as_index=False)[temp_col]
        .mean()
        .rename(columns={temp_col: 'local_temp'})
        .sort_values('month_num')
    )

    return local_monthly


df = load_data(DATA_PATH)
met_df, met_long_df, met_monthly_df = load_met_data(MET_DATA_PATH)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    raise ValueError('No numeric columns found after cleaning.')

# USEFUL DEFAULTS
DEFAULT_METRIC = 'Temp_Out' if 'Temp_Out' in numeric_cols else numeric_cols[0]
DEFAULT_HUM = 'Out_Hum' if 'Out_Hum' in numeric_cols else numeric_cols[min(1, len(numeric_cols)-1)]
WIND_SPEED_COL = 'Wind_Speed' if 'Wind_Speed' in df.columns else None
WIND_DIR_COL = 'Wind_Dir' if 'Wind_Dir' in df.columns else None
TEMP_COLS = [c for c in numeric_cols if 'temp' in c.lower()] or numeric_cols
HUM_COLS = [c for c in numeric_cols if 'hum' in c.lower()] or numeric_cols
RAIN_COL = 'Rain' if 'Rain' in df.columns else None
SOLAR_COL = 'Solar_Rad' if 'Solar_Rad' in df.columns else None
UV_COL = 'UV_Index' if 'UV_Index' in df.columns else None
PRESSURE_COL = 'Bar' if 'Bar' in df.columns else None


# STYLING HELPERS
ACCENT = '#1f4e79'
CARD_STYLE = {
    'styles': {
        'background': 'white',
        'border-radius': '14px',
        'padding': '10px 14px',
        'box-shadow': '0 2px 8px rgba(0,0,0,0.08)',
        'border': '1px solid #e8edf2',
    }
}


def stat_card(title: str, value: str, note: str = ''):
    return pn.Column(
        pn.pane.Markdown(f"### {title}"),
        pn.pane.HTML(f"<div style='font-size:28px;font-weight:700;color:{ACCENT}'>{value}</div>"),
        pn.pane.Markdown(note) if note else pn.Spacer(height=0),
        sizing_mode='stretch_width',
        **CARD_STYLE,
    )


def latest_value(col: str, suffix: str = '') -> str:
    if col not in df.columns:
        return 'N/A'
    series = pd.to_numeric(df[col], errors='coerce').dropna()
    if series.empty:
        return 'N/A'
    return f"{series.iloc[-1]:.1f}{suffix}"


latest_cards = pn.Row(
    stat_card('🌡 Temperature', latest_value('Temp_Out', ' °C')),
    stat_card('💧 Humidity', latest_value('Out_Hum', ' %')),
    stat_card('🌬 Wind speed', latest_value('Wind_Speed', ' m/s')),
    stat_card('🧭 Pressure', latest_value('Bar', ' hPa')),
    stat_card('🌧 Rain', latest_value('Rain', ' mm') if RAIN_COL else 'N/A'),
    stat_card('☀️ Solar / UV', f"{latest_value('Solar_Rad')} / {latest_value('UV_Index')}"),
    sizing_mode='stretch_width'
)

dt_min = df['datetime'].min()
dt_max = df['datetime'].max()

date_range_text = "Unknown"
if pd.notna(dt_min) and pd.notna(dt_max):
    date_range_text = f"{dt_min:%d %b %Y %H:%M} → {dt_max:%d %b %Y %H:%M}"

summary_md = pn.pane.Markdown(
    f"""
## Weather Dashboard

**Main dataset:** `{os.path.basename(DATA_PATH)}`  
**Rows:** {len(df):,}  
**Date range:** {date_range_text}  

**MET dataset:** `{os.path.basename(MET_DATA_PATH)}`  
**MET rows:** {len(met_df):,}
""",
    sizing_mode='stretch_width'
)


# LIVE ALERTS (from polished dashboard)
alert_md = pn.pane.HTML("", sizing_mode='stretch_width')
local_live_md = pn.pane.Markdown("", sizing_mode='stretch_width')


def generate_alerts(row: pd.Series):
    alerts = []
    temp = row.get('Temp_Out')
    rain = row.get('Rain')
    wind = row.get('Wind_Speed')

    if pd.notna(temp):
        if temp >= 30:
            alerts.append("🔥 Heatwave Alert")
        elif temp <= 0:
            alerts.append("❄️ Freezing Alert")

    if pd.notna(rain) and rain > 0:
        alerts.append("🌧 Rain Alert")

    if pd.notna(wind) and wind > 10:
        alerts.append("🌬 Strong Wind Alert")

    if not alerts:
        alerts.append("✅ No active alerts")

    return alerts


def refresh_local_alerts():
    global df
    try:
        refreshed_df = load_data(DATA_PATH)
        if refreshed_df.empty:
            alert_md.object = "<b>⚠️ No data available</b>"
            local_live_md.object = "### Latest Reading\n\nNo local readings available."
            return

        df = refreshed_df
        latest = df.iloc[-1]
        alerts = generate_alerts(latest)

        alert_md.object = "<div style='padding:12px 14px;background:#fff7e6;border:1px solid #f5d48a;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.05)'>" + "<br>".join(alerts) + "</div>"
        local_live_md.object = f"""### Latest Local Reading

- **Time:** {latest.get('datetime')}
- **Temperature:** {latest.get('Temp_Out', 'N/A')} °C
- **Humidity:** {latest.get('Out_Hum', 'N/A')} %
- **Wind speed:** {latest.get('Wind_Speed', 'N/A')} m/s
- **Rain:** {latest.get('Rain', 'N/A')} mm
"""
    except Exception as e:
        alert_md.object = f"<div style='padding:12px 14px;background:#fdecea;border:1px solid #f5c2c7;border-radius:12px'>⚠️ Could not refresh alerts: {e}</div>"
        local_live_md.object = "### Latest Local Reading\n\nUnable to load local alert data."


refresh_local_alerts()
pn.state.add_periodic_callback(refresh_local_alerts, period=WEATHER_REFRESH_SECONDS * 1000, start=True)


# LIVE WEATHER TAB
live_md = pn.pane.Markdown('Loading live weather…', sizing_mode='stretch_width')


def fetch_live_weather(lat: float, lon: float) -> dict:
    url = (
        'https://api.open-meteo.com/v1/forecast'
        f'?latitude={lat}&longitude={lon}'
        '&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,pressure_msl'
        '&timezone=auto'
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    cur = r.json().get('current', {})
    return cur


def refresh_live_weather():
    try:
        w = fetch_live_weather(LAT, LON)
        live_md.object = f"""
### Live Weather (Open-Meteo)

- **Time:** {w.get('time')}
- **Temperature (°C):** {w.get('temperature_2m')}
- **Humidity (%):** {w.get('relative_humidity_2m')}
- **Wind speed (km/h):** {w.get('wind_speed_10m')}
- **Wind direction (°):** {w.get('wind_direction_10m')}
- **Pressure (hPa):** {w.get('pressure_msl')}
"""
    except Exception as e:
        live_md.object = f"### Live Weather\n\nCould not fetch live data.\n\n`{e}`"


refresh_live_weather()
pn.state.add_periodic_callback(refresh_live_weather, period=WEATHER_REFRESH_SECONDS * 1000, start=True)

real_time_tab = pn.Column(
    pn.pane.Markdown('## Live Weather'),
    live_md,
    pn.pane.Markdown('This is optional live data from Open-Meteo. The rest of the dashboard uses your local dataset.'),
    sizing_mode='stretch_width'
)

live_alerts_tab = pn.Column(
    pn.pane.Markdown('## Live Alerts'),
    alert_md,
    local_live_md,
    pn.pane.Markdown('These alerts are generated from the latest reading in your local weather dataset.'),
    sizing_mode='stretch_width'
)


# MAIN TIME SERIES EXPLORER
metric = pn.widgets.Select(name='Metric', options=sorted(numeric_cols), value=DEFAULT_METRIC)
resample = pn.widgets.Select(name='Resample', options=['None', '30min', '1h', '1D', '1W', '1M'], value='1h')
date_range = pn.widgets.DateRangeSlider(
    name='Date range',
    start=df['datetime'].min().date(),
    end=df['datetime'].max().date(),
    value=(max(df['datetime'].min().date(), (df['datetime'].max() - pd.Timedelta(days=30)).date()), df['datetime'].max().date()),
)
show_rolling = pn.widgets.Checkbox(name='Show 7-point rolling mean', value=False)


@pn.depends(metric, resample, date_range, show_rolling)
def ts_plot(metric, resample, date_range, show_rolling):
    start, end = date_range
    d = df[(df['datetime'].dt.date >= start) & (df['datetime'].dt.date <= end)].copy()
    if d.empty:
        return pn.pane.Markdown('No data in selected range.')

    d = d[['datetime', metric]].dropna().set_index('datetime')
    if resample != 'None':
        d = d.resample(resample).mean()

    plot = d.hvplot.line(
        height=450,
        min_width=1200,
        title=f'{metric} over time',
        ylabel=metric,
        line_width=2
    )

    if show_rolling and len(d) >= 7:
        rolling = d[metric].rolling(7, min_periods=1).mean().rename('rolling_mean')
        plot *= rolling.hvplot.line(color='orange', line_width=2, min_width=1200, height=450)

    return plot


time_series_tab = pn.Column(
    pn.Row(metric, resample, show_rolling),
    date_range,
    ts_plot,
    sizing_mode='stretch_width'
)


# OVERVIEW TAB
main_metric = pn.widgets.Select(
    name='Metric',
    options=[m for m in ['Temp_Out', 'Out_Hum', 'Wind_Speed', 'Bar', 'Rain', 'Solar_Rad', 'UV_Index'] if m in numeric_cols] or sorted(numeric_cols),
    value='Temp_Out' if 'Temp_Out' in numeric_cols else sorted(numeric_cols)[0]
)
window = pn.widgets.Select(
    name='Timeframe',
    options={'Last day': 48, 'Last 7 days': 336, 'Last month': 1440, 'Last year': 17520, 'All data': len(df)},
    value=min(336, len(df))
)


@pn.depends(main_metric, window)
def main_plot(main_metric, window):
    d = df[['datetime', main_metric]].dropna().tail(window).sort_values('datetime')
    if d.empty:
        return pn.pane.Markdown('No data for this metric.')

    return d.hvplot.line(
        x='datetime',
        y=main_metric,
        height=450,
        min_width=1200,
        title=f'{main_metric} ({len(d):,} points)',
        line_width=2
    )


overview_tab = pn.Column(
    summary_md,
    alert_md,
    latest_cards,
    pn.Row(main_metric, window),
    main_plot,
    sizing_mode='stretch_width'
)


# HUMIDITY VS TEMP TAB
temp_col = pn.widgets.Select(name='Temperature column', options=TEMP_COLS, value='Temp_Out' if 'Temp_Out' in TEMP_COLS else TEMP_COLS[0])
hum_col = pn.widgets.Select(name='Humidity column', options=HUM_COLS, value='Out_Hum' if 'Out_Hum' in HUM_COLS else HUM_COLS[0])


@pn.depends(temp_col, hum_col)
def scatter_th(tcol, hcol):
    d = df[[tcol, hcol]].apply(pd.to_numeric, errors='coerce').dropna()
    if d.empty:
        return pn.pane.Markdown('No data for this selection.')
    return d.hvplot.scatter(
        x=tcol,
        y=hcol,
        height=420,
        min_width=1200,
        alpha=0.4,
        title=f'{hcol} vs {tcol}'
    )


@pn.depends(temp_col, hum_col)
def corr_card(tcol, hcol):
    d = df[[tcol, hcol]].apply(pd.to_numeric, errors='coerce').dropna()
    if len(d) < 5:
        return pn.pane.Markdown('Not enough points for correlation.')
    corr = float(d.corr().iloc[0, 1])
    colour = '#188038' if corr > 0 else '#b3261e'
    return pn.pane.HTML(
        f"<div style='padding:16px;border-radius:14px;background:white;border:1px solid #e8edf2;box-shadow:0 2px 8px rgba(0,0,0,0.08)'>"
        f"<div style='font-size:14px;color:#5f6368'>Correlation</div>"
        f"<div style='font-size:30px;font-weight:700;color:{colour}'>{corr:.3f}</div></div>"
    )


hum_temp_tab = pn.Column(
    pn.Row(temp_col, hum_col, corr_card),
    scatter_th,
    sizing_mode='stretch_width'
)


# WIND TAB
def wind_layout():
    parts = [pn.pane.Markdown('## Wind Analysis')]

    if WIND_SPEED_COL:
        parts.append(
            df[[WIND_SPEED_COL]].dropna().hvplot.hist(
                bins=35,
                height=350,
                min_width=1200,
                title='Wind speed distribution'
            )
        )
    else:
        parts.append(pn.pane.Markdown('No wind speed column found.'))

    if WIND_DIR_COL:
        top = (
            df[WIND_DIR_COL]
            .dropna()
            .astype(str)
            .value_counts()
            .head(16)
            .rename_axis('direction')
            .reset_index(name='count')
        )
        parts.append(
            top.hvplot.bar(
                x='direction',
                y='count',
                height=350,
                min_width=1200,
                title='Wind direction counts (top 16)'
            )
        )
    else:
        parts.append(pn.pane.Markdown('No wind direction column found.'))

    return pn.Column(*parts, sizing_mode='stretch_width')


wind_tab = wind_layout()


# SEASONAL / MONTHLY SUMMARIES
summary_metric = pn.widgets.Select(name='Summary metric', options=sorted(numeric_cols), value=DEFAULT_METRIC)


@pn.depends(summary_metric)
def monthly_summary(summary_metric):
    d = df[['month', summary_metric]].dropna().copy()
    if d.empty:
        return pn.pane.Markdown('No monthly summary available.')
    s = d.groupby('month')[summary_metric].mean().reset_index()
    return s.hvplot.bar(
        x='month',
        y=summary_metric,
        rot=45,
        height=420,
        min_width=1200,
        title=f'Monthly average of {summary_metric}'
    )


@pn.depends(summary_metric)
def yearly_summary(summary_metric):
    d = df[['year', summary_metric]].dropna().copy()
    if d.empty:
        return pn.pane.Markdown('No yearly summary available.')
    s = d.groupby('year')[summary_metric].agg(['mean', 'min', 'max']).reset_index()
    return pn.Column(
        s.hvplot.bar(
            x='year',
            y='mean',
            height=320,
            min_width=1200,
            title=f'Yearly average of {summary_metric}'
        ),
        pn.widgets.Tabulator(s, pagination='remote', page_size=10, sizing_mode='stretch_width', height=220),
        sizing_mode='stretch_width'
    )


summaries_tab = pn.Column(
    pn.Row(summary_metric),
    monthly_summary,
    yearly_summary,
    sizing_mode='stretch_width'
)


# MET COMPARISON TAB
comparison_metric = pn.widgets.Select(
    name='Local temperature metric',
    options=TEMP_COLS,
    value='Temp_Out' if 'Temp_Out' in TEMP_COLS else TEMP_COLS[0]
)


@pn.depends(comparison_metric)
def build_comparison_df(metric_name):
    local_monthly = build_local_monthly_temperature(df, metric_name)

    comparison_df = pd.merge(
        local_monthly,
        met_monthly_df,
        on=['month_num', 'month_name'],
        how='inner'
    )

    comparison_df['difference'] = comparison_df['local_temp'] - comparison_df['met_temp']
    comparison_df['abs_difference'] = comparison_df['difference'].abs()
    comparison_df['month_name'] = pd.Categorical(
        comparison_df['month_name'],
        categories=MONTH_ORDER,
        ordered=True
    )
    comparison_df = comparison_df.sort_values('month_num').reset_index(drop=True)
    return comparison_df


@pn.depends(comparison_metric)
def comparison_cards(metric_name):
    d = build_comparison_df(metric_name)
    if d.empty:
        return pn.pane.Markdown('No comparison data available.')

    annual_local = d['local_temp'].mean()
    annual_met = d['met_temp'].mean()
    avg_diff = d['difference'].mean()

    max_warmer = d.loc[d['difference'].idxmax()]
    max_cooler = d.loc[d['difference'].idxmin()]

    return pn.Row(
        stat_card('Local Avg', f'{annual_local:.2f} °C', f'{metric_name} monthly mean'),
        stat_card('MET Avg', f'{annual_met:.2f} °C', 'Monthly mean from MET dataset'),
        stat_card('Avg Difference', f'{avg_diff:.2f} °C', 'Local minus MET'),
        stat_card('Warmest Gap', f"{max_warmer['difference']:.2f} °C", f"{max_warmer['month_name']}"),
        stat_card('Coolest Gap', f"{max_cooler['difference']:.2f} °C", f"{max_cooler['month_name']}"),
        sizing_mode='stretch_width'
    )


@pn.depends(comparison_metric)
def comparison_plot(metric_name):
    d = build_comparison_df(metric_name)
    if d.empty:
        return pn.pane.Markdown('No comparison data available.')

    local_line = d.hvplot.line(
        x='month_name',
        y='local_temp',
        height=430,
        min_width=1200,
        title=f'Monthly Comparison: {metric_name} vs MET Monthly Temperature',
        ylabel='Temperature (°C)',
        line_width=3,
        legend='top_left',
        label='Local dataset'
    )

    met_line = d.hvplot.line(
        x='month_name',
        y='met_temp',
        height=430,
        min_width=1200,
        line_width=3,
        label='MET dataset'
    )

    return local_line * met_line


@pn.depends(comparison_metric)
def comparison_difference_plot(metric_name):
    d = build_comparison_df(metric_name)
    if d.empty:
        return pn.pane.Markdown('No comparison data available.')

    return d.hvplot.bar(
        x='month_name',
        y='difference',
        height=350,
        min_width=1200,
        title=f'Monthly Difference: {metric_name} - MET Temperature',
        ylabel='Difference (°C)',
        rot=45
    )


@pn.depends(comparison_metric)
def comparison_table(metric_name):
    d = build_comparison_df(metric_name).copy()
    if d.empty:
        return pn.pane.Markdown('No comparison data available.')

    table_df = d[['month_name', 'local_temp', 'met_temp', 'difference', 'abs_difference']].copy()
    table_df.columns = ['Month', 'Local Avg Temp', 'MET Avg Temp', 'Difference', 'Absolute Difference']

    return pn.widgets.Tabulator(
        table_df.round(3),
        pagination='remote',
        page_size=12,
        sizing_mode='stretch_width',
        height=320
    )


met_raw_table = pn.widgets.Tabulator(
    met_df.head(500),
    pagination='remote',
    page_size=20,
    sizing_mode='stretch_width',
    height=360
)

comparison_tab = pn.Column(
    pn.pane.Markdown(
        """
## MET Comparison

This tab compares the monthly average temperature from your local weather dataset
with the monthly average temperature from the MET dataset.

- **Local dataset** = monthly mean from your uploaded weather observations
- **MET dataset** = monthly mean across all rows in `Met data.csv`
- **Difference** = Local minus MET
"""
    ),
    pn.Row(comparison_metric),
    comparison_cards,
    comparison_plot,
    comparison_difference_plot,
    pn.pane.Markdown("### Monthly Comparison Table"),
    comparison_table,
    pn.pane.Markdown("### MET Raw Data Preview"),
    met_raw_table,
    sizing_mode='stretch_width'
)


# DATA TABLE
table = pn.widgets.Tabulator(df, pagination='remote', page_size=20, sizing_mode='stretch_width', height=520)
data_table_tab = pn.Column(
    pn.pane.Markdown('## Raw Data'),
    table,
    sizing_mode='stretch_width'
)

# APP
app = pn.Column(
    pn.pane.Markdown("# Weather Dashboard"),
    pn.Tabs(
        ("Overview", overview_tab),
        ("Live Alerts", live_alerts_tab),
        ("Live Weather", real_time_tab),
        ("Time Series", time_series_tab),
        ("Correlation Analysis", hum_temp_tab),
        ("Wind", wind_tab),
        ("Monthly & Yearly Trends", summaries_tab),
        ("MET Comparison", comparison_tab),
        ("Raw Data", data_table_tab),
        dynamic=False,
        sizing_mode="stretch_width",
    ),
    sizing_mode="stretch_width",
)

app.servable()