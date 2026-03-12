import os
from datetime import datetime

import numpy as np
import pandas as pd
import panel as pn
import hvplot.pandas  # noqa: F401
import requests

pn.extension('tabulator', notifications=True)


# CONFIG
DATA_PATH = os.environ.get('WEATHER_DATA_PATH', 'DATA2(in).csv')
SHEET_NAME = os.environ.get('WEATHER_SHEET_NAME', 'in')
LAT = 53.7960
LON = -1.7594
WEATHER_REFRESH_SECONDS = 300


# DATA LOADING / CLEANSING
MISSING_MARKERS = ['---', '--', '—', 'nan', 'NaN', '']


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
    df['weekday'] = df['datetime'].dt.day_name()
    df['year'] = df['datetime'].dt.year

    return df


df = load_data(DATA_PATH)
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

**Dataset:** `{os.path.basename(DATA_PATH)}`  
**Rows:** {len(df):,}  
**Date range:** {date_range_text}
""",
    sizing_mode='stretch_width'
)


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


# MAIN TIME SERIES EXPLORER
metric = pn.widgets.Select(name='Metric', options=sorted(numeric_cols), value=DEFAULT_METRIC)
resample = pn.widgets.Select(name='Resample', options=['None', '30min', '1h', '1D', '1W', '1M'], value='1h')
date_range = pn.widgets.DateRangeSlider(
    name='Date range',
    start=df['datetime'].min().date(),
    end=df['datetime'].max().date(),
    value=(df['datetime'].max().date() - pd.Timedelta(days=30), df['datetime'].max().date()),
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
    value=336
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
        ("Live Weather", real_time_tab),
        ("Time Series", time_series_tab),
        ("Correlation Analysis", hum_temp_tab),
        ("Wind", wind_tab),
        ("Monthly & Yearly Trends", summaries_tab),
        ("Raw Data", data_table_tab),
        dynamic=False,
        sizing_mode="stretch_width",
    ),
    sizing_mode="stretch_width",
)

app.servable()
