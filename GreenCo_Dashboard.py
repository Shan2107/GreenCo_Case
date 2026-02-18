#!/usr/bin/env python3
"""
================================================================================
  GreenCo E5 Dashboard Generator
  Generates the 3 DAM Price Volatility Exposure visuals from the case study
  Based on SAPP Market Data Excel file
================================================================================

USAGE:
  python greenco_e5_dashboard.py

  When prompted, enter the path to your Excel file:
    e.g.  C:\\Users\\You\\Downloads\\Data_Analyst_SAPP_Market_Extract__1___1_.xlsx
    or    /home/you/data/Data_Analyst_SAPP_Market_Extract__1___1_.xlsx

  Then choose which quarter to analyse (or use the default SAPP Q4).

REQUIREMENTS:
  pip install pandas openpyxl matplotlib seaborn numpy

OUTPUTS (saved to same folder as this script):
  visual1_revenue_at_risk.png
  visual2_volatility_heatmap.png
  visual3_rolling_correlation.png
================================================================================
"""

import sys
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
warnings.filterwarnings('ignore')


# ── COLOUR PALETTE (dark theme) ────────────────────────────────────────────
DARK_BG  = '#0d1117'
PANEL_BG = '#161b22'
GRID_CLR = '#21262d'
TEXT_CLR = '#e6edf3'
MUTED    = '#8b949e'
BLUE     = '#58a6ff'
GREEN    = '#3fb950'
RED      = '#f85149'
AMBER    = '#d29922'
PURPLE   = '#bc8cff'

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor':   PANEL_BG,
    'axes.edgecolor':   GRID_CLR,
    'axes.labelcolor':  TEXT_CLR,
    'xtick.color':      MUTED,
    'ytick.color':      MUTED,
    'text.color':       TEXT_CLR,
    'grid.color':       GRID_CLR,
    'grid.linewidth':   0.6,
    'font.family':      'monospace',
})


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD & CLEAN DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load and clean the SAPP Market Data Excel file.
    Returns a cleaned DataFrame with proper column names and data types.
    """
    print(f"\n  Loading: {filepath}")
    df = pd.read_excel(filepath, sheet_name='Market data', skiprows=1)

    df.columns = [
        'Date', 'Year', 'Month', 'Hour', 'ToU_Period',
        'DAM_Price', 'DAM_Volume',
        'Monthly_Price', 'Monthly_Volume',
        'Weekly_Price',  'Weekly_Volume',
        'IntraDay_Price','IntraDay_Volume',
    ]

    df['Date']  = pd.to_datetime(df['Date'], errors='coerce')
    df['Hour']  = pd.to_numeric(df['Hour'],  errors='coerce')

    for col in ['DAM_Price','DAM_Volume','Monthly_Price','Monthly_Volume',
                'Weekly_Price','Weekly_Volume','IntraDay_Price','IntraDay_Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['Date','Hour','DAM_Price'], inplace=True)
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['Date_Only'] = df['Date'].dt.date

    print(f"  ✓ {len(df):,} records loaded  "
          f"({df['Date'].min().date()} → {df['Date'].max().date()})")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — PV GENERATION PROFILE
# ══════════════════════════════════════════════════════════════════════════════

def pv_weight(hour: int) -> float:
    """
    Normalised PV generation weight for a given hour.
    Bell curve centred at solar noon (12:00), zero outside 06–18.
    """
    if hour < 6 or hour > 18:
        return 0.0
    return float(np.exp(-((hour - 12) ** 2) / (2 * 3.5 ** 2)))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FILTER TO CHOSEN QUARTER
# ══════════════════════════════════════════════════════════════════════════════

def filter_quarter(df: pd.DataFrame,
                   start: str, end: str) -> pd.DataFrame:
    """Return rows within [start, end] date range (inclusive)."""
    mask = (df['Date'] >= pd.Timestamp(start)) & \
           (df['Date'] <= pd.Timestamp(end))
    result = df[mask].copy()
    result['PV_Weight'] = result['Hour'].apply(pv_weight)
    result['Revenue']   = (result['DAM_Price']
                           * result['PV_Weight']
                           * 50   # MW capacity
                           * 0.30 # capacity factor
                           )
    print(f"  ✓ Quarter filtered: {len(result):,} rows  "
          f"({result['Date'].min().date()} → {result['Date'].max().date()})")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# VISUAL 1 — WEEKLY REVENUE AT RISK
# ══════════════════════════════════════════════════════════════════════════════

def plot_visual1_revenue_at_risk(df_q: pd.DataFrame,
                                  quarter_label: str,
                                  output_dir: str) -> None:
    """
    Bar chart of weekly revenue vs. expected, P5, P95 bands.
    Bars coloured green (above expected) / red (below expected).
    """
    print("\n  Building Visual 1 — Revenue at Risk...")

    # ── aggregate to weekly ────────────────────────────────────────────────
    df_q = df_q.copy()
    df_q['Week'] = df_q['Date'].dt.to_period('W')
    weekly = (df_q.groupby('Week')['Revenue']
                  .sum()
                  .reset_index())
    weekly.columns = ['Week', 'Revenue']

    # drop partial edge weeks (< 5 days of data)
    weekly = weekly[weekly['Revenue'] > 1000].reset_index(drop=True)
    weekly['Week_str'] = weekly['Week'].astype(str).str[:10]

    revenues = weekly['Revenue'].values
    weeks    = weekly['Week_str'].tolist()
    expected = revenues.mean()
    p5_val   = np.percentile(revenues, 5)
    p95_val  = np.percentile(revenues, 95)

    # ── plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    bar_colors = [GREEN if r >= expected else RED for r in revenues]
    bars = ax.bar(range(len(revenues)), revenues / 1000,
                  color=bar_colors, width=0.65, zorder=3,
                  alpha=0.85, edgecolor=DARK_BG, linewidth=0.8)

    # P5–P95 shaded band
    ax.fill_between([-0.5, len(revenues) - 0.5],
                    [p5_val / 1000] * 2,
                    [p95_val / 1000] * 2,
                    color=BLUE, alpha=0.08, zorder=1)

    ax.axhline(expected / 1000, color=BLUE,  lw=2,   ls='--', zorder=4,
               label=f'Expected  ${expected/1000:.1f}k/wk')
    ax.axhline(p5_val  / 1000, color=RED,   lw=1.4, ls=':',  zorder=4,
               label=f'P5 (worst)  ${p5_val/1000:.1f}k')
    ax.axhline(p95_val / 1000, color=GREEN, lw=1.4, ls=':',  zorder=4,
               label=f'P95 (best)  ${p95_val/1000:.1f}k')

    # value labels on bars
    for bar, rev in zip(bars, revenues):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f'${rev/1000:.0f}k',
                ha='center', va='bottom',
                fontsize=8, color=TEXT_CLR, fontweight='bold')

    # callout: lowest week
    min_idx = revenues.argmin()
    ax.annotate(
        f'Lowest week\n${revenues[min_idx]/1000:.0f}k\n'
        f'({(revenues[min_idx]-expected)/expected*100:.0f}% vs exp)',
        xy=(min_idx, revenues[min_idx] / 1000),
        xytext=(min_idx + 1.4, revenues[min_idx] / 1000 + 3),
        fontsize=8.5, color=RED, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=RED, lw=1.3))

    # callout: highest week
    max_idx = revenues.argmax()
    ax.annotate(
        f'Peak week\n${revenues[max_idx]/1000:.0f}k',
        xy=(max_idx, revenues[max_idx] / 1000),
        xytext=(max_idx - 2, revenues[max_idx] / 1000 + 2),
        fontsize=8.5, color=AMBER, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=AMBER, lw=1.3))

    ax.set_xticks(range(len(weeks)))
    ax.set_xticklabels(weeks, rotation=40, ha='right', fontsize=8)
    ax.set_xlabel('Week starting', fontsize=10, labelpad=8)
    ax.set_ylabel('Revenue  ($k)', fontsize=10, labelpad=8)
    ax.set_title(
        f'VISUAL 1 — WEEKLY REVENUE AT RISK  (VaR)\n'
        f'GreenCo DAM Exposure  ·  {quarter_label}',
        fontsize=13, fontweight='bold', color=TEXT_CLR, pad=14)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    # legend 1: lines
    leg1 = ax.legend(loc='upper left', fontsize=8.5,
                     facecolor=PANEL_BG, edgecolor=GRID_CLR,
                     labelcolor=TEXT_CLR, framealpha=0.9)
    # legend 2: bar colours
    handles2 = [Patch(color=GREEN, alpha=0.85, label='Above expected'),
                Patch(color=RED,   alpha=0.85, label='Below expected')]
    ax.legend(handles=handles2, loc='upper right', fontsize=8.5,
              facecolor=PANEL_BG, edgecolor=GRID_CLR,
              labelcolor=TEXT_CLR, framealpha=0.9)
    ax.add_artist(leg1)

    # VaR summary box
    var_pct = abs(p5_val - expected) / expected * 100
    ax.text(0.01, 0.04,
            f'VaR (P5):  −${abs(p5_val-expected)/1000:.1f}k  (−{var_pct:.0f}% vs expected)',
            transform=ax.transAxes, fontsize=9, color=RED,
            bbox=dict(facecolor=PANEL_BG, edgecolor=RED,
                      alpha=0.85, boxstyle='round,pad=0.4'))

    plt.tight_layout(pad=1.8)
    outpath = os.path.join(output_dir, 'visual1_revenue_at_risk.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  ✓ Saved → {outpath}")


# ══════════════════════════════════════════════════════════════════════════════
# VISUAL 2 — PRICE VOLATILITY HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def plot_visual2_volatility_heatmap(df_q: pd.DataFrame,
                                     quarter_label: str,
                                     output_dir: str) -> None:
    """
    Side-by-side heatmaps: price std-dev (left) and avg price (right),
    broken down by Hour of Day × Day of Week.
    Solar window and 10–15 reversal band are highlighted.
    """
    print("\n  Building Visual 2 — Volatility Heatmap...")

    day_order = ['Monday','Tuesday','Wednesday',
                 'Thursday','Friday','Saturday','Sunday']

    vol_matrix = (df_q.pivot_table(values='DAM_Price', index='Hour',
                                    columns='DayOfWeek', aggfunc='std')
                      [day_order].fillna(0))
    avg_matrix = (df_q.pivot_table(values='DAM_Price', index='Hour',
                                    columns='DayOfWeek', aggfunc='mean')
                      [day_order].fillna(0))

    cmap_vol = LinearSegmentedColormap.from_list(
        'vol', ['#0d2b3e', '#1e4d6b', AMBER, RED], N=256)
    cmap_avg = LinearSegmentedColormap.from_list(
        'avg', ['#0d2b3e', '#1a3f5c', GREEN, AMBER, RED], N=256)

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), facecolor=DARK_BG,
                              gridspec_kw={'width_ratios': [1.05, 1]})

    panels = [
        (axes[0], vol_matrix, cmap_vol,
         'PRICE VOLATILITY  (Std Dev $/MWh)', 'Std Dev ($/MWh)'),
        (axes[1], avg_matrix, cmap_avg,
         'AVERAGE PRICE  ($/MWh)',             'Avg Price ($/MWh)'),
    ]

    for ax, matrix, cmap, title, cbar_lbl in panels:
        ax.set_facecolor(PANEL_BG)
        im = ax.imshow(matrix.values, aspect='auto', cmap=cmap,
                       interpolation='nearest')

        # cell value annotations
        vmax = matrix.values.max() or 1
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                val = matrix.iloc[r, c]
                if val == 0:
                    continue
                clr = '#0d1117' if (val / vmax) > 0.6 else MUTED
                ax.text(c, r, f'{val:.0f}',
                        ha='center', va='center',
                        fontsize=7, color=clr, fontweight='bold')

        # solar window border (dashed blue, hours 6-18)
        for r in range(6, 19):
            ax.add_patch(plt.Rectangle(
                (-0.5, r - 0.5), len(day_order), 1,
                fill=False, edgecolor=BLUE,
                lw=0.6, ls='--', alpha=0.5))

        # 10–15 reversal shading
        ax.add_patch(plt.Rectangle(
            (-0.5, 9.5), len(day_order), 6,
            fill=True, facecolor=PURPLE, alpha=0.13, linewidth=0))

        ax.set_xticks(range(len(day_order)))
        ax.set_xticklabels([d[:3] for d in day_order], fontsize=9.5)
        ax.set_yticks(range(24))
        ax.set_yticklabels([f'{h:02d}:00' for h in range(24)], fontsize=7.5)
        ax.set_title(title, fontsize=11, fontweight='bold',
                     color=TEXT_CLR, pad=10)
        ax.set_xlabel('Day of Week', fontsize=10)
        ax.set_ylabel('Hour of Day', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)

        cb = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
        cb.set_label(cbar_lbl, color=TEXT_CLR, fontsize=8.5)
        cb.ax.yaxis.set_tick_params(color=MUTED, labelcolor=MUTED)

        handles = [
            Line2D([0],[0], color=BLUE, lw=1.2, ls='--',
                   label='Solar hours (06–18)'),
            Patch(facecolor=PURPLE, alpha=0.4,
                  label='10:00–15:00 reversal zone'),
        ]
        ax.legend(handles=handles, loc='lower right', fontsize=8,
                  facecolor=PANEL_BG, edgecolor=GRID_CLR,
                  labelcolor=TEXT_CLR, framealpha=0.9)

    # insight box on left panel
    axes[0].text(0.02, 0.98,
        f'Insight: mid-day volatility elevated\nin 10–15 reversal band\n'
        f'(likely solar duck curve effect)',
        transform=axes[0].transAxes, fontsize=8.5, va='top',
        color=AMBER,
        bbox=dict(facecolor=PANEL_BG, edgecolor=AMBER,
                  alpha=0.88, boxstyle='round,pad=0.4'))

    fig.suptitle(
        f'VISUAL 2 — DAM PRICE VOLATILITY HEATMAP  ·  Hour × Day of Week\n'
        f'GreenCo SAPP Exposure  ·  {quarter_label}',
        fontsize=13, fontweight='bold', color=TEXT_CLR, y=1.01)

    plt.tight_layout(pad=2)
    outpath = os.path.join(output_dir, 'visual2_volatility_heatmap.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  ✓ Saved → {outpath}")


# ══════════════════════════════════════════════════════════════════════════════
# VISUAL 3 — ROLLING GENERATION–PRICE CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_visual3_rolling_correlation(df_q: pd.DataFrame,
                                      quarter_label: str,
                                      output_dir: str) -> None:
    """
    Three-panel chart:
      Top    — 14-day rolling correlation between daily gen and daily avg price
      Middle — 14-day rolling average DAM price
      Bottom — Daily generation (MWh)
    """
    print("\n  Building Visual 3 — Rolling Correlation...")

    # ── build daily solar aggregates ───────────────────────────────────────
    solar = df_q[df_q['PV_Weight'] > 0].copy()

    daily = (solar.groupby('Date_Only')
                  .apply(lambda x: pd.Series({
                      'Avg_Price': ((x['DAM_Price'] * x['PV_Weight']).sum()
                                    / x['PV_Weight'].sum()),
                      'Total_Gen': x['PV_Weight'].sum() * 50 * 0.30,
                  }))
                  .reset_index())

    daily['Date_Only'] = pd.to_datetime(daily['Date_Only'])
    daily = daily.sort_values('Date_Only').reset_index(drop=True)

    # 14-day rolling window (appropriate for a single quarter)
    daily['Rolling_Corr']      = (daily['Avg_Price']
                                   .rolling(14)
                                   .corr(daily['Total_Gen']))
    daily['Rolling_Price_Avg'] = daily['Avg_Price'].rolling(14).mean()

    corr_valid   = daily.dropna(subset=['Rolling_Corr']).copy()
    overall_corr = daily['Avg_Price'].corr(daily['Total_Gen'])

    # ── build figure ───────────────────────────────────────────────────────
    fig  = plt.figure(figsize=(14, 9), facecolor=DARK_BG)
    gs   = fig.add_gridspec(3, 1, height_ratios=[2.6, 1.4, 1], hspace=0.07)
    ax_c = fig.add_subplot(gs[0])                        # correlation
    ax_p = fig.add_subplot(gs[1], sharex=ax_c)           # price
    ax_g = fig.add_subplot(gs[2], sharex=ax_c)           # generation

    for ax in [ax_c, ax_p, ax_g]:
        ax.set_facecolor(PANEL_BG)
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.grid(True, zorder=0)

    # ── TOP: correlation ──────────────────────────────────────────────────
    dates = corr_valid['Date_Only']
    corr  = corr_valid['Rolling_Corr']

    ax_c.fill_between(dates, corr, 0,
                      where=corr >= 0, color=GREEN, alpha=0.35,
                      interpolate=True, label='Positive (favourable)')
    ax_c.fill_between(dates, corr, 0,
                      where=corr < 0,  color=RED,   alpha=0.35,
                      interpolate=True, label='Negative (adverse)')
    ax_c.plot(dates, corr, color=BLUE, lw=2.5, zorder=4)
    ax_c.axhline(0,    color=MUTED, lw=1.0, ls='--', zorder=3)
    ax_c.axhline(-0.5, color=RED,   lw=0.8, ls=':',  alpha=0.55)

    # annotate worst & best
    if len(corr) > 0:
        w_i = corr.idxmin()
        b_i = corr.idxmax()
        ax_c.annotate(f'Min: {corr[w_i]:.2f}',
                      xy=(dates[w_i], corr[w_i]),
                      xytext=(dates[w_i], corr[w_i] - 0.12),
                      ha='center', fontsize=9, color=RED, fontweight='bold',
                      arrowprops=dict(arrowstyle='->', color=RED, lw=1.3))
        ax_c.annotate(f'Max: {corr[b_i]:.2f}',
                      xy=(dates[b_i], corr[b_i]),
                      xytext=(dates[b_i], corr[b_i] + 0.12),
                      ha='center', fontsize=9, color=GREEN, fontweight='bold',
                      arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.3))

    # month separator lines
    months_in_quarter = pd.date_range(
        start=daily['Date_Only'].min().replace(day=1),
        end=daily['Date_Only'].max(),
        freq='MS'
    )[1:]
    for ms in months_in_quarter:
        for ax in [ax_c, ax_p, ax_g]:
            ax.axvline(ms, color=MUTED, lw=0.9, ls=':', alpha=0.6)
        ax_c.text(ms, -0.95, ms.strftime('%b %Y'),
                  ha='center', fontsize=7.5, color=MUTED, style='italic')

    ax_c.set_ylim(-1.05, 0.5)
    ax_c.set_ylabel('Rolling Corr\n(14-day)', fontsize=9.5, labelpad=8)
    ax_c.set_title(
        f'VISUAL 3 — GENERATION–PRICE ROLLING CORRELATION  (14-day window)\n'
        f'GreenCo SAPP Exposure  ·  {quarter_label}',
        fontsize=13, fontweight='bold', color=TEXT_CLR, pad=12)
    ax_c.tick_params(labelbottom=False)
    ax_c.legend(loc='upper right', fontsize=8.5,
                facecolor=PANEL_BG, edgecolor=GRID_CLR,
                labelcolor=TEXT_CLR, framealpha=0.9)

    ax_c.text(0.02, 0.10,
              f'Overall correlation: {overall_corr:.3f}  '
              f'({"adverse — gen↑ when price↓" if overall_corr < 0 else "favourable"})',
              transform=ax_c.transAxes, fontsize=8.5,
              color=RED if overall_corr < 0 else GREEN,
              bbox=dict(facecolor=PANEL_BG,
                        edgecolor=RED if overall_corr < 0 else GREEN,
                        alpha=0.88, boxstyle='round,pad=0.4'))

    for y_val, lbl, clr in [(-0.75, 'ADVERSE\n(gen↑ → price↓)', RED),
                              (0.25,  'FAVOURABLE\n(gen↑ → price↑)', GREEN)]:
        ax_c.text(1.012, y_val, lbl,
                  transform=ax_c.get_yaxis_transform(),
                  ha='left', va='center',
                  fontsize=7.5, color=clr, fontweight='bold')

    # ── MIDDLE: rolling avg price ─────────────────────────────────────────
    ax_p.plot(corr_valid['Date_Only'],
              corr_valid['Rolling_Price_Avg'],
              color=AMBER, lw=2.2, zorder=3,
              label='14-day avg price')
    ax_p.fill_between(corr_valid['Date_Only'],
                      corr_valid['Rolling_Price_Avg'],
                      corr_valid['Rolling_Price_Avg'].min(),
                      color=AMBER, alpha=0.18)

    # annotate price surge if present
    if len(corr_valid) > 0:
        peak_i = corr_valid['Rolling_Price_Avg'].idxmax()
        peak_p = corr_valid['Rolling_Price_Avg'][peak_i]
        peak_d = corr_valid['Date_Only'][peak_i]
        ax_p.annotate(
            f'Peak: ${peak_p:.0f}/MWh',
            xy=(peak_d, peak_p),
            xytext=(peak_d - pd.Timedelta(days=15), peak_p - 15),
            fontsize=8.5, color=AMBER, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=AMBER, lw=1.2))

    ax_p.set_ylabel('14-day avg\nPrice ($/MWh)', fontsize=9, labelpad=8)
    ax_p.tick_params(labelbottom=False)
    ax_p.legend(loc='upper left', fontsize=8,
                facecolor=PANEL_BG, edgecolor=GRID_CLR,
                labelcolor=TEXT_CLR, framealpha=0.9)

    # ── BOTTOM: daily generation ──────────────────────────────────────────
    ax_g.bar(daily['Date_Only'], daily['Total_Gen'],
             color=BLUE, alpha=0.55, width=1.2, zorder=3,
             label='Daily generation (MWh)')

    # mark the two generation tiers (weekday vs weekend)
    gen_median  = daily['Total_Gen'].median()
    gen_weekend = daily[daily['Date_Only'].dt.dayofweek >= 5]['Total_Gen'].mean()
    gen_weekday = daily[daily['Date_Only'].dt.dayofweek <  5]['Total_Gen'].mean()
    if not np.isnan(gen_weekday):
        ax_g.axhline(gen_weekday, color=GREEN,  lw=1, ls='--', alpha=0.7,
                     label=f'Weekday avg ~{gen_weekday:.0f} MWh')
    if not np.isnan(gen_weekend):
        ax_g.axhline(gen_weekend, color=PURPLE, lw=1, ls='--', alpha=0.7,
                     label=f'Weekend avg ~{gen_weekend:.0f} MWh')

    ax_g.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax_g.xaxis.set_major_locator(
        mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax_g.get_xticklabels(),
             rotation=25, ha='right', fontsize=8.5)
    ax_g.set_ylabel('Daily gen\n(MWh)', fontsize=9, labelpad=8)
    ax_g.legend(loc='upper right', fontsize=7.5, ncol=3,
                facecolor=PANEL_BG, edgecolor=GRID_CLR,
                labelcolor=TEXT_CLR, framealpha=0.9)

    plt.savefig(os.path.join(output_dir, 'visual3_rolling_correlation.png'),
                dpi=300, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    outpath = os.path.join(output_dir, 'visual3_rolling_correlation.png')
    print(f"  ✓ Saved → {outpath}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — interactive entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  GreenCo E5 Dashboard Generator")
    print("  Generates 3 DAM Price Volatility Exposure Visuals")
    print("=" * 65)

    # ── 1. Get Excel file path ─────────────────────────────────────────────
    print("\nSTEP 1 — Excel file")
    default_name = 'Data Analyst SAPP Market Extract.xlsx'
    user_path = input(
        f"  Enter path to Excel file\n"
        f"  (press Enter for default: '{default_name}'): "
    ).strip()
    filepath = user_path if user_path else default_name

    if not os.path.exists(filepath):
        print(f"\n  ✗ File not found: {filepath}")
        print("    Please check the path and try again.")
        sys.exit(1)

    # ── 2. Load data ───────────────────────────────────────────────────────
    print("\nSTEP 2 — Loading data")
    df = load_data(filepath)

    # Show available date range so user can pick a valid quarter
    print(f"\n  Available date range: "
          f"{df['Date'].min().date()} → {df['Date'].max().date()}")
    print("""
  Suggested quarters (SAPP year Apr 2021 – Mar 2022):
    Q1  Apr 2021 – Jun 2021   →  enter: 2021-04-01  2021-06-30
    Q2  Jul 2021 – Sep 2021   →  enter: 2021-07-01  2021-09-30
    Q3  Oct 2021 – Dec 2021   →  enter: 2021-10-01  2021-12-31
    Q4  Jan 2022 – Mar 2022   →  enter: 2022-01-01  2022-03-31  (default)
    """)

    # ── 3. Get quarter date range ──────────────────────────────────────────
    print("STEP 3 — Choose quarter")
    start_inp = input(
        "  Start date YYYY-MM-DD  (press Enter for 2022-01-01): "
    ).strip()
    end_inp   = input(
        "  End   date YYYY-MM-DD  (press Enter for 2022-03-31): "
    ).strip()

    start_date = start_inp if start_inp else '2022-01-01'
    end_date   = end_inp   if end_inp   else '2022-03-31'

    try:
        pd.Timestamp(start_date)
        pd.Timestamp(end_date)
    except Exception:
        print("  ✗ Invalid date format. Use YYYY-MM-DD.")
        sys.exit(1)

    quarter_label = f"{pd.Timestamp(start_date).strftime('%b %Y')} – " \
                    f"{pd.Timestamp(end_date).strftime('%b %Y')}"

    print(f"\n  Quarter selected: {quarter_label}")

    # ── 4. Filter to quarter ───────────────────────────────────────────────
    print("\nSTEP 4 — Filtering to quarter")
    df_q = filter_quarter(df, start_date, end_date)

    if len(df_q) == 0:
        print("  ✗ No data found for this date range.")
        sys.exit(1)

    # ── 5. Output directory ────────────────────────────────────────────────
    print("\nSTEP 5 — Output directory")
    out_inp = input(
        "  Where to save images?\n"
        "  (press Enter to save in same folder as this script): "
    ).strip()
    output_dir = out_inp if out_inp else os.path.dirname(
        os.path.abspath(__file__))

    os.makedirs(output_dir, exist_ok=True)

    # ── 6. Generate all three visuals ─────────────────────────────────────
    print("\nSTEP 6 — Generating visuals")

    plot_visual1_revenue_at_risk(df_q, quarter_label, output_dir)
    plot_visual2_volatility_heatmap(df_q, quarter_label, output_dir)
    plot_visual3_rolling_correlation(df_q, quarter_label, output_dir)

    print(f"""
{"=" * 65}
  ✓ All 3 visuals saved to:
    {output_dir}

  Files:
    visual1_revenue_at_risk.png
    visual2_volatility_heatmap.png
    visual3_rolling_correlation.png
{"=" * 65}
    """)


if __name__ == '__main__':
    main()