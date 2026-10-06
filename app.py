from datetime import datetime, timedelta
import io
import sys
import textwrap
import traceback
import zipfile
from flask import Flask, jsonify, request, send_file
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy
import numpy as np

app = Flask(__name__)


def excel_to_datetime(excel_date):
    try:
        return datetime(1899, 12, 30) + timedelta(days=float(excel_date))
    except (ValueError, TypeError, OverflowError):
        return datetime(2000, 1, 1)


def safe_float_array(arr_data):
    """Safely parses any list/array to float numpy array, converting invalid entries to NaN."""
    if not arr_data:
        return np.array([], dtype=float)
    cleaned = []
    for val in arr_data:
        try:
            cleaned.append(float(str(val).strip()))
        except (ValueError, TypeError):
            cleaned.append(np.nan)
    return np.array(cleaned, dtype=float)


@app.route('/plot', methods=['POST'])
def plot():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON payload provided'}), 400

        sensors = data.get('sensors', [])
        if not sensors:
            return jsonify({'error': 'No sensor data provided'}), 400

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            for idx, sensor in enumerate(sensors):
                sensor_id = sensor.get('id', f'sensor_{idx + 1}')

                # Dynamic CCW rotation for TILT DATA ONLY (defaults to 0)
                rotation_deg = float(sensor.get('rotationCCW', 0))
                theta_rad = np.radians(rotation_deg)

                x_raw = safe_float_array(sensor.get('ew', []))
                y_raw = safe_float_array(sensor.get('ns', []))
                dates = safe_float_array(sensor.get('dates', []))

                # Raw and parsed wind data
                wind_gusts = safe_float_array(sensor.get('wind_gust', []))
                wind_dirs = safe_float_array(sensor.get('wind_direction', []))
                wind_gust_min = float(sensor.get('wind_gust_minimum', 0))

                # Apply CCW rotation matrix strictly to tilt coordinates
                xplot = x_raw * np.cos(theta_rad) - y_raw * np.sin(theta_rad)
                yplot = x_raw * np.sin(theta_rad) + y_raw * np.cos(theta_rad)

                fig, ax = plt.subplots(figsize=(7.5, 7.5))

                # Format title: break line before Phase if present, otherwise wrap at width=32
                if ' Phase' in sensor_id:
                    parts = sensor_id.split(' Phase')
                    formatted_title = (
                        f'{textwrap.fill(parts[0], width=32)}\nPhase{parts[1]}'
                    )
                else:
                    formatted_title = textwrap.fill(sensor_id, width=32)

                ax.set_title(
                    formatted_title, fontsize=11, fontweight='bold', pad=18
                )

                # Fixed radial scale: 0.02°, 0.04°, 0.06°
                radii = [0.02, 0.04, 0.06]
                max_r = radii[-1]

                # Draw concentric circles & labels (smaller font 7pt, not bold, shifted slightly left)
                theta = np.linspace(0, 2 * np.pi, 300)
                for r in radii:
                    ax.plot(
                        r * np.cos(theta),
                        r * np.sin(theta),
                        color='black',
                        lw=0.9,
                        zorder=1,
                    )
                    ax.text(
                        r + 0.0006,
                        -0.0035,
                        f'{r:.2f}°',
                        ha='left',
                        va='top',
                        fontsize=7,
                        fontweight='normal',
                        zorder=4,
                    )

                # Draw crosshairs
                ax.plot(
                    [-max_r, max_r], [0, 0], color='black', lw=0.9, zorder=0
                )
                ax.plot(
                    [0, 0], [-max_r, max_r], color='black', lw=0.9, zorder=0
                )

                # Colored Scatter (zorder=3)
                if len(dates) > 0 and len(xplot) > 0:
                    dates_dt = np.array([excel_to_datetime(d) for d in dates])
                    day_of_year = np.array(
                        [d.timetuple().tm_yday for d in dates_dt]
                    )

                    sc = ax.scatter(
                        xplot,
                        yplot,
                        c=day_of_year,
                        s=40,
                        cmap='jet',
                        edgecolors='none',
                        zorder=3,
                    )
                    sc.set_clim(1, 365)

                    # Colorbar
                    month_starts = [
                        1,
                        32,
                        60,
                        91,
                        121,
                        152,
                        182,
                        213,
                        244,
                        274,
                        305,
                        335,
                    ]
                    month_labels = [
                        'Jan',
                        'Feb',
                        'Mar',
                        'Apr',
                        'May',
                        'Jun',
                        'Jul',
                        'Aug',
                        'Sep',
                        'Oct',
                        'Nov',
                        'Dec',
                    ]

                    cbar = plt.colorbar(sc, ax=ax, pad=0.12, shrink=0.75)
                    cbar.set_ticks(month_starts)
                    cbar.set_ticklabels(month_labels)

                # Scaling Parameters for Wind Gust Arrows
                min_gust_ref = 15.0
                min_tail_pt = 4.0
                scale_rate = 0.56  # points per mph

                # Overlay Wind Direction Arrows scaled by Wind Gust
                if len(wind_gusts) > 0 and len(wind_dirs) > 0:
                    min_len = min(len(xplot), len(wind_gusts), len(wind_dirs))
                    x_sub = xplot[:min_len]
                    y_sub = yplot[:min_len]
                    gusts_sub = wind_gusts[:min_len]
                    dirs_sub = wind_dirs[:min_len]

                    # Filter points where gust >= wind_gust_min and direction is valid
                    valid_gusts = np.nan_to_num(gusts_sub, nan=0.0)
                    mask = (valid_gusts >= wind_gust_min) & (
                        ~np.isnan(dirs_sub)
                    )

                    if np.any(mask):
                        x_high = x_sub[mask]
                        y_high = y_sub[mask]
                        gusts_high = gusts_sub[mask]
                        dirs_high = dirs_sub[mask]

                        wind_math_deg = (90 - dirs_high) % 360
                        wind_math_rad = np.radians(wind_math_deg)

                        arrow_lengths = min_tail_pt + np.maximum(
                            0, gusts_high - min_gust_ref
                        ) * scale_rate

                        for x_pt, y_pt, rad, tail_len in zip(
                            x_high, y_high, wind_math_rad, arrow_lengths
                        ):
                            dx = np.cos(rad) * tail_len
                            dy = np.sin(rad) * tail_len

                            trans_centered = offset_copy(
                                ax.transData,
                                fig=fig,
                                x=dx / 2.0,
                                y=dy / 2.0,
                                units='points',
                            )

                            ax.annotate(
                                '',
                                xy=(x_pt, y_pt),
                                xycoords=trans_centered,
                                xytext=(-dx, -dy),
                                textcoords='offset points',
                                arrowprops=dict(
                                    arrowstyle=(
                                        '->,head_length=0.2,head_width=0.15'
                                    ),
                                    color='black',
                                    lw=0.8,
                                ),
                                zorder=10,
                            )

                # Formatting Limits and Cardinal Directions
                padding = max_r * 0.22
                limit = max_r + padding
                ax.set_xlim(-limit, limit)
                ax.set_ylim(-limit, limit)
                ax.set_aspect('equal')

                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.tick_params(
                    left=False, bottom=False, labelleft=False, labelbottom=False
                )

                label_dist = max_r + 0.005
                ax.text(
                    0,
                    label_dist,
                    'North',
                    ha='center',
                    va='bottom',
                    fontsize=11,
                    fontweight='bold',
                )
                ax.text(
                    0,
                    -label_dist,
                    'South',
                    ha='center',
                    va='top',
                    fontsize=11,
                    fontweight='bold',
                )
                ax.text(
                    label_dist,
                    0,
                    'East',
                    ha='left',
                    va='center',
                    fontsize=11,
                    fontweight='bold',
                )
                ax.text(
                    -label_dist,
                    0,
                    'West',
                    ha='right',
                    va='center',
                    fontsize=11,
                    fontweight='bold',
                )

                # --- Legend Box: 30 mph and 40 mph Reference Arrows ---
                ax_leg = fig.add_axes([0.08, 0.02, 0.35, 0.16])
                ax_leg.set_xlim(0, 1)
                ax_leg.set_ylim(0, 1)
                ax_leg.axis('off')

                ref_angle_rad = np.radians(45)

                # 1. 30 mph Reference Arrow
                dot30_x, dot30_y = 0.15, 0.70
                ax_leg.scatter(
                    [dot30_x],
                    [dot30_y],
                    color='mediumturquoise',
                    s=30,
                    zorder=3,
                )

                ref30_gust = 30.0
                ref30_len_pt = (
                    min_tail_pt
                    + max(0.0, ref30_gust - min_gust_ref) * scale_rate
                )
                dx30_pt = np.cos(ref_angle_rad) * ref30_len_pt
                dy30_pt = np.sin(ref_angle_rad) * ref30_len_pt

                trans_leg30 = offset_copy(
                    ax_leg.transData,
                    fig=fig,
                    x=dx30_pt / 2.0,
                    y=dy30_pt / 2.0,
                    units='points',
                )
                ax_leg.annotate(
                    '',
                    xy=(dot30_x, dot30_y),
                    xycoords=trans_leg30,
                    xytext=(-dx30_pt, -dy30_pt),
                    textcoords='offset points',
                    arrowprops=dict(
                        arrowstyle='->,head_length=0.2,head_width=0.15',
                        color='black',
                        lw=0.8,
                    ),
                    zorder=4,
                )
                ax_leg.text(
                    dot30_x + 0.18,
                    dot30_y - 0.05,
                    '30 mph',
                    fontsize=8,
                    va='center',
                    fontweight='bold',
                )

                # 2. 40 mph Reference Arrow
                dot40_x, dot40_y = 0.15, 0.25
                ax_leg.scatter(
                    [dot40_x],
                    [dot40_y],
                    color='mediumturquoise',
                    s=30,
                    zorder=3,
                )

                ref40_gust = 40.0
                ref40_len_pt = (
                    min_tail_pt
                    + max(0.0, ref40_gust - min_gust_ref) * scale_rate
                )
                dx40_pt = np.cos(ref_angle_rad) * ref40_len_pt
                dy40_pt = np.sin(ref_angle_rad) * ref40_len_pt

                trans_leg40 = offset_copy(
                    ax_leg.transData,
                    fig=fig,
                    x=dx40_pt / 2.0,
                    y=dy40_pt / 2.0,
                    units='points',
                )
                ax_leg.annotate(
                    '',
                    xy=(dot40_x, dot40_y),
                    xycoords=trans_leg40,
                    xytext=(-dx40_pt, -dy40_pt),
                    textcoords='offset points',
                    arrowprops=dict(
                        arrowstyle='->,head_length=0.2,head_width=0.15',
                        color='black',
                        lw=0.8,
                    ),
                    zorder=4,
                )
                ax_leg.text(
                    dot40_x + 0.22,
                    dot40_y - 0.05,
                    '40 mph',
                    fontsize=8,
                    va='center',
                    fontweight='bold',
                )

                # Adjusted figure margins to ensure title and labels are well within bounds
                plt.subplots_adjust(
                    left=0.12, right=0.82, bottom=0.18, top=0.84
                )

                # Save image using sanitized filename (dpi=100 ensures image size < 1MB)
                safe_filename = (
                    ''.join(
                        c
                        for c in sensor_id
                        if c.isalnum() or c in (' ', '_', '-')
                    )
                    .strip()
                    .replace(' ', '_')
                )
                if not safe_filename:
                    safe_filename = f'plot_{idx + 1}'

                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, transparent=True)
                plt.close(fig)
                buf.seek(0)

                zf.writestr(f'{safe_filename}.png', buf.read())

        zip_buffer.seek(0)
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='tilt_plots.zip',
        )

    except Exception as e:
        error_msg = traceback.format_exc()
        print(
            f'=== ERROR PROCESSING PLOT ===\n{error_msg}',
            file=sys.stderr,
            flush=True,
        )
        return jsonify({'error': str(e), 'traceback': error_msg}), 500
