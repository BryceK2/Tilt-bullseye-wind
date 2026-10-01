from datetime import datetime, timedelta
import io
import sys
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
      for sensor in sensors:
        sensor_id = sensor.get('id', 'unknown')

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

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(
            f'Tilt Meter: {sensor_id}', fontsize=14, fontweight='bold', pad=34
        )

        # Determine ring spacing dynamically
        base_spacing = 0.01
        radial_distances = np.sqrt(xplot**2 + yplot**2)
        max_tilt = (
            radial_distances.max() if len(radial_distances) > 0 else 0.03
        )

        scale_factor = int(np.ceil(max_tilt / 0.03))
        if scale_factor < 1:
          scale_factor = 1

        spacing = base_spacing * scale_factor
        radii = [round(spacing * i, 4) for i in range(1, 4)]

        # Draw circles
        label_offset = spacing * 0.02
        offset = spacing * 0.5 + 0.005
        for r in radii:
          theta = np.linspace(0, 2 * np.pi, 300)
          ax.plot(r * np.cos(theta), r * np.sin(theta), color='black', lw=1)
          ax.text(
              r + label_offset,
              -offset,
              f'{r:.2f}°',
              va='center',
              fontsize=10,
              zorder=4,
          )

        # Draw crosshairs
        ax.plot([-radii[2], radii[2]], [0, 0], color='black', lw=1, zorder=0)
        ax.plot([0, 0], [-radii[2], radii[2]], color='black', lw=1, zorder=0)

        # Colored Scatter (zorder=3)
        if len(dates) > 0 and len(xplot) > 0:
          dates_dt = np.array([excel_to_datetime(d) for d in dates])
          day_of_year = np.array([d.timetuple().tm_yday for d in dates_dt])

          sc = ax.scatter(
              xplot,
              yplot,
              c=day_of_year,
              s=50,
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

          cbar = plt.colorbar(sc, ax=ax, pad=0.18)
          cbar.set_ticks(month_starts)
          cbar.set_ticklabels(month_labels)

        # Overlay Wind Direction Arrows scaled by Wind Gust (Centered on tilt dot)
        if len(wind_gusts) > 0 and len(wind_dirs) > 0:
          min_len = min(len(xplot), len(wind_gusts), len(wind_dirs))
          x_sub = xplot[:min_len]
          y_sub = yplot[:min_len]
          gusts_sub = wind_gusts[:min_len]
          dirs_sub = wind_dirs[:min_len]

          # Filter points where gust >= wind_gust_min and direction is valid
          valid_gusts = np.nan_to_num(gusts_sub, nan=0.0)
          mask = (valid_gusts >= wind_gust_min) & (~np.isnan(dirs_sub))

          if np.any(mask):
            x_high = x_sub[mask]
            y_high = y_sub[mask]
            gusts_high = gusts_sub[mask]
            dirs_high = dirs_sub[mask]

            # Convert compass degrees (0=North, 90=East) to polar angle
            wind_math_deg = (90 - dirs_high) % 360
            wind_math_rad = np.radians(wind_math_deg)

            # Scaling Parameters
            min_gust_ref = 15.0
            min_tail_pt = 4.0
            scale_rate = 0.56  # points per mph

            arrow_lengths = min_tail_pt + np.maximum(
                0, gusts_high - min_gust_ref
            ) * scale_rate

            for x_pt, y_pt, rad, tail_len in zip(
                x_high, y_high, wind_math_rad, arrow_lengths
            ):
              dx = np.cos(rad) * tail_len
              dy = np.sin(rad) * tail_len

              # Shift tip coordinate forward by half of dx and dy in points
              trans_centered = offset_copy(
                  ax.transData, fig=fig, x=dx / 2.0, y=dy / 2.0, units='points'
              )

              ax.annotate(
                  '',
                  xy=(x_pt, y_pt),  # Data point center
                  xycoords=trans_centered,  # Tip shifted forward by +length/2
                  xytext=(-dx, -dy),  # Tail starts -length behind tip
                  textcoords='offset points',
                  arrowprops=dict(
                      arrowstyle='->,head_length=0.2,head_width=0.15',
                      color='black',
                      lw=0.8,
                  ),
                  zorder=10,  # Render above scatter points
              )

        # Formatting
        padding = spacing * 0.2
        limit = radii[-1] + padding
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')

        for spine in ax.spines.values():
          spine.set_visible(False)
        ax.tick_params(
            left=False, bottom=False, labelleft=False, labelbottom=False
        )

        label_offset_factor = 2.5
        ax.text(
            0,
            radii[-1] + label_offset * label_offset_factor,
            'North',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold',
        )
        ax.text(
            0,
            -radii[-1] - label_offset * label_offset_factor,
            'South',
            ha='center',
            va='top',
            fontsize=12,
            fontweight='bold',
        )
        ax.text(
            radii[-1] + label_offset * label_offset_factor,
            0,
            'East',
            ha='left',
            va='center',
            fontsize=12,
            fontweight='bold',
        )
        ax.text(
            -radii[-1] - label_offset * label_offset_factor,
            0,
            'West',
            ha='right',
            va='center',
            fontsize=12,
            fontweight='bold',
        )

        plt.tight_layout()
        plt.subplots_adjust(right=0.85, bottom=0.15, top=0.90)

        # Save image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, transparent=True)
        plt.close(fig)
        buf.seek(0)

        zf.writestr(f'{sensor_id}.png', buf.read())

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
