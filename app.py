from flask import Flask, request, send_file, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import zipfile
import sys
from datetime import datetime, timedelta

app = Flask(__name__)

def excel_to_datetime(excel_date):
    return datetime(1899, 12, 30) + timedelta(days=float(excel_date))

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
    data = request.get_json()
    sensors = data.get("sensors", [])

    if not sensors:
        return jsonify({"error": "No sensor data provided"}), 400

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for sensor in sensors:
            sensor_id = sensor.get("id", "unknown")

            # Dynamic CCW rotation for TILT DATA ONLY (defaults to 22)
            rotation_deg = float(sensor.get("rotationCCW", 22))
            theta_rad = np.radians(rotation_deg)

            x_raw = safe_float_array(sensor.get("ew", []))
            y_raw = safe_float_array(sensor.get("ns", []))
            dates = safe_float_array(sensor.get("dates", []))

            # Raw and parsed wind data
            wind_gusts = safe_float_array(sensor.get("wind_gust", []))
            wind_dirs = safe_float_array(sensor.get("wind_direction", []))

            # Apply CCW rotation matrix strictly to tilt coordinates
            xplot = x_raw * np.cos(theta_rad) - y_raw * np.sin(theta_rad)
            yplot = x_raw * np.sin(theta_rad) + y_raw * np.cos(theta_rad)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_title(f"Tilt Meter: {sensor_id}", fontsize=14, fontweight='bold', pad=34)

            # Determine ring spacing dynamically
            base_spacing = 0.01
            radial_distances = np.sqrt(xplot**2 + yplot**2)
            max_tilt = radial_distances.max() if len(radial_distances) > 0 else 0.03

            scale_factor = int(np.ceil(max_tilt / 0.03))
            if scale_factor < 1:
                scale_factor = 1

            spacing = base_spacing * scale_factor
            radii = [round(spacing * i, 4) for i in range(1, 4)]

            # Draw circles
            label_offset = spacing * 0.02
            offset = spacing * 0.5 + 0.005
            for r in radii:
                theta = np.linspace(0, 2*np.pi, 300)
                ax.plot(r*np.cos(theta), r*np.sin(theta), color='black', lw=1)
                ax.text(r + label_offset, -offset, f"{r:.2f}°", va='center', fontsize=10, zorder=4)

            # Draw crosshairs
            ax.plot([-radii[2], radii[2]], [0, 0], color='black', lw=1, zorder=0)
            ax.plot([0, 0], [-radii[2], radii[2]], color='black', lw=1, zorder=0)

            # Colored Scatter (zorder=3)
            dates_dt = np.array([excel_to_datetime(d) for d in dates])
            day_of_year = np.array([d.timetuple().tm_yday for d in dates_dt])
            
            sc = ax.scatter(
                xplot, yplot,
                c=day_of_year,
                s=50,
                cmap='jet',
                edgecolors='none',
                zorder=3
            )
            sc.set_clim(1, 365)

            # Overlay Tiny Wind Direction Arrows for High Wind Gusts (≥ 20 mph)
            if len(wind_gusts) > 0 and len(wind_dirs) > 0:
                min_len = min(len(xplot), len(wind_gusts), len(wind_dirs))
                x_sub = xplot[:min_len]
                y_sub = yplot[:min_len]
                gusts_sub = wind_gusts[:min_len]
                dirs_sub = wind_dirs[:min_len]

                # Filter points where gust >= 20 and wind direction is valid
                valid_gusts = np.nan_to_num(gusts_sub, nan=0.0)
                mask = (valid_gusts >= 20.0) & (~np.isnan(dirs_sub))

                if np.any(mask):
                    x_high = x_sub[mask]
                    y_high = y_sub[mask]
                    dirs_high = dirs_sub[mask]

                    # Convert true compass degrees (0=North, 90=East) directly to math polar angle
                    wind_math_deg = (90 - dirs_high) % 360
                    wind_math_rad = np.radians(wind_math_deg)

                    # Fixed pixel length (6pt) ensures tiny, consistent arrows across any axis scale
                    arrow_len_pixels = 6.0

                    for x_pt, y_pt, rad in zip(x_high, y_high, wind_math_rad):
                        dx = np.cos(rad) * arrow_len_pixels
                        dy = np.sin(rad) * arrow_len_pixels

                        ax.annotate(
                            '',
                            xy=(x_pt, y_pt),                      # Arrow tip
                            xytext=(-dx, -dy),                    # Tail offset
                            textcoords='offset points',
                            arrowprops=dict(
                                arrowstyle='->,head_length=0.2,head_width=0.15',
                                color='black',
                                lw=0.8
                            ),
                            zorder=10                             # Renders above all scatter dots
                        )

            # Colorbar
            month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
            month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

            cbar = plt.colorbar(sc, ax=ax, pad=0.18)
            cbar.set_ticks(month_starts)
            cbar.set_ticklabels(month_labels)

            # Formatting
            padding = spacing * 0.2
            limit = radii[-1] + padding
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect('equal')

            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            label_offset_factor = 2.5
            ax.text(0, radii[-1] + label_offset*label_offset_factor, "North", ha='center', va='bottom', fontsize=12, fontweight='bold')
            ax.text(0, -radii[-1] - label_offset*label_offset_factor, "South", ha='center', va='top', fontsize=12, fontweight='bold')
            ax.text(radii[-1] + label_offset*label_offset_factor, 0, "East", ha='left', va='center', fontsize=12, fontweight='bold')
            ax.text(-radii[-1] - label_offset*label_offset_factor, 0, "West", ha='right', va='center', fontsize=12, fontweight='bold')

            plt.tight_layout()
            plt.subplots_adjust(right=0.85, bottom=0.15, top=0.90)

            # Save image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, transparent=True)
            plt.close(fig)
            buf.seek(0)

            zf.writestr(f"{sensor_id}.png", buf.read())

    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='tilt_plots.zip')
