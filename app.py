from flask import Flask, request, send_file, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import zipfile
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
            cleaned.append(float(val))
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
    theta_rad = np.radians(22)

    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for sensor in sensors:
            sensor_id = sensor.get("id", "unknown")

            x_raw = safe_float_array(sensor.get("ew", []))
            y_raw = safe_float_array(sensor.get("ns", []))
            dates = safe_float_array(sensor.get("dates", []))

            # Robust extraction of optional wind data
            wind_gusts = safe_float_array(sensor.get("wind_gust", []))
            wind_dirs = safe_float_array(sensor.get("wind_direction", []))

            # Debug logging (Check these in your server/Cloud Run logs)
            print(f"--- [DEBUG] Processing Sensor: {sensor_id} ---")
            print(f"xplot len: {len(x_raw)}, wind_gusts len: {len(wind_gusts)}, wind_dirs len: {len(wind_dirs)}")
            if len(wind_gusts) > 0:
                print(f"Max Gust Value parsed: {np.nanmax(wind_gusts)}")

            # Apply 22° CCW rotation matrix
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

            # Scatter Plot
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

            # Draw Wind Arrows
            if len(wind_gusts) > 0 and len(wind_dirs) > 0:
                # Ensure array bounds match exactly
                min_len = min(len(xplot), len(wind_gusts), len(wind_dirs))
                
                x_sub = xplot[:min_len]
                y_sub = yplot[:min_len]
                gusts_sub = wind_gusts[:min_len]
                dirs_sub = wind_dirs[:min_len]

                # Filter valid numbers >= 20
                mask = np.nan_to_num(gusts_sub, nan=0.0) >= 20
                print(f"[DEBUG] Points passing >= 20 threshold: {np.sum(mask)}")

                if np.any(mask):
                    x_wind = x_sub[mask]
                    y_wind = y_sub[mask]
                    dirs = dirs_sub[mask]

                    # Standard meteorological compass bearing (0° N, 90° E)
                    wind_polar_deg = 90 - dirs
                    wind_rad = np.radians(wind_polar_deg)

                    u = np.cos(wind_rad)
                    v = np.sin(wind_rad)

                    # Scale arrow relative to plot grid
                    arrow_length = radii[0] * 0.35

                    ax.quiver(
                        x_wind, y_wind, u, v,
                        color='red',
                        angles='xy',
                        scale_units='xy',
                        scale=1 / arrow_length,
                        width=0.007,
                        headwidth=4,
                        headlength=5,
                        zorder=5
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
