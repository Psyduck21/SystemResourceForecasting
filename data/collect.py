import psutil
import time
import csv
from datetime import datetime

# Output CSV file name
OUTPUT_FILE = "system_metrics.csv"

# Sampling interval (in seconds)
INTERVAL = 5   # collect data every 5 seconds

# Number of top processes to track
TOP_N = 3

def get_top_processes(n=TOP_N):
    """Return a list of top N processes by CPU usage."""
    procs = []
    for p in psutil.process_iter(['pid', 'name', 'cpu_percent']):
        try:
            procs.append(p.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    top_procs = sorted(procs, key=lambda x: x['cpu_percent'], reverse=True)[:n]
    return top_procs

def collect_metrics():
    """Collect CPU, memory, battery, and process info and write to CSV."""
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)

        # CSV Header
        header = ["timestamp", "cpu_total", "mem_percent", "battery_percent", "proc_count"]
        for i in range(1, TOP_N + 1):
            header.append(f"top{i}_pid")
            header.append(f"top{i}_name")
            header.append(f"top{i}_cpu")
        writer.writerow(header)
        f.flush()

        print(f"✅ Collecting system metrics every {INTERVAL} seconds...")
        print(f"📁 Writing to {OUTPUT_FILE}\nPress Ctrl+C to stop.\n")

        while True:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cpu_total = psutil.cpu_percent()
                mem_percent = psutil.virtual_memory().percent
                proc_count = len(psutil.pids())

                # Battery info (if available)
                battery = psutil.sensors_battery()
                battery_percent = battery.percent if battery else None

                # Top processes
                top = get_top_processes()
                top_data = []
                for p in top:
                    top_data.extend([p['pid'], p['name'], p['cpu_percent']])
                # Pad if fewer than TOP_N processes
                while len(top_data) < TOP_N * 3:
                    top_data.extend([None, None, None])

                row = [timestamp, cpu_total, mem_percent, battery_percent, proc_count] + top_data
                writer.writerow(row)
                f.flush()

                time.sleep(INTERVAL)

            except KeyboardInterrupt:
                print("\n🛑 Data collection stopped by user.")
                break
            except Exception as e:
                print(f"⚠️ Error: {e}")
                time.sleep(INTERVAL)

if __name__ == "__main__":
    collect_metrics()
