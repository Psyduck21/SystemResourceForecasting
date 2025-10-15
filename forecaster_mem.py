"""
Real-time memory forecasting using a saved N-BEATS model and nbeats_utils helpers.

Requirements:
 - nbeats_utils.py (must provide load_nbeats_model and forecast_future)
 - saved_model/ (contains model.pth, config.json, scaling.json)
 - psutil
 - pandas, numpy, torch
"""

import time
import json
from pathlib import Path
from collections import deque
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import psutil
import torch

# Try to import helper utils from nbeats_utils.py
try:
    from nbeats_utils import load_nbeats_model, forecast_future
    _HAS_NBEATS_UTILS = True
except Exception:
    _HAS_NBEATS_UTILS = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("realtime_forecast")

SAVED_DIR = Path("saved_model")
CSV_PATH = Path("memory_data.csv")
PRED_CSV_PATH = Path("memory_forecasts.csv")
SAMPLE_INTERVAL = 0.5  # seconds between samples
FUTURE_STEPS = 5       # steps to forecast ahead
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Helpers
# -------------------------
def append_row_to_csv(csv_path: Path, row: Dict):
    """Append a single-row dict to a CSV (create file with header if missing)."""
    df = pd.DataFrame([row])
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", index=False, header=header)


# -------------------------
# Model loader
# -------------------------
class NBeatsRealtimeWrapper:
    """Wrap loaded N-Beats model and scaling info for forecasting."""

    def __init__(self, save_dir: str = "saved_model", device: Optional[torch.device] = None):
        self.save_dir = Path(save_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if _HAS_NBEATS_UTILS:
            self.model, self.scaling_info, self.config = load_nbeats_model(str(self.save_dir), device=self.device)
        else:
            raise RuntimeError("nbeats_utils.py with load_nbeats_model is required")

        # Always store backcast/forecast lengths
        self.backcast_length = self.config.get("backcast_length", getattr(self.model, "backcast_length", None))
        self.forecast_length = self.config.get("forecast_length", getattr(self.model, "forecast_length", None))

    def recursive_forecast(self, input_series: np.ndarray, future_steps: int = FUTURE_STEPS) -> np.ndarray:
        """Forecast future values using forecast_future."""
        if len(input_series) < self.backcast_length:
            raise ValueError(f"Need at least backcast_length={self.backcast_length} points to forecast. Got {len(input_series)}")
        return forecast_future(self.model, self.scaling_info, input_series, device=self.device)


# -------------------------
# Live Data Collector
# -------------------------
class LiveCollector:
    """Collect system metrics and append to CSV."""

    def __init__(self, csv_path: Path = CSV_PATH):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            pd.DataFrame(columns=[
                "timestamp", "memory_percent", "available_gb", "used_gb", "total_gb",
                "cpu_percent", "battery_percent",
                "process_count", "top1_cpu", "top2_cpu", "top3_cpu"
            ]).to_csv(self.csv_path, index=False)

    def collect(self) -> Dict:
        try:
            mem = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=None)
            battery = psutil.sensors_battery()
            battery_percent = battery.percent if battery else None
            proc_count = len(psutil.pids())

            processes = []
            for p in psutil.process_iter(["pid", "name", "cpu_percent"]):
                try:
                    info = p.info
                    info["cpu_percent"] = info.get("cpu_percent") or 0.0
                    processes.append(info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            processes.sort(key=lambda x: x.get("cpu_percent", 0.0), reverse=True)
            t1 = processes[0]["cpu_percent"] if len(processes) > 0 else 0.0
            t2 = processes[1]["cpu_percent"] if len(processes) > 1 else 0.0
            t3 = processes[2]["cpu_percent"] if len(processes) > 2 else 0.0

            row = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "memory_percent": float(mem.percent),
                "available_gb": mem.available / (1024**3),
                "used_gb": mem.used / (1024**3),
                "total_gb": mem.total / (1024**3),
                "cpu_percent": float(cpu_percent),
                "battery_percent": float(battery_percent) if battery_percent is not None else None,
                "process_count": int(proc_count),
                "top1_cpu": float(t1),
                "top2_cpu": float(t2),
                "top3_cpu": float(t3),
            }

            append_row_to_csv(self.csv_path, row)
            return row

        except Exception:
            logger.exception("collect error")
            return {
                "timestamp": pd.Timestamp.now().isoformat(),
                "memory_percent": 0.0,
                "cpu_percent": 0.0,
                "battery_percent": None,
                "process_count": 0,
                "top1_cpu": 0.0,
                "top2_cpu": 0.0,
                "top3_cpu": 0.0,
            }


# -------------------------
# Main real-time loop
# -------------------------
def main():
    try:
        wrapper = NBeatsRealtimeWrapper(save_dir=str(SAVED_DIR), device=DEVICE)
    except Exception:
        logger.exception("Failed to load model from saved_model/")
        return

    collector = LiveCollector(csv_path=CSV_PATH)
    mem_buffer = deque(maxlen=max(2000, wrapper.backcast_length * 3))

    if CSV_PATH.exists():
        try:
            df_hist = pd.read_csv(CSV_PATH)
            if "memory_percent" in df_hist.columns:
                for v in df_hist["memory_percent"].astype(float).values:
                    mem_buffer.append(v)
            logger.info(f"Preloaded {len(mem_buffer)} samples from {CSV_PATH}")
        except Exception:
            logger.exception("Failed to preload CSV history")

    print("Starting real-time forecasting. Press Ctrl+C to stop.")
    try:
        while True:
            sample = collector.collect()
            mem = float(sample["memory_percent"])
            mem_buffer.append(mem)

            if len(mem_buffer) >= wrapper.backcast_length:
                series = np.array(mem_buffer)
                try:
                    forecast = wrapper.recursive_forecast(series, future_steps=FUTURE_STEPS)
                except Exception:
                    logger.exception("Forecasting failed")
                    forecast = None

                if forecast is not None:
                    ts = pd.Timestamp.now().isoformat()
                    append_row_to_csv(PRED_CSV_PATH, {
                        "timestamp": ts,
                        "current_memory": mem,
                        "forecast_steps": FUTURE_STEPS,
                        "forecast_values": json.dumps(forecast.tolist())
                    })
                    logger.info(f"Now {mem:.2f}% | Forecast next {FUTURE_STEPS} (avg={np.mean(forecast):.2f}%)")
                else:
                    logger.info(f"Now {mem:.2f}% | Forecast failed")
            else:
                logger.info(f"Now {mem:.2f}% | Waiting for {wrapper.backcast_length - len(mem_buffer)} more samples")

            time.sleep(SAMPLE_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopping real-time forecasting.")
    except Exception:
        logger.exception("Unexpected error in main loop")


# -------------------------
# Adapters for interface.py
# -------------------------
class MemoryForecaster:
    """Adapter for interface.py compatibility."""

    def __init__(self, save_dir: str = "saved_model", seq_length: int = 20, future_steps: int = FUTURE_STEPS, device: Optional[torch.device] = None):
        self.wrapper = NBeatsRealtimeWrapper(save_dir=save_dir, device=device or DEVICE)
        self.future_steps = future_steps
        maxlen = max(2000, int(self.wrapper.backcast_length) * 3 if self.wrapper.backcast_length else seq_length * 3)
        self.memory_buffer = deque(maxlen=maxlen)
        self.predictions_history: List[float] = []

    def add_data_point(self, memory_percent: float):
        try:
            self.memory_buffer.append(float(memory_percent))
        except Exception:
            pass

    def predict_next(self) -> Dict[str, Optional[float]]:
        try:
            if self.wrapper.backcast_length and len(self.memory_buffer) < self.wrapper.backcast_length:
                return {"prediction": None}
            series = np.array(self.memory_buffer, dtype=np.float32)
            forecast = self.wrapper.recursive_forecast(series, future_steps=self.future_steps)
            if forecast is None or len(forecast) == 0:
                return {"prediction": None}
            pred_val = float(forecast[-1])
            self.predictions_history.append(pred_val)
            if len(self.predictions_history) > 1000:
                self.predictions_history = self.predictions_history[-500:]
            return {"prediction": pred_val}
        except Exception:
            logger.exception("predict_next error")
            return {"prediction": None}

    def get_forecast_statistics(self) -> Optional[Dict[str, float]]:
        if not self.predictions_history:
            return None
        arr = np.array(self.predictions_history, dtype=np.float32)
        trend = "flat"
        if len(arr) >= 5:
            first, last = float(arr[-5]), float(arr[-1])
            if last - first > 0.5:
                trend = "up"
            elif first - last > 0.5:
                trend = "down"
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "trend": trend,
        }


class LiveDataCollector:
    """Adapter that proxies to LiveCollector."""

    def __init__(self, csv_path: str = None):
        self.collector = LiveCollector(csv_path=Path(csv_path) if csv_path else CSV_PATH)

    def collect(self) -> Dict:
        return self.collector.collect()


if __name__ == "__main__":
    main()
