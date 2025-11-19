"""
Process Anomaly Detection using Isolation Forest
Basic ML approach for identifying suspicious processes based on historical behavior patterns.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, deque
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import psutil
import threading
import pickle
from pathlib import Path
import logging


class ProcessAnomalyDetector:
    """
    ML-powered process anomaly detection using Isolation Forest.

    Features monitored:
    - CPU usage patterns (rolling means, std)
    - Memory usage patterns
    - Disk I/O patterns
    - Network activity
    - Process lifecycle metrics
    """

    def __init__(self, model_dir="process_anomaly_models", training_duration_days=3, test_mode=False, debug_level=0):
        self.model_dir = model_dir
        self.test_mode = test_mode
        self.training_duration_days = training_duration_days
        self.debug_level = debug_level  # 0=none, 1=basic, 2=detailed

        # Data collection
        self.process_history = defaultdict(lambda: deque(maxlen=1000))  # Per-process data
        self.training_data = []  # Collected training samples
        self.is_training = False
        self.training_start_time = None

        # Debug counters and tracking
        self.debug_stats = {
            'samples_collected': 0,
            'processes_monitored': set(),
            'models_trained': 0,
            'detection_calls': 0,
            'anomalies_detected': 0
        }

        # Model storage
        self.models = {}  # Process name -> (scaler, model) tuple
        self.feature_counts = {}

        # Threading
        self.collection_thread = None
        self.stop_collection = False

        # Configuration
        self.config = {
            "contamination": 0.05,  # Expected anomaly ratio
            "n_estimators": 100,    # Number of trees in Isolation Forest
            "max_features": 8,      # Max features to consider per split
            "window_size": 20,      # Rolling window for feature calculation
            "score_threshold": 0.35  # Anomaly score threshold (0-1)
        }

        # Initialize model directory
        os.makedirs(model_dir, exist_ok=True)

        if self.debug_level >= 1:
            print("🔍 ANOMALY DEBUG: Initializing ProcessAnomalyDetector")
            print(f"🔍 ANOMALY DEBUG: Debug level: {self.debug_level} (0=none, 1=basic, 2=detailed)")
            print(f"🔍 ANOMALY DEBUG: Model directory: {self.model_dir}")

        # Load existing models if available
        self.load_models()

    def start_training(self):
        """Start background collection of normal process behavior data"""
        if self.is_training:
            print("Training already in progress")
            return

        self.is_training = True
        self.training_start_time = datetime.now()
        self.training_data = []

        print(f"Starting process anomaly training ({self.training_duration_days} days)...")

        self.collection_thread = threading.Thread(target=self._collect_training_data, daemon=True)
        self.collection_thread.start()

    def start_quick_test(self, test_minutes=5):
        """Start quick test training for anomaly detection (minutes instead of days)"""
        if self.is_training:
            print("Training already in progress")
            return

        # Set test mode and convert minutes to days for internal consistency
        self.test_mode = True
        self.training_duration_days = test_minutes / (24 * 60)  # Convert minutes to days

        self.is_training = True
        self.training_start_time = datetime.now()
        self.training_data = []

        print(f"🧪 Starting ENHANCED QUICK TEST training for anomaly detection ({test_minutes} minutes)...")
        print(f"⚡ STRESS-AWARE MODE: Now includes automatic stress-ng testing during training!")
        print(f"🔥 System will learn both normal AND anomalous (high-CPU) patterns.")
        print(f"🏃 You'll get ~{int(test_minutes * 6)} samples including stress patterns.")
        print(f"⏰ Training will automatically stop in {test_minutes} minutes.")

        # Add timeout wrapper for the entire training process
        def training_with_timeout():
            try:
                self._collect_training_data_with_stress()
                self.finish_training()
            except Exception as e:
                print(f"❌ Training failed with error: {e}")

        self.collection_thread = threading.Thread(target=training_with_timeout, daemon=True)
        self.collection_thread.start()

    def finish_training(self):
        """Finish training process and save models"""
        print("🏁 Finishing training process...")
        self.is_training = False

        # Stop collection thread gracefully
        if self.collection_thread and self.collection_thread.is_alive():
            self.stop_collection = True
            # DON'T join the current thread (causes deadlock in quick tests)
            if not getattr(self, 'test_mode', False):
                self.collection_thread.join(timeout=10)

        print(f"📊 Training data collected: {len(self.training_data)} samples")
        self._train_models()
        self.save_models()

        # 🔄 AUTO-RELOAD newly trained models
        print("🔄 Reloading trained models...")
        self.reload_trained_models()

    def reload_trained_models(self):
        """Reload models from disk after training completes"""
        self.models = {}  # Clear current models
        self.load_models()  # Reload from disk

        loaded_count = len(self.models)
        if loaded_count > 0:
            print(f"✅ Successfully loaded {loaded_count} trained models")
        else:
            print("⚠️ No models loaded - check if training completed successfully")

    def stop_training(self):
        """Stop training and train models"""
        if not self.is_training:
            print("No training in progress")
            return

        self.is_training = False
        self.stop_collection = True

        # Wait for collection to stop
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join()

        # Train models on collected data
        self._train_models()
        self.save_models()

        print(f"Training completed. Collected {len(self.training_data)} samples across {len(set([d['name'] for d in self.training_data]))} processes.")

    def _collect_training_data(self):
        """Background thread to collect normal process behavior data"""
        print("Process anomaly monitoring started. Go about normal computer usage...")
        if self.test_mode:
            print(f"⚡ TEST MODE: Training will run for {self.training_duration_days * 24 * 60:.0f} minutes (simulated in fast time).")
            # For testing, convert days to minutes (1 day = 1440 minutes)
            total_minutes = self.training_duration_days * 24 * 60
            end_time = datetime.now() + timedelta(minutes=total_minutes)
        else:
            print(f"Training will run for {self.training_duration_days} days.")
            end_time = datetime.now() + timedelta(days=self.training_duration_days)

        sample_count = 0
        last_debug_time = datetime.now()

        while datetime.now() < end_time and not self.stop_collection:
            try:
                # Collect current process data
                timestamp = datetime.now()

                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                    try:
                        proc_info = proc.info
                        if not proc_info or proc_info['cpu_percent'] is None:
                            continue

                        pid = proc_info['pid']
                        name = proc_info['name']

                        # Debug: Skip system processes during training
                        if name in ['System', 'kernel_task', 'sshd', 'cron', 'systemd', 'launchd']:
                            if self.debug_level >= 2:
                                print(f"🔍 ANOMALY DEBUG: Skipping system process '{name}'")
                            continue

                        # Debug: Track unique processes
                        if name not in self.debug_stats['processes_monitored']:
                            self.debug_stats['processes_monitored'].add(name)
                            if self.debug_level >= 1:
                                print(f"🆕 ANOMALY DEBUG: Discovered new process '{name}' (total: {len(self.debug_stats['processes_monitored'])})")

                        # Collect additional metrics
                        try:
                            p = psutil.Process(pid)
                            cpu_times = p.cpu_times()
                            memory_info = p.memory_info()
                            io_counters = p.io_counters()

                            sample = {
                                'timestamp': timestamp,
                                'pid': pid,
                                'name': name,
                                'cpu_percent': round(proc_info['cpu_percent'], 2),
                                'memory_percent': round(proc_info['memory_percent'], 2),
                                'cpu_user': round(cpu_times.user, 2) if cpu_times else 0,
                                'cpu_system': round(cpu_times.system, 2) if cpu_times else 0,
                                'memory_rss': memory_info.rss // (1024 * 1024) if memory_info else 0,  # MB
                                'memory_vms': memory_info.vms // (1024 * 1024) if memory_info else 0,  # MB
                                'read_bytes': io_counters.read_bytes // (1024 * 1024) if io_counters else 0,  # MB
                                'write_bytes': io_counters.write_bytes // (1024 * 1024) if io_counters else 0,  # MB
                            }

                            # Add to history for feature calculation
                            self.process_history[name].append(sample)

                            # Store training sample
                            self.training_data.append(sample)
                            self.debug_stats['samples_collected'] += 1
                            sample_count += 1

                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            if self.debug_level >= 2:
                                print(f"🚨 ANOMALY DEBUG: Access denied for process {pid} '{name}'")
                            continue

                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                # Periodic debug output
                if self.debug_level >= 1 and (datetime.now() - last_debug_time).seconds >= 60:  # Every minute
                    print(f"📊 ANOMALY DEBUG: Collection progress - Samples: {self.debug_stats['samples_collected']}, "
                          f"Processes: {len(self.debug_stats['processes_monitored'])}, "
                          f"Memory: {len(self.training_data)} items")
                    last_debug_time = datetime.now()

                time.sleep(10)  # Sample every 10 seconds during training

            except Exception as e:
                print(f"Training data collection error: {e}")
                if self.debug_level >= 2:
                    print(f"🚨 ANOMALY DEBUG: Collection exception details: {str(e)}")
                time.sleep(5)

        if self.debug_level >= 1:
            print(f"✅ ANOMALY DEBUG: Collection complete - Total samples: {self.debug_stats['samples_collected']}, "
                  f"Processes monitored: {len(self.debug_stats['processes_monitored'])}")

        print("Training data collection completed.")

    def _collect_training_data_with_stress(self):
        """Enhanced training collection that includes stress-ng stress testing"""
        print("🎯 STRESS-AWARE TRAINING: Collecting both normal AND anomalous process behavior!")
        print("   This will create models that properly distinguish stress patterns.")
        print()

        import subprocess
        stress_processes = []  # Track active stress processes

        # For 5-minute test, structure as:
        # - First 2 minutes: Normal behavior only
        # - Minutes 2-4: Mix normal + stress patterns
        # - Final minute: Normal behavior again

        total_minutes = self.training_duration_days * 24 * 60  # Convert back to minutes
        end_time = datetime.now() + timedelta(minutes=total_minutes)

        print(f"⏰ Training schedule:")
        print(f"   0-2 min: Normal behavior baseline")
        print(f"   2-4 min: Mixed normal + stress patterns")
        print(f"   4-5 min: Normal behavior cooldown")

        sample_count = 0
        last_debug_time = datetime.now()
        stress_active = False
        stress_start_time = None

        while datetime.now() < end_time and not self.stop_collection:
            try:
                timestamp = datetime.now()
                elapsed_minutes = (timestamp - self.training_start_time).total_seconds() / 60

                # Stress schedule logic
                should_have_stress = 2.0 <= elapsed_minutes <= 4.0  # 2-4 minutes

                if should_have_stress and not stress_active:
                    # Start stress
                    print("🔥 LAUNCHING STRESS-NG: Creating anomalous training patterns...")
                    try:
                        stress_proc = subprocess.Popen([
                            'stress-ng', '--cpu', '2', '--cpu-method', 'fft',
                            '--timeout', '120', '--quiet'  # 2 minutes of stress
                        ])
                        stress_processes.append(stress_proc)
                        stress_active = True
                        stress_start_time = datetime.now()
                        print("   ✅ Stress-ng launched - learning anomalous CPU patterns!")
                    except FileNotFoundError:
                        print("   ❌ stress-ng not found - using Python fallback stress")
                        try:
                            # Python CPU stress fallback
                            stress_proc = subprocess.Popen(['python3', '-c', '''
import time
start = time.time()
while time.time() - start < 120:  # 2 minutes
    for i in range(100000):
        _ = i * i  # CPU work
'''])
                            stress_processes.append(stress_proc)
                            stress_active = True
                            stress_start_time = datetime.now()
                            print("   ✅ Python CPU stress launched!")
                        except Exception as e:
                            print(f"   ❌ Python stress failed: {e}")

                elif not should_have_stress and stress_active:
                    # Stop stress
                    print("🛑 STOPPING STRESS: Returning to normal patterns...")
                    for proc in stress_processes:
                        try:
                            proc.terminate()
                            proc.wait(timeout=5)
                        except:
                            try:
                                proc.kill()
                            except:
                                pass
                    stress_processes.clear()
                    stress_active = False
                    print("   ✅ Normal behavior training resumed.")

                # Clean up any finished stress processes
                stress_processes = [p for p in stress_processes if p.poll() is None]

                # Collect current process data
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                    try:
                        proc_info = proc.info
                        if not proc_info or proc_info['cpu_percent'] is None:
                            continue

                        pid = proc_info['pid']
                        name = proc_info['name']

                        # Skip system processes during training
                        if name in ['System', 'kernel_task', 'sshd', 'cron', 'systemd', 'launchd']:
                            continue

                        # Collect additional metrics
                        try:
                            p = psutil.Process(pid)
                            cpu_times = p.cpu_times()
                            memory_info = p.memory_info()
                            io_counters = p.io_counters()

                            sample = {
                                'timestamp': timestamp,
                                'pid': pid,
                                'name': name,
                                'cpu_percent': round(proc_info['cpu_percent'], 2),
                                'memory_percent': round(proc_info['memory_percent'], 2),
                                'cpu_user': round(cpu_times.user, 2) if cpu_times else 0,
                                'cpu_system': round(cpu_times.system, 2) if cpu_times else 0,
                                'memory_rss': memory_info.rss // (1024 * 1024) if memory_info else 0,
                                'memory_vms': memory_info.vms // (1024 * 1024) if memory_info else 0,
                                'read_bytes': io_counters.read_bytes // (1024 * 1024) if io_counters else 0,
                                'write_bytes': io_counters.write_bytes // (1024 * 1024) if io_counters else 0,
                            }

                            # Add to history for feature calculation
                            self.process_history[name].append(sample)

                            # Store training sample
                            self.training_data.append(sample)
                            self.debug_stats['samples_collected'] += 1
                            sample_count += 1

                            # Debug: Track unique processes
                            if name not in self.debug_stats['processes_monitored']:
                                self.debug_stats['processes_monitored'].add(name)
                                phase = "STRESS" if stress_active else "NORMAL"
                                print(f"🆕 [{phase}] Discovered process: {name}")

                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue

                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                # Periodic debug output
                if self.debug_level >= 1 and (datetime.now() - last_debug_time).seconds >= 30:  # Every 30 seconds
                    phase = "🔥 STRESS PHASE" if stress_active else "📊 NORMAL PHASE"
                    print(f"{phase}: Samples: {self.debug_stats['samples_collected']}, "
                          f"Processes: {len(self.debug_stats['processes_monitored'])}, "
                          f"Elapsed: {elapsed_minutes:.1f}min")
                    last_debug_time = datetime.now()

                time.sleep(8)  # Sample every 8 seconds for better coverage

            except Exception as e:
                print(f"Training data collection error: {e}")
                if self.debug_level >= 2:
                    print(f"🚨 Collection exception details: {str(e)}")
                time.sleep(5)

        # Clean up stress processes
        print("🧹 Cleaning up stress processes...")
        for proc in stress_processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                try:
                    proc.kill()
                except:
                    pass

        if self.debug_level >= 1:
            print(f"✅ STRESS-AWARE TRAINING COMPLETE")
            print(f"   Total samples: {self.debug_stats['samples_collected']}")
            print(f"   Processes monitored: {len(self.debug_stats['processes_monitored'])}")
            print(f"   Models will now recognize stress-ng patterns as anomalous!")

        print("Enhanced training data collection completed with stress pattern integration!")

    def _train_models(self):
        """Train Isolation Forest models on collected data"""
        print("Training anomaly detection models...")

        if self.debug_level >= 1:
            print(f"🏗️ MODEL DEBUG: Starting training with {len(self.training_data)} total samples")

        # Group data by process name
        process_data = defaultdict(list)
        for sample in self.training_data:
            process_data[sample['name']].append(sample)

        if self.debug_level >= 2:
            print(f"🔍 MODEL DEBUG: Grouped data by process - {len(process_data)} unique processes found")

        # Train one model per process (if we have enough data)
        self.models = {}
        total_samples = 0
        skipped_processes = 0

        # Adjust minimum samples for test mode
        min_samples = 10 if self.test_mode else 100  # Much lower for quick tests

        for process_name, samples in process_data.items():
            if len(samples) < min_samples:  # Need minimum samples
                if self.debug_level >= 1:
                    print(f"⚠️ MODEL DEBUG: Skipping {process_name} - insufficient data ({len(samples)} samples, need {min_samples})")
                skipped_processes += 1
                continue

            if self.debug_level >= 1:
                print(f"🏗️ MODEL DEBUG: Training model for {process_name} ({len(samples)} samples)")

            total_samples += len(samples)

            # Extract features
            features = self._extract_features_batch(samples)

            if self.debug_level >= 2:
                print(f"🔬 FEATURE DEBUG: {process_name} - Extracted {len(features)} feature vectors with {features.shape[1]} features each")

            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Train Isolation Forest
            model = IsolationForest(
                n_estimators=self.config['n_estimators'],
                contamination=self.config['contamination'],
                max_features=self.config['max_features'],
                random_state=42
            )

            model.fit(features_scaled)

            # Store model and scaler
            self.models[process_name] = (scaler, model)
            self.debug_stats['models_trained'] += 1

            if self.debug_level >= 2:
                # Test the model on its own training data to show scores
                test_scores = model.decision_function(features_scaled)
                anomaly_ratio = (test_scores < 0).mean() * 100  # Isolation Forest scores < 0 are outliers
                print(f"✅ MODEL DEBUG: {process_name} training complete - {anomaly_ratio:.1f}% samples classified as anomalous in training")

        if self.debug_level >= 1:
            print(f"📊 MODEL DEBUG: Trained {len(self.models)} process models on {total_samples} total samples")
            if skipped_processes > 0:
                print(f"⚠️ MODEL DEBUG: Skipped {skipped_processes} processes due to insufficient training data")

        print(f"Trained {len(self.models)} process models on {total_samples} total samples.")

        # Also train a general model for unrecognized processes
        if len(self.training_data) > 500:
            if self.debug_level >= 1:
                print(f"🏗️ MODEL DEBUG: Training general fallback model for unrecognized processes")

            all_features = self._extract_features_batch(self.training_data)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(all_features)

            model = IsolationForest(
                n_estimators=self.config['n_estimators'],
                contamination=self.config['contamination'],
                max_features=min(self.config['max_features'], features_scaled.shape[1]),
                random_state=42
            )

            model.fit(features_scaled)
            self.models['__general__'] = (scaler, model)
            self.debug_stats['models_trained'] += 1

            if self.debug_level >= 2:
                print(f"✅ MODEL DEBUG: General model trained on {len(all_features)} samples with {features_scaled.shape[1]} features")

    def _extract_features_batch(self, samples):
        """Extract features from a batch of process samples"""
        if len(samples) < self.config['window_size'] + 10:
            # Not enough data for rolling features, use simple stats
            cpu_values = [s['cpu_percent'] for s in samples]
            mem_values = [s['memory_percent'] for s in samples]

            return np.array([[
                np.mean(cpu_values), np.std(cpu_values), np.max(cpu_values),
                np.mean(mem_values), np.std(mem_values), np.max(mem_values),
                len(set(cpu_values)), len(set(mem_values))  # Feature diversity
            ]])

        # Use sliding windows for feature extraction
        features = []
        window_size = self.config['window_size']

        for i in range(window_size, len(samples), window_size // 2):
            window = samples[i-window_size:i]

            cpu_vals = [s['cpu_percent'] for s in window]
            mem_vals = [s['memory_percent'] for s in window]

            # Statistical features
            features.append([
                np.mean(cpu_vals), np.std(cpu_vals), np.min(cpu_vals), np.max(cpu_vals),
                np.percentile(cpu_vals, 75) - np.percentile(cpu_vals, 25),  # IQR
                np.mean(mem_vals), np.std(mem_vals), np.min(mem_vals), np.max(mem_vals),
                np.percentile(mem_vals, 75) - np.percentile(mem_vals, 25),
                self._calc_trend_slope(cpu_vals),
                self._calc_trend_slope(mem_vals),
                self._calc_volatility(cpu_vals),
                self._calc_volatility(mem_vals)
            ])

        return np.array(features)

    def _calc_trend_slope(self, values):
        """Calculate slope of linear trend"""
        if len(values) < 3:
            return 0
        x = np.arange(len(values))
        return np.polyfit(x, values, 1)[0] if len(set(values)) > 1 else 0

    def _calc_volatility(self, values):
        """Calculate coefficient of variation (relative volatility)"""
        if len(values) == 0 or np.mean(values) == 0:
            return 0
        return np.std(values) / np.mean(values)

    def get_anomaly_score(self, process_name, current_metrics, recent_history=None):
        """
        Get anomaly score for a process (0-1, where 1 is most anomalous)

        Args:
            process_name: Name of the process
            current_metrics: Dict with current process metrics
            recent_history: List of recent metric samples (optional)

        Returns:
            float: Anomaly score (0-1)
        """
        if process_name not in self.models:
            # Try general model
            if '__general__' not in self.models:
                return 0.0  # No model available
            model_name = '__general__'
        else:
            model_name = process_name

        scaler, model = self.models[model_name]

        # Extract features from current metrics
        features = self._extract_current_features(process_name, current_metrics, recent_history)

        if features is None:
            return 0.0  # Not enough data

        # Scale and score
        features_scaled = scaler.transform([features])
        scores = model.decision_function(features_scaled)

        # Convert to 0-1 range (Isolation Forest decision function can be negative)
        # Scores closer to 1 are normal, closer to -1 are anomalous
        anomaly_score = (1 - (scores[0] + 1) / 2)  # Convert to 0-1 where 1 is anomalous

        return max(0, min(1, anomaly_score))

    def _extract_current_features(self, process_name, current_metrics, recent_history=None):
        """Extract features for current process state"""
        # Use recent history if available and sufficient
        if recent_history and len(recent_history) >= 3:
            if self.debug_level >= 2:
                print(f"🔍 FEATURE DEBUG: Using recent_history path ({len(recent_history)} items)")
            return self._extract_features_batch(recent_history)[0]

        # Extract available historical data from process history
        if process_name in self.process_history and len(self.process_history[process_name]) >= 3:
            recent_data = list(self.process_history[process_name])[-min(len(self.process_history[process_name]), self.config['window_size']):]
            if self.debug_level >= 2:
                print(f"🔍 FEATURE DEBUG: Using process_history path ({len(recent_data)} items from history)")

            # Create batch sample for feature extraction
            current_sample = current_metrics.copy()
            current_sample['timestamp'] = datetime.now()
            current_sample['pid'] = current_metrics.get('pid', 0)
            current_sample['name'] = process_name
            batch_samples = recent_data + [current_sample]
            batch_samples = batch_samples[-self.config['window_size']:]  # Limit to window size

            return self._extract_features_batch(batch_samples)[0]

        # Fallback: single point with statistical defaults (must match 14 features expected by trained models)
        if self.debug_level >= 1:
            print(f"🔍 FEATURE DEBUG: Using fallback single-point path for {process_name}")

        cpu_val = current_metrics.get('cpu_percent', 0)
        mem_val = current_metrics.get('memory_percent', 0)

        features = np.array([
            cpu_val,  # mean cpu
            0.0,      # std cpu (unknown)
            cpu_val,  # min cpu (single point)
            cpu_val,  # max cpu (single point)
            0.0,      # IQR cpu (single point)
            mem_val,  # mean mem
            0.0,      # std mem (unknown)
            mem_val,  # min mem (single point)
            mem_val,  # max mem (single point)
            0.0,      # IQR mem (single point)
            0.0,      # cpu trend slope (single point)
            0.0,      # mem trend slope (single point)
            0.0,      # cpu volatility (single point)
            0.0       # mem volatility (single point)
        ])

        if self.debug_level >= 2:
            print(f"🔍 FEATURE DEBUG: Generated {len(features)} features: {features}")

        return features

    def update_process_history(self, process_name, metrics):
        """Update historical data for a process"""
        metrics_with_timestamp = metrics.copy()
        metrics_with_timestamp['timestamp'] = datetime.now()
        self.process_history[process_name].append(metrics_with_timestamp)

    def is_anomalous(self, process_name, current_metrics, recent_history=None):
        """Check if process behavior is anomalous"""
        score = self.get_anomaly_score(process_name, current_metrics, recent_history)
        return score > self.config['score_threshold']

    def save_models(self):
        """Save trained models to disk"""
        try:
            data = {
                'models': {},
                'config': self.config,
                'training_metadata': {
                    'trained_processes': list(self.models.keys()),
                    'training_samples': len(self.training_data),
                    'training_duration_days': self.training_duration_days
                }
            }

            for name, (scaler, model) in self.models.items():
                model_data = {
                    'scaler_params': {
                        'mean_': scaler.mean_.tolist(),
                        'var_': scaler.var_.tolist(),
                        'scale_': scaler.scale_.tolist()
                    },
                    'model_params': model.get_params(),
                    'feature_importance': getattr(model, 'feature_importances_', None)
                }

                # Save model using pickle
                filename = os.path.join(self.model_dir, f"{name.replace('/', '_')}_model.pkl")
                with open(filename, 'wb') as f:
                    pickle.dump({'scaler': scaler, 'model': model, 'data': model_data}, f)

            filename = os.path.join(self.model_dir, 'anomaly_detector_config.json')
            with open(filename, 'w') as f:
                json.dump(data['config'], f, indent=2)

            print(f"Saved {len(self.models)} trained models.")

        except Exception as e:
            print(f"Error saving models: {e}")

    def load_models(self):
        """Load trained models from disk"""
        try:
            config_file = os.path.join(self.model_dir, 'anomaly_detector_config.json')
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    self.config.update(json.load(f))

            # Load individual process models
            for filename in os.listdir(self.model_dir):
                if filename.endswith('_model.pkl'):
                    try:
                        filepath = os.path.join(self.model_dir, filename)
                        with open(filepath, 'rb') as f:
                            data = pickle.load(f)
                            scaler, model = data['scaler'], data['model']

                            process_name = filename.replace('_model.pkl', '').replace('_', '/')
                            self.models[process_name] = (scaler, model)
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")

            print(f"Loaded {len(self.models)} pre-trained models.")

        except Exception as e:
            print(f"Error loading models: {e}")

    def get_training_status(self):
        """Get current training status"""
        if not self.is_training:
            return "Not training"

        elapsed = datetime.now() - self.training_start_time
        remaining = timedelta(days=self.training_duration_days) - elapsed

        if remaining.total_seconds() <= 0:
            return "Training complete"
        else:
            hours = int(remaining.total_seconds() // 3600)
            minutes = int((remaining.total_seconds() % 3600) // 60)
            return f"Training: {hours}h {minutes}m remaining"

    def reset_models(self):
        """Clear all trained models"""
        self.models = {}
        import shutil
        shutil.rmtree(self.model_dir, ignore_errors=True)
        os.makedirs(self.model_dir, exist_ok=True)
        print("All models cleared.")


# -------------------------
# Enhanced Training Data Collection Pipeline
# -------------------------
class TrainingDataCollector:
    """
    Dedicated pipeline for collecting high-quality training data for process anomaly detection.

    Features:
    - Continuous background collection
    - Data quality validation
    - Automatic cleanup of corrupted samples
    - Export/import training datasets
    - Balanced sampling across different system conditions
    - Training data visualization and statistics
    """

    def __init__(self, data_dir="training_data", max_samples_per_process=5000):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.max_samples_per_process = max_samples_per_process
        self.is_collecting = False
        self.collection_start_time = None
        self.collection_thread = None

        # Data storage
        self.training_samples = defaultdict(list)  # Process name -> list of samples
        self.system_conditions = []  # Track system state changes
        self.corrupted_samples = 0

        # Quality thresholds
        self.quality_filters = {
            'cpu_min': 0.1,
            'cpu_max': 200.0,
            'memory_min': 0.1,
            'memory_max': 150.0,
            'min_process_age': 30,  # seconds
            'max_collection_rate': 10  # samples per second
        }

        # Collection statistics
        self.stats = {
            'total_collected': 0,
            'quality_filtered': 0,
            'processes_discovered': set(),
            'collection_duration': 0,
            'last_collection_time': None
        }

        logging.basicConfig(
            filename=self.data_dir / 'collection.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def start_collection(self, duration_minutes=60):
        """Start training data collection"""
        if self.is_collecting:
            print("Collection already running")
            return

        self.is_collecting = True
        self.collection_start_time = datetime.now()
        self.stats['last_collection_time'] = self.collection_start_time

        print(f"🎯 Started training data collection")
        print(f"⏱️ Duration: {duration_minutes} minutes")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"🔍 Quality filters active: CPU {self.quality_filters['cpu_min']:.1f}%-{self.quality_filters['cpu_max']:.1f}%, "
              f"Memory {self.quality_filters['memory_min']:.1f}%-{self.quality_filters['memory_max']:.1f}%")
        self.logger.info(f"Started collection for {duration_minutes} minutes")

        self.collection_thread = threading.Thread(target=self._collect_data, args=(duration_minutes,), daemon=True)
        self.collection_thread.start()

    def stop_collection(self):
        """Stop training data collection"""
        if not self.is_collecting:
            return

        print("🛑 Stopping training data collection...")
        self.is_collecting = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join()

        duration = datetime.now() - self.collection_start_time
        self.stats['collection_duration'] = duration.total_seconds()

        self.save_collected_data()
        self.logger.info(f"Collection stopped. Duration: {duration}")

        print(f"✅ Collection complete!")
        print(f"📊 Samples collected: {self.stats['total_collected']}")
        print(f"🧹 Samples filtered (quality): {self.stats['quality_filtered']}")
        print(f"🔍 Unique processes: {len(self.stats['processes_discovered'])}")
        print(f"💾 Data saved to: {self.data_dir}")

    def _collect_data(self, duration_minutes):
        """Main data collection loop with quality control"""
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        last_sample_time = datetime.now()
        sample_count = 0

        print(f"🏁 Collection started. Target: {duration_minutes} minutes")

        while datetime.now() < end_time and self.is_collecting:
            try:
                current_time = datetime.now()

                # Rate limiting: don't collect too frequently
                time_since_last = (current_time - last_sample_time).total_seconds()
                if time_since_last < (1.0 / self.quality_filters['max_collection_rate']):
                    time.sleep(0.1)
                    continue

                # Collect current system state
                system_state = self._get_system_state()
                self.system_conditions.append({
                    'timestamp': current_time,
                    'state': system_state
                })

                # Collect process data
                collected_this_round = 0

                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                    try:
                        proc_info = proc.info
                        if not self._passes_quality_filters(proc_info):
                            self.stats['quality_filtered'] += 1
                            continue

                        pid = proc_info['pid']
                        name = proc_info['name']

                        # Skip system processes
                        if name in ['System', 'kernel_task', 'sshd', 'cron', 'systemd', 'launchd']:
                            continue

                        # Check per-process sample limit
                        if len(self.training_samples[name]) >= self.max_samples_per_process:
                            continue

                        # Collect comprehensive metrics
                        sample = self._collect_process_metrics(pid, name, proc_info)
                        if sample:
                            self.training_samples[name].append(sample)
                            collected_this_round += 1
                            self.stats['total_collected'] += 1
                            self.stats['processes_discovered'].add(name)

                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        self.corrupted_samples += 1

                last_sample_time = current_time
                sample_count += 1

                # Progress logging
                if sample_count % 60 == 0:  # Every minute
                    elapsed = datetime.now() - self.collection_start_time
                    remaining = end_time - datetime.now()
                    print(f"📊 {elapsed.total_seconds()//60:.0f}m elapsed | Collected: {self.stats['total_collected']} | "
                          f"Processes: {len(self.stats['processes_discovered'])} | Remaining: {remaining.total_seconds()//60:.0f}m")

                time.sleep(1)  # Sample every second

            except Exception as e:
                self.logger.error(f"Collection error: {e}")
                time.sleep(2)

        print("🏁 Collection loop ended naturally.")

    def _get_system_state(self):
        """Get current system state for sampling diversity"""
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_percent = psutil.virtual_memory().percent

        # Classify system state
        if cpu_percent > 80 or memory_percent > 85:
            return "high_load"
        elif cpu_percent > 50 or memory_percent > 70:
            return "medium_load"
        elif cpu_percent > 20 or memory_percent > 50:
            return "low_load"
        else:
            return "idle"

    def _passes_quality_filters(self, proc_info):
        """Check if process data passes quality filters"""
        if not proc_info or proc_info.get('cpu_percent') is None or proc_info.get('memory_percent') is None:
            return False

        cpu = proc_info['cpu_percent']
        memory = proc_info['memory_percent']

        if not (self.quality_filters['cpu_min'] <= cpu <= self.quality_filters['cpu_max']):
            return False

        if not (self.quality_filters['memory_min'] <= memory <= self.quality_filters['memory_max']):
            return False

        return True

    def _collect_process_metrics(self, pid, name, proc_info):
        """Collect comprehensive metrics for a process"""
        try:
            p = psutil.Process(pid)
            current_time = datetime.now()

            # Check process age
            try:
                create_time = datetime.fromtimestamp(p.create_time())
                age_seconds = (current_time - create_time).total_seconds()
                if age_seconds < self.quality_filters['min_process_age']:
                    return None  # Process too young
            except:
                pass  # Some processes may not have create time

            # Get detailed metrics
            cpu_times = p.cpu_times()
            memory_info = p.memory_info()
            io_counters = p.io_counters()
            net_connections = p.connections()

            sample = {
                'timestamp': current_time,
                'pid': pid,
                'name': name,
                'cpu_percent': round(proc_info['cpu_percent'], 2),
                'memory_percent': round(proc_info['memory_percent'], 2),
                'cpu_user': round(cpu_times.user, 2) if cpu_times else 0,
                'cpu_system': round(cpu_times.system, 2) if cpu_times else 0,
                'memory_rss': memory_info.rss // (1024 * 1024) if memory_info else 0,  # MB
                'memory_vms': memory_info.vms // (1024 * 1024) if memory_info else 0,  # MB
                'read_bytes': io_counters.read_bytes // (1024 * 1024) if io_counters else 0,  # MB
                'write_bytes': io_counters.write_bytes // (1024 * 1024) if io_counters else 0,  # MB
                'num_threads': p.num_threads() if hasattr(p, 'num_threads') else 0,
                'num_fds': len(p.open_files()) if hasattr(p, 'open_files') else 0,
                'network_connections': len(net_connections) if net_connections else 0,
                'process_age_seconds': age_seconds if 'age_seconds' in locals() else 0,
            }

            return sample

        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            return None

    def save_collected_data(self):
        """Save collected training data to disk"""
        print(f"💾 Saving training data to {self.data_dir}...")

        # Save individual process data as CSV files
        for process_name, samples in self.training_samples.items():
            if len(samples) > 10:  # Only save if meaningful amount
                df = pd.DataFrame(samples)
                filename = self.data_dir / f"{process_name.replace('/', '_')}.csv"
                df.to_csv(filename, index=False)
                print(f"   Saved {len(samples)} samples for '{process_name}'")

        # Save collection statistics
        stats_file = self.data_dir / 'collection_stats.json'
        stats_data = {
            'collection_stats': self.stats,
            'quality_filters': self.quality_filters,
            'processes': list(self.stats['processes_discovered']),
            'collection_completed': datetime.now().isoformat()
        }

        with open(stats_file, 'w') as f:
            json.dump(stats_data, f, indent=2, default=str)

        # Save system conditions
        if self.system_conditions:
            system_df = pd.DataFrame(self.system_conditions)
            system_df.to_csv(self.data_dir / 'system_conditions.csv', index=False)

        print("✅ Training data saved successfully!")
        self.logger.info(f"Saved training data for {len(self.training_samples)} processes")

    def load_saved_data(self):
        """Load previously saved training data"""
        print(f"📂 Loading saved training data from {self.data_dir}...")

        loaded_count = 0
        for csv_file in self.data_dir.glob("*.csv"):
            if csv_file.name != 'system_conditions.csv':
                try:
                    df = pd.read_csv(csv_file)
                    process_name = csv_file.stem.replace('_', '/')
                    samples = df.to_dict('records')

                    self.training_samples[process_name] = samples
                    loaded_count += len(samples)
                    self.stats['processes_discovered'].add(process_name)
                    self.stats['total_collected'] += len(samples)

                    print(f"   Loaded {len(samples)} samples for '{process_name}'")
                except Exception as e:
                    print(f"   Error loading {csv_file}: {e}")

        # Load stats
        stats_file = self.data_dir / 'collection_stats.json'
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    stats_data = json.load(f)
                self.stats.update(stats_data.get('collection_stats', {}))
            except:
                pass

        print(f"✅ Loaded {loaded_count} training samples across {len(self.training_samples)} processes")

    def export_training_data(self, export_path="exported_training_data.zip"):
        """Export training data as a compressed archive"""
        import zipfile
        import shutil

        print(f"📦 Exporting training data to {export_path}...")

        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in self.data_dir.glob('*'):
                zipf.write(file_path, arcname=file_path.name)

        print("✅ Training data exported successfully!")

    def generate_training_summary(self):
        """Generate summary statistics of collected training data"""
        summary = {
            'total_processes': len(self.training_samples),
            'total_samples': sum(len(samples) for samples in self.training_samples.values()),
            'per_process_stats': {}
        }

        for process_name, samples in self.training_samples.items():
            if len(samples) == 0:
                continue

            cpu_values = [s['cpu_percent'] for s in samples]
            mem_values = [s['memory_percent'] for s in samples]

            summary['per_process_stats'][process_name] = {
                'samples': len(samples),
                'cpu_stats': {
                    'mean': round(np.mean(cpu_values), 2),
                    'std': round(np.std(cpu_values), 2),
                    'min': round(np.min(cpu_values), 2),
                    'max': round(np.max(cpu_values), 2)
                },
                'memory_stats': {
                    'mean': round(np.mean(mem_values), 2),
                    'std': round(np.std(mem_values), 2),
                    'min': round(np.min(mem_values), 2),
                    'max': round(np.max(mem_values), 2)
                }
            }

        # Save summary
        summary_file = self.data_dir / 'training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"📋 Training summary generated: {summary['total_processes']} processes, {summary['total_samples']} samples")
        return summary


def demonstrate_training_data_collection():
    """Demo of the enhanced training data collection pipeline"""
    print("🔬 Enhanced Training Data Collection Demo")
    print("=" * 50)

    collector = TrainingDataCollector(data_dir="demo_training_data")

    # Simulate quick collection (1 minute)
    print("🎯 Starting 1-minute training data collection...")
    collector.start_collection(duration_minutes=1)

    # Wait a bit
    time.sleep(10)

    # Stop collection
    collector.stop_collection()

    # Show what was collected
    print(f"\n📊 Collection Results:")
    print(f"   Total samples: {collector.stats['total_collected']}")
    print(f"   Filtered samples: {collector.stats['quality_filtered']}")
    print(f"   Unique processes: {len(collector.stats['processes_discovered'])}")

    # Generate summary
    if collector.stats['total_collected'] > 0:
        summary = collector.generate_training_summary()

        print("\n\342\223\206 Top 5 processes by sample count:")
        top_processes = sorted(summary['per_process_stats'].items(),
                              key=lambda x: x[1]['samples'], reverse=True)[:5]

        for name, stats in top_processes:
            print(f"   {name}: {stats['samples']} samples "
                  f"(CPU: {stats['cpu_stats']['mean']:.1f}%, "
                  f"Mem: {stats['memory_stats']['mean']:.1f}%)")


def quick_demonstration():
    """Demo function to show anomaly detection in action"""
    detector = ProcessAnomalyDetector()

    # Simulate some process metrics
    processes = [
        {"name": "chrome", "cpu_percent": 25.0, "memory_percent": 15.0},
        {"name": "chrome", "cpu_percent": 95.0, "memory_percent": 95.0},  # Anomalous
        {"name": "vscode", "cpu_percent": 45.0, "memory_percent": 25.0},
        {"name": "python", "cpu_percent": 120.0, "memory_percent": 110.0}  # Anomalous
    ]

    print("Process Anomaly Detection Demo")
    print("=" * 40)

    for proc in processes:
        # Note: Without training, models will return score 0
        score = detector.get_anomaly_score(proc["name"], proc)
        print(".2f")


if __name__ == "__main__":
    # Demonstrate both training data collection and anomaly detection
    demonstrate_training_data_collection()
    print("\n" + "="*50)
    quick_demonstration()
