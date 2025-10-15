import tkinter as tk
from tkinter import ttk, messagebox
import psutil
import platform
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
import numpy as np
import os

# Import forecasting module
try:
    from forecaster_mem import MemoryForecaster, LiveDataCollector
    FORECASTING_AVAILABLE = True
except ImportError:
    FORECASTING_AVAILABLE = False
    print("Warning: Forecasting module not available. Install required dependencies.")


class SystemMonitor:
    def __init__(self, root, enable_forecasting=True):
        self.root = root
        self.root.title("System Monitor with Forecasting")
        self.root.geometry("1600x950")
        self.root.configure(bg="#2b2b2b")
        
        # Data storage for graphs
        self.cpu_history = deque(maxlen=60)
        self.memory_history = deque(maxlen=60)
        self.memory_forecast = deque(maxlen=60)
        self.timestamps = deque(maxlen=60)
        
        # Forecasting
        self.forecasting_enabled = enable_forecasting and FORECASTING_AVAILABLE
        self.forecaster = None
        self.collector = None
        
        if self.forecasting_enabled:
            try:
                # Uses N-BEATS artifacts under saved_model/ via adapter in forecaster_mem.py
                self.forecaster = MemoryForecaster(seq_length=50)
                self.collector = LiveDataCollector()
                self.forecast_thread = threading.Thread(
                    target=self._forecasting_loop, 
                    daemon=True
                )
            except Exception as e:
                print(f"Error initializing forecaster: {e}")
                self.forecasting_enabled = False
        
        # Auto-refresh flag
        self.auto_refresh = True
        self.running = True
        self.all_processes = []
        
        # Create UI
        self.create_widgets()
        
        # Start update loop
        self.update_data()
        
        if self.forecasting_enabled:
            self.forecast_thread.start()
    
    def _forecasting_loop(self):
        """Background thread for continuous forecasting"""
        log_file = "./realtime_data/memory_usage_log.csv"

        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        while self.running:
            try:
                if self.forecaster and self.collector:
                    # Collect live data
                    data = self.collector.collect()
                    memory_percent = data['memory_percent']
                    self.forecaster.add_data_point(memory_percent)
                    
                    # Make prediction
                    result = self.forecaster.predict_next()
                    if result['prediction'] is not None:
                        self.memory_forecast.append(result['prediction'])
                    
                    # Log to file
                    with open(log_file, "a") as f:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"{timestamp},{memory_percent:.2f}\n")

                time.sleep(2)
            except Exception as e:
                print(f"Forecasting loop error: {e}")

    def create_widgets(self):
        # Main container
        main_frame = tk.Frame(self.root, bg="#2b2b2b")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text="🖥️ System Monitor with Forecasting", 
                               font=("Arial", 24, "bold"), bg="#2b2b2b", fg="#ffffff")
        title_label.pack(pady=(0, 10))
        
        # Top section - System Overview
        self.create_system_overview(main_frame)
        
        # Middle section - Graphs (Enhanced with forecasting)
        self.create_graphs_section(main_frame)
        
        # Bottom section - Process Management
        self.create_process_section(main_frame)
        
        # Control buttons
        self.create_controls(main_frame)
    
    def create_system_overview(self, parent):
        overview_frame = tk.LabelFrame(parent, text="📊 System Overview", 
                                       font=("Arial", 12, "bold"), 
                                       bg="#3c3c3c", fg="#ffffff", bd=2)
        overview_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Metrics container
        metrics_container = tk.Frame(overview_frame, bg="#3c3c3c")
        metrics_container.pack(fill=tk.X, padx=15, pady=15)
        
        # First row - CPU and Memory cards
        row1 = tk.Frame(metrics_container, bg="#3c3c3c")
        row1.pack(fill=tk.X, pady=(0, 10))
        
        # CPU Card
        cpu_card = tk.Frame(row1, bg="#1a1a1a", relief=tk.FLAT, bd=1)
        cpu_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        cpu_inner = tk.Frame(cpu_card, bg="#1a1a1a")
        cpu_inner.pack(fill=tk.BOTH, padx=15, pady=12)
        
        tk.Label(cpu_inner, text="🔥 CPU Usage", font=("Arial", 11, "bold"),
                bg="#1a1a1a", fg="#b0b0b0").pack(anchor="w")
        self.cpu_label = tk.Label(cpu_inner, text="---%", 
                                  font=("Arial", 24, "bold"), bg="#1a1a1a", fg="#ff5252")
        self.cpu_label.pack(anchor="w", pady=(5, 0))
        self.cpu_cores_label = tk.Label(cpu_inner, text="-- cores", 
                                        font=("Arial", 10), bg="#1a1a1a", fg="#808080")
        self.cpu_cores_label.pack(anchor="w")
        
        # Memory Card (Enhanced with forecast)
        memory_card = tk.Frame(row1, bg="#1a1a1a", relief=tk.FLAT, bd=1)
        memory_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        memory_inner = tk.Frame(memory_card, bg="#1a1a1a")
        memory_inner.pack(fill=tk.BOTH, padx=15, pady=12)
        
        tk.Label(memory_inner, text="💾 Memory Usage", font=("Arial", 11, "bold"),
                bg="#1a1a1a", fg="#b0b0b0").pack(anchor="w")
        self.memory_label = tk.Label(memory_inner, text="---%", 
                                     font=("Arial", 24, "bold"), bg="#1a1a1a", fg="#00bcd4")
        self.memory_label.pack(anchor="w", pady=(5, 0))
        self.memory_details_label = tk.Label(memory_inner, text="-- / -- GB", 
                                            font=("Arial", 10), bg="#1a1a1a", fg="#808080")
        self.memory_details_label.pack(anchor="w")
        
        # Memory Forecast Card (New)
        if self.forecasting_enabled:
            forecast_card = tk.Frame(row1, bg="#1a1a1a", relief=tk.FLAT, bd=1)
            forecast_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            
            forecast_inner = tk.Frame(forecast_card, bg="#1a1a1a")
            forecast_inner.pack(fill=tk.BOTH, padx=15, pady=12)
            
            tk.Label(forecast_inner, text="🔮 Memory Forecast (Next 25 seconds)", font=("Arial", 11, "bold"),
                    bg="#1a1a1a", fg="#b0b0b0").pack(anchor="w")
            self.forecast_label = tk.Label(forecast_inner, text="---%", 
                                          font=("Arial", 24, "bold"), bg="#1a1a1a", fg="#9c27b0")
            self.forecast_label.pack(anchor="w", pady=(5, 0))
            self.forecast_status_label = tk.Label(forecast_inner, text="Loading...", 
                                                 font=("Arial", 10), bg="#1a1a1a", fg="#808080")
            self.forecast_status_label.pack(anchor="w")
        
        # Disk Card
        disk_card = tk.Frame(row1, bg="#1a1a1a", relief=tk.FLAT, bd=1)
        disk_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        disk_inner = tk.Frame(disk_card, bg="#1a1a1a")
        disk_inner.pack(fill=tk.BOTH, padx=15, pady=12)
        
        tk.Label(disk_inner, text="💿 Disk Usage", font=("Arial", 11, "bold"),
                bg="#1a1a1a", fg="#b0b0b0").pack(anchor="w")
        self.disk_label = tk.Label(disk_inner, text="---%", 
                                   font=("Arial", 24, "bold"), bg="#1a1a1a", fg="#ab47bc")
        self.disk_label.pack(anchor="w", pady=(5, 0))
        self.disk_details_label = tk.Label(disk_inner, text="-- / -- GB", 
                                          font=("Arial", 10), bg="#1a1a1a", fg="#808080")
        self.disk_details_label.pack(anchor="w")
        
        # Battery Card
        battery_card = tk.Frame(row1, bg="#1a1a1a", relief=tk.FLAT, bd=1)
        battery_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        battery_inner = tk.Frame(battery_card, bg="#1a1a1a")
        battery_inner.pack(fill=tk.BOTH, padx=15, pady=12)
        
        tk.Label(battery_inner, text="🔋 Battery", font=("Arial", 11, "bold"),
                bg="#1a1a1a", fg="#b0b0b0").pack(anchor="w")
        self.battery_label = tk.Label(battery_inner, text="N/A", 
                                      font=("Arial", 24, "bold"), bg="#1a1a1a", fg="#4caf50")
        self.battery_label.pack(anchor="w", pady=(5, 0))
        self.battery_status_label = tk.Label(battery_inner, text="--", 
                                             font=("Arial", 10), bg="#1a1a1a", fg="#808080")
        self.battery_status_label.pack(anchor="w")
        
        # Second row - System info cards
        row2 = tk.Frame(metrics_container, bg="#3c3c3c")
        row2.pack(fill=tk.X)
        
        # Processes Card
        processes_card = tk.Frame(row2, bg="#1a1a1a", relief=tk.FLAT, bd=1)
        processes_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        processes_inner = tk.Frame(processes_card, bg="#1a1a1a")
        processes_inner.pack(fill=tk.BOTH, padx=15, pady=10)
        
        tk.Label(processes_inner, text="⚙️ Processes", font=("Arial", 10, "bold"),
                bg="#1a1a1a", fg="#b0b0b0").pack(anchor="w")
        self.processes_count_label = tk.Label(processes_inner, text="--", 
                                              font=("Arial", 18, "bold"), bg="#1a1a1a", fg="#ffa726")
        self.processes_count_label.pack(anchor="w", pady=(2, 0))
        
        # Uptime Card
        uptime_card = tk.Frame(row2, bg="#1a1a1a", relief=tk.FLAT, bd=1)
        uptime_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        uptime_inner = tk.Frame(uptime_card, bg="#1a1a1a")
        uptime_inner.pack(fill=tk.BOTH, padx=15, pady=10)
        
        tk.Label(uptime_inner, text="⏱️ System Uptime", font=("Arial", 10, "bold"),
                bg="#1a1a1a", fg="#b0b0b0").pack(anchor="w")
        self.uptime_label = tk.Label(uptime_inner, text="--", 
                                     font=("Arial", 18, "bold"), bg="#1a1a1a", fg="#66bb6a")
        self.uptime_label.pack(anchor="w", pady=(2, 0))
        
        # Network Card
        network_card = tk.Frame(row2, bg="#1a1a1a", relief=tk.FLAT, bd=1)
        network_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        network_inner = tk.Frame(network_card, bg="#1a1a1a")
        network_inner.pack(fill=tk.BOTH, padx=15, pady=10)
        
        tk.Label(network_inner, text="🌐 Network", font=("Arial", 10, "bold"),
                bg="#1a1a1a", fg="#b0b0b0").pack(anchor="w")
        self.network_label = tk.Label(network_inner, text="-- / --", 
                                      font=("Arial", 14, "bold"), bg="#1a1a1a", fg="#29b6f6")
        self.network_label.pack(anchor="w", pady=(2, 0))
        
        # Platform Card
        platform_card = tk.Frame(row2, bg="#1a1a1a", relief=tk.FLAT, bd=1)
        platform_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        platform_inner = tk.Frame(platform_card, bg="#1a1a1a")
        platform_inner.pack(fill=tk.BOTH, padx=15, pady=10)
        
        tk.Label(platform_inner, text="🖥️ Platform", font=("Arial", 10, "bold"),
                bg="#1a1a1a", fg="#b0b0b0").pack(anchor="w")
        platform_text = f"{platform.system()} {platform.release()}"
        self.platform_label = tk.Label(platform_inner, text=platform_text, 
                                       font=("Arial", 14, "bold"), bg="#1a1a1a", fg="#78909c")
        self.platform_label.pack(anchor="w", pady=(2, 0))
    
    def create_graphs_section(self, parent):
        graphs_frame = tk.LabelFrame(parent, text="📈 Performance Graphs with Forecast", 
                                     font=("Arial", 12, "bold"), 
                                     bg="#3c3c3c", fg="#ffffff", bd=2)
        graphs_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create figure for graphs with dark background (CPU and Memory on two subplots)
        self.fig = Figure(figsize=(14, 4.5), facecolor="#2b2b2b", dpi=100)
        # CPU Graph
        self.cpu_ax = self.fig.add_subplot(121)
        
        self.cpu_ax.set_facecolor("#1a1a1a")
        self.cpu_ax.set_title("CPU", color="#ffffff", fontsize=13, fontweight='bold', pad=10)
        self.cpu_ax.set_ylabel("%", color="#ffffff", fontsize=11, fontweight='bold')
        self.cpu_ax.tick_params(colors="#b0b0b0", labelsize=9)
        self.cpu_ax.set_ylim(0, 100)
        self.cpu_ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.8, color="#ffffff")
        
        # Memory Graph (will overlay forecast when available)
        self.memory_ax = self.fig.add_subplot(122)
        
        self.memory_ax.set_facecolor("#1a1a1a")
        self.memory_ax.set_title("Memory", color="#ffffff", fontsize=13, fontweight='bold', pad=10)
        self.memory_ax.set_ylabel("%", color="#ffffff", fontsize=11, fontweight='bold')
        self.memory_ax.tick_params(colors="#b0b0b0", labelsize=9)
        self.memory_ax.set_ylim(0, 100)
        self.memory_ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.8, color="#ffffff")
        
        # No separate forecast graph; forecast will be overlaid on the memory graph
        
        # Adjust layout
        self.fig.tight_layout(pad=2.0)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=graphs_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def create_process_section(self, parent):
        process_frame = tk.LabelFrame(parent, text="⚙️ Running Processes", 
                                      font=("Arial", 12, "bold"), 
                                      bg="#3c3c3c", fg="#ffffff", bd=2)
        process_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Control frame
        control_frame = tk.Frame(process_frame, bg="#3c3c3c")
        control_frame.pack(fill=tk.X, padx=15, pady=10)
        
        # Search
        tk.Label(control_frame, text="🔍 Search:", bg="#3c3c3c", fg="#ffffff", 
                font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5, sticky="w")
        self.search_var = tk.StringVar()
        self.search_var.trace('w', lambda *args: self.filter_processes())
        search_entry = tk.Entry(control_frame, textvariable=self.search_var, width=35,
                               font=("Arial", 10), bg="#4a4a4a", fg="#ffffff",
                               insertbackground="#ffffff", relief=tk.FLAT, bd=5)
        search_entry.grid(row=0, column=1, padx=5, sticky="w")
        
        # Filter
        tk.Label(control_frame, text="Filter:", bg="#3c3c3c", fg="#ffffff",
                font=("Arial", 10, "bold")).grid(row=0, column=2, padx=(20, 5), sticky="w")
        self.filter_var = tk.StringVar(value="All Processes")
        filter_combo = ttk.Combobox(control_frame, textvariable=self.filter_var, 
                                    values=["All Processes", "High CPU (>10%)", "High Memory (>5%)"],
                                    state="readonly", width=18, font=("Arial", 10))
        filter_combo.grid(row=0, column=3, padx=5, sticky="w")
        filter_combo.bind('<<ComboboxSelected>>', lambda e: self.filter_processes())
        
        # Sort
        tk.Label(control_frame, text="Sort by:", bg="#3c3c3c", fg="#ffffff",
                font=("Arial", 10, "bold")).grid(row=0, column=4, padx=(20, 5), sticky="w")
        self.sort_var = tk.StringVar(value="CPU %")
        sort_combo = ttk.Combobox(control_frame, textvariable=self.sort_var,
                                 values=["PID", "Name", "CPU %", "Memory %", "Priority"],
                                 state="readonly", width=15, font=("Arial", 10))
        sort_combo.grid(row=0, column=5, padx=5, sticky="w")
        sort_combo.bind('<<ComboboxSelected>>', lambda e: self.filter_processes())
        
        # Configure custom ttk style for treeview
        style = ttk.Style()
        style.theme_use("clam")
        
        style.configure("Custom.Treeview",
                       background="#2b2b2b",
                       foreground="#ffffff",
                       fieldbackground="#2b2b2b",
                       borderwidth=0,
                       font=("Consolas", 11),
                       rowheight=28)
        
        style.configure("Custom.Treeview.Heading",
                       background="#1a1a1a",
                       foreground="#ffffff",
                       borderwidth=1,
                       relief="flat",
                       font=("Arial", 12, "bold"))
        
        style.map("Custom.Treeview",
                 background=[("selected", "#0d7377")],
                 foreground=[("selected", "#ffffff")])
        
        style.map("Custom.Treeview.Heading",
                 background=[("active", "#323232")])
        
        # Process tree
        tree_frame = tk.Frame(process_frame, bg="#2b2b2b", relief=tk.FLAT)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(5, 10))
        
        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        
        columns = ("PID", "Name", "CPU %", "Memory %", "Priority")
        self.process_tree = ttk.Treeview(tree_frame, columns=columns, show="headings",
                                         yscrollcommand=vsb.set, xscrollcommand=hsb.set,
                                         style="Custom.Treeview", height=15)
        
        vsb.config(command=self.process_tree.yview)
        hsb.config(command=self.process_tree.xview)
        
        self.process_tree.heading("PID", text="  PID", anchor="w")
        self.process_tree.heading("Name", text="  Process Name", anchor="w")
        self.process_tree.heading("CPU %", text="CPU %", anchor="center")
        self.process_tree.heading("Memory %", text="Memory %", anchor="center")
        self.process_tree.heading("Priority", text="Priority", anchor="center")
        
        self.process_tree.column("PID", width=100, minwidth=80, anchor="w")
        self.process_tree.column("Name", width=400, minwidth=250, anchor="w")
        self.process_tree.column("CPU %", width=130, minwidth=100, anchor="center")
        self.process_tree.column("Memory %", width=130, minwidth=100, anchor="center")
        self.process_tree.column("Priority", width=130, minwidth=100, anchor="center")
        
        self.process_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Process management controls
        mgmt_frame = tk.Frame(process_frame, bg="#1a1a1a", relief=tk.FLAT)
        mgmt_frame.pack(fill=tk.X, padx=15, pady=(10, 10))
        
        inner_mgmt = tk.Frame(mgmt_frame, bg="#1a1a1a")
        inner_mgmt.pack(fill=tk.X, padx=10, pady=10)

        # Configure grid weights
        for i in range(6):
            inner_mgmt.grid_columnconfigure(i, weight=1)

        tk.Label(inner_mgmt, text="Selected PID:", bg="#1a1a1a", fg="#b0b0b0",
                font=("Arial", 10, "bold")).grid(row=0, column=0, padx=(5, 2), sticky="w")
        self.selected_pid_label = tk.Label(inner_mgmt, text="None", bg="#1a1a1a", 
                                        fg="#ffeb3b", font=("Arial", 10, "bold"))
        self.selected_pid_label.grid(row=0, column=1, padx=(2, 15), sticky="w")

        self.terminate_btn = tk.Button(inner_mgmt, text="🛑 Terminate Process", 
                                    command=self.terminate_process,
                                    bg="#e53935", fg="#ffffff", font=("Arial", 10, "bold"),
                                    activebackground="#c62828", activeforeground="#ffffff",
                                    relief=tk.FLAT, bd=0, padx=15, pady=8,
                                    cursor="hand2")
        self.terminate_btn.grid(row=0, column=2, padx=10, sticky="w")

        # Priority control
        if platform.system() == 'Windows':
            tk.Label(inner_mgmt, text="Priority:", bg="#1a1a1a", fg="#b0b0b0",
                    font=("Arial", 10, "bold")).grid(row=0, column=3, padx=(20, 5), sticky="w")
            self.priority_var = tk.StringVar(value="Normal")
            priority_options = ["Idle", "Below Normal", "Normal", "Above Normal", "High", "Realtime"]
            priority_combo = ttk.Combobox(inner_mgmt, textvariable=self.priority_var,
                                        values=priority_options, state="readonly", 
                                        width=15, font=("Arial", 10))
            priority_combo.grid(row=0, column=4, padx=5, sticky="w")
        else:
            tk.Label(inner_mgmt, text="Priority (nice):", bg="#1a1a1a", fg="#b0b0b0",
                    font=("Arial", 10, "bold")).grid(row=0, column=3, padx=(20, 5), sticky="w")
            self.priority_var = tk.IntVar(value=0)
            priority_scale = tk.Scale(inner_mgmt, from_=-20, to=19, orient=tk.HORIZONTAL,
                                    variable=self.priority_var, bg="#1a1a1a", fg="#ffffff",
                                    troughcolor="#3c3c3c", activebackground="#0d7377",
                                    highlightthickness=0, length=150)
            priority_scale.grid(row=0, column=4, padx=5, sticky="w")

        self.change_priority_btn = tk.Button(inner_mgmt, text="🔧 Change Priority",
                                            command=self.change_priority,
                                            bg="#1976d2", fg="#ffffff", font=("Arial", 10, "bold"),
                                            activebackground="#1565c0", activeforeground="#ffffff",
                                            relief=tk.FLAT, bd=0, padx=15, pady=8,
                                            cursor="hand2")
        self.change_priority_btn.grid(row=0, column=5, padx=10, sticky="w")

        self.process_tree.bind('<<TreeviewSelect>>', self.on_process_select)
    
    def create_controls(self, parent):
        control_frame = tk.Frame(parent, bg="#2b2b2b")
        control_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.auto_refresh_var = tk.BooleanVar(value=True)
        auto_refresh_check = tk.Checkbutton(control_frame, text="Auto-refresh (2 seconds)",
                                           variable=self.auto_refresh_var, bg="#2b2b2b", 
                                           fg="#ffffff", selectcolor="#2b2b2b",
                                           font=("Arial", 10))
        auto_refresh_check.pack(side=tk.LEFT, padx=10)
        
        refresh_btn = tk.Button(control_frame, text="🔄 Refresh Now", command=self.manual_refresh,
                               bg="#4caf50", fg="#ffffff", font=("Arial", 10, "bold"))
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        if self.forecasting_enabled:
            forecast_info_btn = tk.Button(control_frame, text="📊 Forecast Info", 
                                         command=self.show_forecast_info,
                                         bg="#9c27b0", fg="#ffffff", font=("Arial", 10, "bold"))
            forecast_info_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(control_frame, text="Ready", bg="#2b2b2b", fg="#4caf50")
        self.status_label.pack(side=tk.RIGHT, padx=10)
    
    def show_forecast_info(self):
        """Display forecasting statistics"""
        if self.forecaster:
            stats = self.forecaster.get_forecast_statistics()
            if stats:
                info = f"""Memory Forecast Statistics:
                
Mean Prediction: {stats['mean']:.2f}%
Std Deviation: {stats['std']:.2f}%
Min: {stats['min']:.2f}%
Max: {stats['max']:.2f}%
Trend: {stats['trend'].upper()}

Buffer Size: {len(self.forecaster.memory_buffer)}
Predictions: {len(self.forecaster.predictions_history)}"""
                messagebox.showinfo("Forecast Statistics", info)
            else:
                messagebox.showinfo("Forecast", "Collecting initial data...")
    
    def get_priority_display(self, nice_value):
        if platform.system() == 'Windows':
            priority_names = {
                64: "Idle",
                16384: "Below Normal", 
                32: "Normal",
                32768: "Above Normal",
                128: "High",
                256: "Realtime"
            }
            return priority_names.get(nice_value, str(nice_value))
        else:
            return str(nice_value)
    
    def get_system_info(self):
        cpu_percent = psutil.cpu_percent(interval=0.5)
        cpu_cores = psutil.cpu_count(logical=True)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        battery = None
        try:
            battery = psutil.sensors_battery()
        except:
            pass
        
        net_io = psutil.net_io_counters()
        boot_time = psutil.boot_time()
        uptime_seconds = time.time() - boot_time
        process_count = len(psutil.pids())
        
        return {
            'cpu_percent': cpu_percent,
            'cpu_cores': cpu_cores,
            'memory_percent': memory.percent,
            'memory_used': memory.used / (1024**3),
            'memory_total': memory.total / (1024**3),
            'proces_count' : process_count,
            'disk_percent': disk.percent,
            'disk_used': disk.used / (1024**3),
            'disk_total': disk.total / (1024**3),
            'battery': battery,
            'net_sent': net_io.bytes_sent / (1024**2),
            'net_recv': net_io.bytes_recv / (1024**2),
            'uptime_seconds': uptime_seconds
        }
    
    def get_processes(self):
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'nice']):
            try:
                proc_info = proc.info
                if proc_info['cpu_percent'] is not None and proc_info['memory_percent'] is not None:
                    nice_value = proc_info.get('nice', 0)
                    processes.append({
                        'PID': proc_info['pid'],
                        'Name': proc_info['name'],
                        'CPU %': round(proc_info['cpu_percent'], 2),
                        'Memory %': round(proc_info['memory_percent'], 2),
                        'Priority': self.get_priority_display(nice_value)
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return processes
    
    def update_data(self):
        if not self.running:
            return
        
        try:
            system_info = self.get_system_info()
            
            self.cpu_label.config(text=f"{system_info['cpu_percent']:.1f}%")
            self.cpu_cores_label.config(text=f"{system_info['cpu_cores']} cores")
            
            self.memory_label.config(text=f"{system_info['memory_percent']:.1f}%")
            self.memory_details_label.config(
                text=f"{system_info['memory_used']:.1f} / {system_info['memory_total']:.1f} GB"
            )
            
            self.disk_label.config(text=f"{system_info['disk_percent']:.1f}%")
            self.disk_details_label.config(
                text=f"{system_info['disk_used']:.1f} / {system_info['disk_total']:.1f} GB"
            )
            
            # Update forecast display
            if self.forecasting_enabled and len(self.memory_forecast) > 0:
                latest_forecast = list(self.memory_forecast)[-1]
                self.forecast_label.config(text=f"{latest_forecast:.1f}%")
                
                current_mem = system_info['memory_percent']
                diff = latest_forecast - current_mem
                if diff > 2:
                    status = f"↑ Rising ({diff:+.1f}%)"
                    self.forecast_status_label.config(fg="#ff5252")
                elif diff < -2:
                    status = f"↓ Dropping ({diff:+.1f}%)"
                    self.forecast_status_label.config(fg="#4caf50")
                else:
                    status = f"→ Stable ({diff:+.1f}%)"
                    self.forecast_status_label.config(fg="#ffc107")
                
                self.forecast_status_label.config(text=status)
            
            if system_info['battery']:
                battery = system_info['battery']
                self.battery_label.config(text=f"{battery.percent:.0f}%")
                if battery.power_plugged:
                    self.battery_status_label.config(text="Charging")
                    self.battery_label.config(fg="#4caf50")
                else:
                    time_left = battery.secsleft
                    if time_left > 0 and time_left != psutil.POWER_TIME_UNLIMITED:
                        hours = time_left // 3600
                        minutes = (time_left % 3600) // 60
                        self.battery_status_label.config(text=f"{int(hours)}h {int(minutes)}m left")
                    else:
                        self.battery_status_label.config(text="On Battery")
                    
                    if battery.percent < 20:
                        self.battery_label.config(fg="#f44336")
                    elif battery.percent < 50:
                        self.battery_label.config(fg="#ff9800")
                    else:
                        self.battery_label.config(fg="#4caf50")
            else:
                self.battery_label.config(text="N/A")
                self.battery_status_label.config(text="No Battery")
            
            self.network_label.config(
                text=f"↓{system_info['net_recv']:.0f}MB / ↑{system_info['net_sent']:.0f}MB"
            )
            
            uptime_sec = system_info['uptime_seconds']
            days = int(uptime_sec // 86400)
            hours = int((uptime_sec % 86400) // 3600)
            minutes = int((uptime_sec % 3600) // 60)
            
            if days > 0:
                uptime_text = f"{days}d {hours}h {minutes}m"
            elif hours > 0:
                uptime_text = f"{hours}h {minutes}m"
            else:
                uptime_text = f"{minutes}m"
            
            self.uptime_label.config(text=uptime_text)
            
            current_time = datetime.now()
            self.timestamps.append(current_time)
            self.cpu_history.append(system_info['cpu_percent'])
            self.memory_history.append(system_info['memory_percent'])
            
            self.update_graphs()
            self.update_processes()
            self.processes_count_label.config(text=str(len(self.all_processes)))
            self.status_label.config(text=f"Last updated: {current_time.strftime('%H:%M:%S')}", fg="#4caf50")
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", fg="#f44336")
        
        if self.auto_refresh_var.get():
            self.root.after(2000, self.update_data)
        else:
            self.root.after(500, self.update_data)
    
    def update_graphs(self):
        """GNOME System Monitor style with proper titles and correct alignment"""
        if len(self.cpu_history) > 1:
            self.cpu_ax.clear()
            self.memory_ax.clear()
            
            timestamps_list = list(self.timestamps)
            cpu_list = list(self.cpu_history)
            memory_list = list(self.memory_history)

            # Light smoothing
            def _smooth(values, window=3):
                if len(values) < window:
                    return values
                kernel = np.ones(window) / window
                padded = np.pad(values, (window//2, window//2), mode='edge')
                smoothed = np.convolve(padded, kernel, mode='valid')
                return list(smoothed)

            cpu_smooth = _smooth(cpu_list, window=3)
            mem_smooth = _smooth(memory_list, window=3)
            # Clamp to [0,100]
            cpu_smooth = [min(100.0, max(0.0, v)) for v in cpu_smooth]
            mem_smooth = [min(100.0, max(0.0, v)) for v in mem_smooth]
            
            # Create x-axis as indices
            max_points = 60
            if len(cpu_smooth) < max_points:
                x_data = list(range(len(cpu_smooth)))
                actual_points = len(cpu_smooth)
            else:
                x_data = list(range(max_points))
                cpu_smooth = cpu_smooth[-max_points:]
                mem_smooth = mem_smooth[-max_points:]
                actual_points = max_points
            
            # ========== CPU GRAPH ==========
            self.cpu_ax.set_facecolor("#1e1e1e")
            
            # Set proper limits with padding for labels
            self.cpu_ax.set_xlim(-5, max_points + 5)
            self.cpu_ax.set_ylim(-5, 105)
            
            # Add title at top left
            self.cpu_ax.text(0.02, 0.96, 'CPU Usage',
                           transform=self.cpu_ax.transAxes,
                           fontsize=12, fontweight='bold',
                           color='#e0e0e0', ha='left', va='top',
                           family='sans-serif')
            
            # Horizontal grid lines
            for y in [0, 25, 50, 75, 100]:
                self.cpu_ax.axhline(y=y, color='#383838', linewidth=1.0,
                                   linestyle='-', zorder=1, alpha=0.6)
            
            # Draw the line - bright blue
            self.cpu_ax.plot(x_data, cpu_smooth,
                           color='#3584e4',  # GNOME blue
                           linewidth=2.0,
                           zorder=3,
                           antialiased=True,
                           solid_capstyle='round',
                           solid_joinstyle='round')
            
            # Hide all spines
            for spine in self.cpu_ax.spines.values():
                spine.set_visible(False)
            
            # ========== MEMORY GRAPH ==========
            self.memory_ax.set_facecolor("#1e1e1e")
            
            # Set proper limits with padding
            self.memory_ax.set_xlim(-5, max_points + 5)
            self.memory_ax.set_ylim(-5, 105)
            
            # Add title at top left
            self.memory_ax.text(0.02, 0.96, 'Memory Usage',
                              transform=self.memory_ax.transAxes,
                              fontsize=12, fontweight='bold',
                              color='#e0e0e0', ha='left', va='top',
                              family='sans-serif')
            
            # Horizontal grid lines
            for y in [0, 25, 50, 75, 100]:
                self.memory_ax.axhline(y=y, color='#383838', linewidth=1.0,
                                      linestyle='-', zorder=1, alpha=0.6)
            
            # Draw the line - orange
            self.memory_ax.plot(x_data, mem_smooth,
                              color='#e5a50a',  # GNOME orange/yellow
                              linewidth=2.0,
                              zorder=3,
                              antialiased=True,
                              solid_capstyle='round',
                              solid_joinstyle='round')
            
            # Forecast overlay if available
            if self.forecasting_enabled and len(self.memory_forecast) > 0:
                forecast_list = list(self.memory_forecast)
                
                if len(forecast_list) <= len(memory_list):
                    padded_forecast = [None] * (len(memory_list) - len(forecast_list)) + forecast_list
                else:
                    padded_forecast = forecast_list[-len(memory_list):]

                forecast_vals = [v for v in padded_forecast if v is not None]
                
                if len(forecast_vals) > 0:
                    if len(mem_smooth) >= max_points:
                        forecast_smooth = _smooth(forecast_vals[-max_points:], window=3)
                        x_forecast = list(range(max(0, max_points - len(forecast_smooth)), max_points))
                    else:
                        forecast_smooth = _smooth(forecast_vals, window=3)
                        x_forecast = list(range(len(x_data) - len(forecast_smooth), len(x_data)))

                    # Clamp forecast
                    forecast_smooth = [min(100.0, max(0.0, v)) for v in forecast_smooth]

                    # Purple dashed line
                    self.memory_ax.plot(x_forecast, forecast_smooth,
                                      color='#9141ac',  # GNOME purple
                                      linewidth=2.0,
                                      linestyle='--',
                                      dashes=(10, 5),
                                      zorder=4,
                                      antialiased=True,
                                      alpha=0.9)
            
            # Hide all spines
            for spine in self.memory_ax.spines.values():
                spine.set_visible(False)
            
            # ========== LABELS AND ANNOTATIONS ==========
            # Time duration and timestamp
            duration = len(cpu_list) * 2
            if duration < 120:
                time_label = f"{duration} secs"
            else:
                time_label = f"{duration//60} mins"
            
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # CPU bottom labels (using data coordinates)
            self.cpu_ax.text(0, -3, time_label,
                           fontsize=9, color='#909090',
                           ha='left', va='top')
            
            self.cpu_ax.text(actual_points - 1, -3, current_time,
                           fontsize=9, color='#909090',
                           ha='right', va='top')
            
            # Memory bottom labels
            self.memory_ax.text(0, -3, time_label,
                              fontsize=9, color='#909090',
                              ha='left', va='top')
            
            self.memory_ax.text(actual_points - 1, -3, current_time,
                              fontsize=9, color='#909090',
                              ha='right', va='top')
            
            # Y-axis percentage labels for CPU
            for y_val in [0, 25, 50, 75, 100]:
                self.cpu_ax.text(-2, y_val, f'{y_val}%',
                               fontsize=8, color='#707070',
                               ha='right', va='center')
            
            # Y-axis percentage labels for Memory
            for y_val in [0, 25, 50, 75, 100]:
                self.memory_ax.text(-2, y_val, f'{y_val}%',
                                  fontsize=8, color='#707070',
                                  ha='right', va='center')
            
            # Current value annotations
            if len(cpu_smooth) > 0:
                current_cpu = cpu_smooth[-1]
                # Position box outside the graph area
                self.cpu_ax.text(actual_points + 1.5, current_cpu, f' {current_cpu:.1f}%',
                               fontsize=10, color='#ffffff',
                               ha='left', va='center', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.4',
                                       facecolor='#3584e4',
                                       edgecolor='none',
                                       alpha=0.95))
            
            if len(mem_smooth) > 0:
                current_mem = mem_smooth[-1]
                self.memory_ax.text(actual_points + 1.5, current_mem, f' {current_mem:.1f}%',
                                  fontsize=10, color='#ffffff',
                                  ha='left', va='center', fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.4',
                                          facecolor='#e5a50a',
                                          edgecolor='none',
                                          alpha=0.95))
            
            # Forecast annotation
            if self.forecasting_enabled and len(self.memory_forecast) > 0:
                forecast_list = list(self.memory_forecast)
                if len(forecast_list) > 0 and len(mem_smooth) > 0:
                    current_forecast = forecast_list[-1]
                    current_mem = mem_smooth[-1]
                    
                    # Smart positioning to avoid overlap
                    y_pos = current_forecast
                    if abs(current_forecast - current_mem) < 8:
                        # Move forecast box up or down to avoid overlap
                        if current_forecast > 50:
                            y_pos = current_mem - 10
                        else:
                            y_pos = current_mem + 10
                    
                    self.memory_ax.text(actual_points + 1.5, y_pos, f' ↗ {current_forecast:.1f}%',
                                      fontsize=9, color='#ffffff',
                                      ha='left', va='center', fontweight='bold',
                                      bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='#9141ac',
                                              edgecolor='none',
                                              alpha=0.9))
            
            # Remove tick marks
            self.cpu_ax.set_xticks([])
            self.cpu_ax.set_yticks([])
            self.cpu_ax.tick_params(left=False, bottom=False,
                                   labelleft=False, labelbottom=False)
            
            self.memory_ax.set_xticks([])
            self.memory_ax.set_yticks([])
            self.memory_ax.tick_params(left=False, bottom=False,
                                      labelleft=False, labelbottom=False)
            
            # Adjust layout with proper spacing for labels
            self.fig.subplots_adjust(left=0.06, right=0.94, top=0.96, bottom=0.06,
                                    wspace=0.12, hspace=0.12)
            self.canvas.draw()

    def update_processes(self):
        self.all_processes = self.get_processes()
        self.filter_processes()
    
    def filter_processes(self):
    # Clear existing rows
        for item in self.process_tree.get_children():
            self.process_tree.delete(item)
        
        search_term = self.search_var.get().lower()
        filter_option = self.filter_var.get()
        sort_by = self.sort_var.get()
        
        filtered = self.all_processes
        
        # Apply search filter
        if search_term:
            filtered = [p for p in filtered if search_term in p['Name'].lower()]
        
        # Apply CPU/Memory filters
        if filter_option == "High CPU (>10%)":
            filtered = [p for p in filtered if p['CPU %'] > 10]
        elif filter_option == "High Memory (>5%)":
            filtered = [p for p in filtered if p['Memory %'] > 5]
        
        # Determine sorting order
        reverse = True
        if sort_by in ["Name", "Priority"]:
            reverse = False
        filtered.sort(key=lambda x: x[sort_by], reverse=reverse)
        
        # Configure row tags
        self.process_tree.tag_configure('evenrow', background='#2b2b2b', foreground='#ffffff')
        self.process_tree.tag_configure('oddrow', background='#3a3a3a', foreground='#ffffff')
        self.process_tree.tag_configure('highcpu', foreground='#ff5252')
        self.process_tree.tag_configure('highmem', foreground='#ffab40')
        
        # Insert top 50 processes
        for idx, proc in enumerate(filtered[:50]):
            tag = 'evenrow' if idx % 2 == 0 else 'oddrow'
            tags = [tag]
            
            # Highlight very high CPU or Memory usage
            if proc['CPU %'] > 50:
                tags.append('highcpu')
            elif proc['Memory %'] > 50:
                tags.append('highmem')
            
            pid_str = f"  {proc['PID']}"
            name_str = f"  {proc['Name']}"
            cpu_str = f"{proc['CPU %']:.2f}%"
            mem_str = f"{proc['Memory %']:.2f}%"
            priority_str = str(proc['Priority'])
            
            self.process_tree.insert("", tk.END, values=(
                pid_str, name_str, cpu_str, mem_str, priority_str
            ), tags=tags)

    def on_process_select(self, event):
        selection = self.process_tree.selection()
        if selection:
            item = self.process_tree.item(selection[0])
            pid_str = item['values'][0]
            pid = str(pid_str).strip()
            self.selected_pid_label.config(text=pid)
        else:
            self.selected_pid_label.config(text="None")

    def terminate_process(self):
        pid_text = self.selected_pid_label.cget("text")
        if pid_text == "None":
            messagebox.showwarning("No Selection", "Please select a process first")
            return
        
        pid = int(pid_text)
        
        if not messagebox.askyesno("Confirm Termination", 
                                f"Are you sure you want to terminate process {pid}?"):
            return
        
        try:
            process = psutil.Process(pid)
            process_name = process.name()
            process.terminate()
            messagebox.showinfo("Success", f"Process '{process_name}' (PID: {pid}) terminated successfully")
            self.selected_pid_label.config(text="None")
            self.manual_refresh()
        except psutil.NoSuchProcess:
            messagebox.showerror("Error", f"Process with PID {pid} not found")
        except psutil.AccessDenied:
            messagebox.showerror("Error", f"Access denied. Cannot terminate process {pid}")
        except Exception as e:
            messagebox.showerror("Error", f"Error terminating process: {str(e)}")

    def change_priority(self):
        pid_text = self.selected_pid_label.cget("text")
        if pid_text == "None":
            messagebox.showwarning("No Selection", "Please select a process first")
            return
        
        pid = int(pid_text)
        
        try:
            process = psutil.Process(pid)
            process_name = process.name()
            
            if platform.system() == 'Windows':
                priority_map = {
                    "Idle": psutil.IDLE_PRIORITY_CLASS,
                    "Below Normal": psutil.BELOW_NORMAL_PRIORITY_CLASS,
                    "Normal": psutil.NORMAL_PRIORITY_CLASS,
                    "Above Normal": psutil.ABOVE_NORMAL_PRIORITY_CLASS,
                    "High": psutil.HIGH_PRIORITY_CLASS,
                    "Realtime": psutil.REALTIME_PRIORITY_CLASS
                }
                priority_value = priority_map[self.priority_var.get()]
                priority_name = self.priority_var.get()
            else:
                priority_value = self.priority_var.get()
                priority_name = f"nice value {priority_value}"
            
            process.nice(priority_value)
            messagebox.showinfo("Success", 
                                f"Priority changed for '{process_name}' (PID: {pid}) to {priority_name}")
            self.manual_refresh()
        except psutil.NoSuchProcess:
            messagebox.showerror("Error", f"Process with PID {pid} not found")
        except psutil.AccessDenied:
            messagebox.showerror("Error", "Access denied. Try running with elevated permissions")
        except Exception as e:
            messagebox.showerror("Error", f"Error changing priority: {str(e)}")

    def manual_refresh(self):
        self.update_data()

    
    def on_closing(self):
        self.running = False
        # Cleanup generated memory data CSV on exit
        try:
            csv_path = "memory_data.csv"
            if os.path.exists(csv_path):
                os.remove(csv_path)
            # Also remove forecast CSV and realtime log if present
            forecast_csv = "memory_forecasts.csv"
            if os.path.exists(forecast_csv):
                os.remove(forecast_csv)
            realtime_log = os.path.join("realtime_data", "memory_usage_log.csv")
            if os.path.exists(realtime_log):
                os.remove(realtime_log)
        except Exception as e:
            print(f"Cleanup error: {e}")
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SystemMonitor(root, enable_forecasting=True)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()