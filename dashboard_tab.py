# dashboard_tab.py (patched)
import tkinter as tk
from tkinter import ttk, messagebox
import psutil
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from ml_system_manager import MLSystemManager


class GlassCard(tk.Frame):
    """Modern glassmorphism card widget"""
    
    def __init__(self, parent, title="", icon="", value="--", subtitle="", 
                 theme=None, show_progress=False, **kwargs):
        self.theme = theme or {}
        super().__init__(parent, bg=self.theme.get("glass_bg", "#1a1a2e99"), **kwargs)
        
        self.configure(
            highlightbackground=self.theme.get("accent_cyan", "#00d4ff"),
            highlightthickness=1,
            relief=tk.FLAT
        )
        
        inner = tk.Frame(self, bg=self.theme.get("bg_tertiary", "#16213e"))
        inner.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        header = tk.Frame(inner, bg=self.theme.get("bg_tertiary", "#16213e"))
        header.pack(fill=tk.X, padx=10, pady=(8, 5))
        
        icon_label = tk.Label(
            header,
            text=icon,
            font=("Segoe UI Emoji", 16),
            bg=self.theme.get("bg_tertiary", "#16213e"),
            fg=self.theme.get("accent_cyan", "#00d4ff")
        )
        icon_label.pack(side=tk.LEFT, padx=(0, 8))
        
        title_label = tk.Label(
            header,
            text=title,
            font=("Segoe UI", 9, "bold"),
            bg=self.theme.get("bg_tertiary", "#16213e"),
            fg=self.theme.get("text_secondary", "#94a3b8")
        )
        title_label.pack(side=tk.LEFT)
        
        value_frame = tk.Frame(inner, bg=self.theme.get("bg_tertiary", "#16213e"))
        value_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(3, 8))
        
        self.value_label = tk.Label(
            value_frame,
            text=value,
            font=("Segoe UI", 24, "bold"),
            bg=self.theme.get("bg_tertiary", "#16213e"),
            fg=self.theme.get("text_primary", "#ffffff")
        )
        self.value_label.pack(anchor="w")
        
        self.subtitle_label = tk.Label(
            value_frame,
            text=subtitle,
            font=("Segoe UI", 8),
            bg=self.theme.get("bg_tertiary", "#16213e"),
            fg=self.theme.get("text_secondary", "#94a3b8")
        )
        self.subtitle_label.pack(anchor="w", pady=(3, 0))
        
        self.progress_bar = None
        if show_progress:
            self.progress_bar = tk.Canvas(
                value_frame,
                height=4,
                bg=self.theme.get("bg_primary", "#0f0f23"),
                highlightthickness=0
            )
            self.progress_bar.pack(fill=tk.X, pady=(6, 0))
        
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
    
    def _on_enter(self, e):
        self.configure(highlightbackground=self.theme.get("accent_purple", "#a855f7"))
    
    def _on_leave(self, e):
        self.configure(highlightbackground=self.theme.get("accent_cyan", "#00d4ff"))
    
    def update_values(self, value="--", subtitle="", progress=None):
        self.value_label.config(text=value)
        self.subtitle_label.config(text=subtitle)
        
        if self.progress_bar and progress is not None:
            self._draw_progress(progress)
    
    def _draw_progress(self, percent):
        self.progress_bar.delete("all")
        width = self.progress_bar.winfo_width()
        if width <= 1:
            width = 300
        
        self.progress_bar.create_rectangle(
            0, 0, width, 4,
            fill=self.theme.get("bg_primary", "#0f0f23"),
            width=0
        )
        
        fill_width = int(width * (percent / 100))
        color = self._get_progress_color(percent)
        self.progress_bar.create_rectangle(
            0, 0, fill_width, 4,
            fill=color,
            width=0
        )
    
    def _get_progress_color(self, percent):
        if percent < 50:
            return self.theme.get("success", "#10b981")
        elif percent < 75:
            return self.theme.get("warning", "#f59e0b")
        else:
            return self.theme.get("error", "#ef4444")


class DashboardTab:
    """Modern dashboard tab with glassmorphism design"""

    def __init__(self, parent, theme, ml_manager=None):
        self.theme = theme
        self.frame = tk.Frame(parent, bg=theme["bg_primary"])

        # ML System Manager
        self.ml_manager = ml_manager
        self.parent = parent

        # Data storage
        self.cpu_history = deque(maxlen=60)
        self.memory_history = deque(maxlen=60)
        self.timestamps = deque(maxlen=60)

        self.setup_ui()

        # Force initial graph rendering after UI is built
        root = parent.winfo_toplevel()
        root.after(200, self._ensure_graph_display)
    
    def setup_ui(self):
        container = tk.Frame(self.frame, bg=self.theme["bg_primary"])
        container.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        canvas = tk.Canvas(container, bg=self.theme["bg_primary"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=self.theme["bg_primary"])

        scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        # create window and ensure inner frame width follows canvas width
        window_id = canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width))

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        def _on_touch_scroll(event):
            if hasattr(event, 'num') and event.num == 5:
                canvas.yview_scroll(1, "units")
            elif hasattr(event, 'num') and event.num == 4:
                canvas.yview_scroll(-1, "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_touch_scroll)
        canvas.bind_all("<Button-5>", _on_touch_scroll)

        self.canvas = canvas
        self.scrollable = scrollable

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.create_metrics_section(scrollable)
        self.create_graphs_section(scrollable)
        self.create_process_section(scrollable)
    
    def create_metrics_section(self, parent):
        section = tk.Frame(parent, bg=self.theme["bg_primary"])
        section.pack(fill=tk.X, padx=0, pady=0)

        title = tk.Label(
            section,
            text="SYSTEM OVERVIEW",
            font=("Segoe UI", 11, "bold"),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_secondary"]
        )
        title.pack(anchor="w", pady=(0, 5))

        metrics_container = tk.Frame(section, bg=self.theme["bg_primary"])
        metrics_container.pack(fill=tk.X, pady=(0, 5))

        for i in range(3):
            metrics_container.columnconfigure(i, weight=1)
        for i in range(3):
            metrics_container.rowconfigure(i, weight=1)

        self.cpu_card = GlassCard(
            metrics_container,
            title="CPU Usage",
            icon="🔥",
            value="0%",
            subtitle="0 cores active",
            theme=self.theme,
            show_progress=True
        )
        self.cpu_card.grid(row=0, column=0, sticky="nsew", padx=(0, 2), pady=(0, 2))

        self.memory_card = GlassCard(
            metrics_container,
            title="Memory",
            icon="💾",
            value="0%",
            subtitle="0 GB / 0 GB",
            theme=self.theme,
            show_progress=True
        )
        self.memory_card.grid(row=0, column=1, sticky="nsew", padx=1, pady=(0, 2))

        self.disk_card = GlassCard(
            metrics_container,
            title="Disk",
            icon="💿",
            value="0%",
            subtitle="0 GB / 0 GB",
            theme=self.theme,
            show_progress=True
        )
        self.disk_card.grid(row=0, column=2, sticky="nsew", padx=(2, 0), pady=(0, 2))

        self.network_card = GlassCard(
            metrics_container,
            title="Network",
            icon="🌐",
            value="0 MB",
            subtitle="↓ 0 MB  ↑ 0 MB",
            theme=self.theme
        )
        self.network_card.grid(row=1, column=0, sticky="nsew", padx=(0, 2), pady=1)

        self.process_card = GlassCard(
            metrics_container,
            title="Processes",
            icon="⚙️",
            value="0",
            subtitle="Active processes",
            theme=self.theme
        )
        self.process_card.grid(row=1, column=1, sticky="nsew", padx=1, pady=1)

        self.uptime_card = GlassCard(
            metrics_container,
            title="Uptime",
            icon="⏱️",
            value="0h 0m",
            subtitle="System running",
            theme=self.theme
        )
        self.uptime_card.grid(row=1, column=2, sticky="nsew", padx=(2, 0), pady=1)

        self.memory_forecast_card = GlassCard(
            metrics_container,
            title="Memory Forecast",
            icon="🔮",
            value="--",
            subtitle="Next 1 minute",
            theme=self.theme
        )
        self.memory_forecast_card.grid(row=2, column=0, sticky="nsew", padx=(0, 2), pady=(2, 0))

        self.forecast_accuracy_card = GlassCard(
            metrics_container,
            title="Forecast Accuracy",
            icon="🎯",
            value="--",
            subtitle="ML model performance",
            theme=self.theme
        )
        self.forecast_accuracy_card.grid(row=2, column=1, sticky="nsew", padx=1, pady=(2, 0))

        self.model_status_card = GlassCard(
            metrics_container,
            title="ML Status",
            icon="🧠",
            value="--",
            subtitle="System learning",
            theme=self.theme
        )
        self.model_status_card.grid(row=2, column=2, sticky="nsew", padx=(2, 0), pady=(2, 0))
    
    def create_graphs_section(self, parent):
        """Create performance graphs"""
        section = tk.Frame(parent, bg=self.theme["bg_primary"])
        section.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        title = tk.Label(
            section,
            text="PERFORMANCE ANALYTICS",
            font=("Segoe UI", 11, "bold"),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_secondary"]
        )
        title.pack(anchor="w", pady=(0, 5))

        graph_container = tk.Frame(section, bg=self.theme["bg_tertiary"])
        graph_container.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Store container and create a single matplotlib Figure + axes once (resized later)
        self.graph_container = graph_container

        dpi = 100
        # initial figure (will be resized when container gets geometry)
        self.fig = Figure(figsize=(10, 4), facecolor=self.theme["bg_tertiary"], dpi=dpi)
        self.cpu_ax = self.fig.add_subplot(121, facecolor=self.theme["bg_primary"])
        self.memory_ax = self.fig.add_subplot(122, facecolor=self.theme["bg_primary"])

        self.fig.tight_layout(pad=2.0)

        # Embed in tkinter
        self.matplotlib_canvas = FigureCanvasTkAgg(self.fig, master=graph_container)
        self.matplotlib_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bind container resize to update figure size dynamically
        graph_container.bind("<Configure>", lambda e: self.update_graph_size())

    def update_graph_size(self):
        """Resize the existing matplotlib Figure to match graph_container size."""
        try:
            if not hasattr(self, 'graph_container') or not hasattr(self, 'fig'):
                return

            w = max(self.graph_container.winfo_width(), 800)
            h = max(self.graph_container.winfo_height(), 400)

            dpi = self.fig.get_dpi() or 100
            fig_w = max(w / dpi, 4)
            fig_h = max(h / dpi, 3)

            # Update size of existing figure
            self.fig.set_size_inches(fig_w, fig_h, forward=True)
            self.fig.tight_layout(pad=2.0)

            if hasattr(self, 'matplotlib_canvas'):
                try:
                    self.matplotlib_canvas.draw_idle()
                except Exception:
                    self.matplotlib_canvas.draw()
        except Exception as e:
            print(f"update_graph_size error: {e}")

    def _ensure_graph_display(self):
        """Ensure graphs are displayed after UI initialization"""
        try:
            self.update_graph_size()
            if hasattr(self, 'matplotlib_canvas'):
                try:
                    self.matplotlib_canvas.draw_idle()
                except Exception:
                    self.matplotlib_canvas.draw()
        except Exception as e:
            print(f"Graph display error: {e}")

    def create_process_section(self, parent):
        section = tk.Frame(parent, bg=self.theme["bg_primary"])
        section.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        title_frame = tk.Frame(section, bg=self.theme["bg_primary"])
        title_frame.pack(fill=tk.X, pady=(0, 8))
        
        title = tk.Label(
            title_frame,
            text="PROCESS MANAGER",
            font=("Segoe UI", 11, "bold"),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_secondary"]
        )
        title.pack(side=tk.LEFT)
        
        refresh_btn = tk.Button(
            title_frame,
            text="🔄 Refresh",
            font=("Segoe UI", 9, "bold"),
            bg=self.theme["accent_cyan"],
            fg=self.theme["bg_primary"],
            activebackground=self.theme["accent_purple"],
            relief=tk.FLAT,
            padx=15,
            pady=6,
            cursor="hand2",
            command=self.refresh_processes
        )
        refresh_btn.pack(side=tk.RIGHT, padx=(8, 0))
        
        controls = tk.Frame(section, bg=self.theme["bg_primary"])
        controls.pack(fill=tk.X, pady=(0, 8))
        
        tk.Label(
            controls,
            text="🔍",
            font=("Segoe UI", 11),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_secondary"]
        ).pack(side=tk.LEFT, padx=(0, 6))
        
        self.search_var = tk.StringVar()
        self.search_var.trace('w', lambda *args: self.filter_processes())
        
        search_entry = tk.Entry(
            controls,
            textvariable=self.search_var,
            font=("Segoe UI", 10),
            bg=self.theme["bg_tertiary"],
            fg=self.theme["text_primary"],
            insertbackground=self.theme["accent_cyan"],
            relief=tk.FLAT,
            width=30
        )
        search_entry.pack(side=tk.LEFT, padx=(0, 15), ipady=6, ipadx=8)
        
        tk.Label(
            controls,
            text="Sort:",
            font=("Segoe UI", 9, "bold"),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_secondary"]
        ).pack(side=tk.LEFT, padx=(0, 6))
        
        self.sort_var = tk.StringVar(value="CPU %")
        sort_combo = ttk.Combobox(
            controls,
            textvariable=self.sort_var,
            values=["Name", "PID", "CPU %", "Memory %"],
            state="readonly",
            width=12,
            font=("Segoe UI", 9)
        )
        sort_combo.pack(side=tk.LEFT)
        sort_combo.bind('<<ComboboxSelected>>', lambda e: self.filter_processes())
        
        tree_container = tk.Frame(section, bg=self.theme["bg_tertiary"])
        tree_container.pack(fill=tk.BOTH, expand=True)
        
        vsb = ttk.Scrollbar(tree_container, orient="vertical")
        hsb = ttk.Scrollbar(tree_container, orient="horizontal")
        
        style = ttk.Style()
        style.configure(
            "Glass.Treeview",
            background=self.theme["bg_tertiary"],
            foreground=self.theme["text_primary"],
            fieldbackground=self.theme["bg_tertiary"],
            borderwidth=0,
            font=("Consolas", 9),
            rowheight=26
        )
        
        style.configure(
            "Glass.Treeview.Heading",
            background=self.theme["bg_primary"],
            foreground=self.theme["text_secondary"],
            borderwidth=0,
            font=("Segoe UI", 10, "bold")
        )
        
        style.map(
            "Glass.Treeview",
            background=[("selected", self.theme["accent_cyan"])],
            foreground=[("selected", self.theme["bg_primary"])]
        )
        
        columns = ("PID", "Name", "CPU %", "Memory %")
        self.process_tree = ttk.Treeview(
            tree_container,
            columns=columns,
            show="headings",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
            style="Glass.Treeview",
            height=15
        )
        
        vsb.config(command=self.process_tree.yview)
        hsb.config(command=self.process_tree.xview)
        
        def configure_treeview_columns(event=None):
            if self.process_tree.winfo_exists():
                total_width = self.process_tree.winfo_width()
                if total_width > 100:
                    pid_width = int(total_width * 0.15)
                    name_width = int(total_width * 0.55)
                    cpu_width = int(total_width * 0.15)
                    mem_width = int(total_width * 0.15)

                    self.process_tree.column("PID", width=max(pid_width, 60), minwidth=60, anchor="center")
                    self.process_tree.column("Name", width=max(name_width, 200), minwidth=200, anchor="w", stretch=True)
                    self.process_tree.column("CPU %", width=max(cpu_width, 80), minwidth=80, anchor="center")
                    self.process_tree.column("Memory %", width=max(mem_width, 80), minwidth=80, anchor="center")
                else:
                    self.process_tree.column("PID", width=60, anchor="center")
                    self.process_tree.column("Name", width=200, anchor="w", stretch=True)
                    self.process_tree.column("CPU %", width=80, anchor="center")
                    self.process_tree.column("Memory %", width=80, anchor="center")

        configure_treeview_columns()
        tree_container.bind("<Configure>", lambda e: configure_treeview_columns())
        
        self.process_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        tree_container.grid_rowconfigure(0, weight=1)
        tree_container.grid_columnconfigure(0, weight=1)
        
        control_frame = tk.Frame(section, bg=self.theme["bg_tertiary"])
        control_frame.pack(fill=tk.X, pady=(8, 0))
        
        inner = tk.Frame(control_frame, bg=self.theme["bg_tertiary"])
        inner.pack(fill=tk.X, padx=10, pady=10)
        
        self.selected_pid_label = tk.Label(
            inner,
            text="No process selected",
            font=("Segoe UI", 9),
            bg=self.theme["bg_tertiary"],
            fg=self.theme["text_secondary"]
        )
        self.selected_pid_label.pack(side=tk.LEFT)
        
        terminate_btn = tk.Button(
            inner,
            text="🛑 Terminate",
            font=("Segoe UI", 9, "bold"),
            bg=self.theme["error"],
            fg=self.theme["text_primary"],
            activebackground="#dc2626",
            relief=tk.FLAT,
            padx=15,
            pady=6,
            cursor="hand2",
            command=self.terminate_process
        )
        terminate_btn.pack(side=tk.RIGHT)
        
        self.process_tree.bind('<<TreeviewSelect>>', self.on_process_select)
        
        self.all_processes = []

    def on_resize(self, width, height):
        try:
            dpi = 100
            fig_width = max(width * 0.6 / dpi, 10)
            fig_height = max(height * 0.35 / dpi, 4)

            self.fig.set_size_inches(fig_width, fig_height)
            self.fig.tight_layout(pad=2.0)
            try:
                self.matplotlib_canvas.draw_idle()
            except Exception:
                self.matplotlib_canvas.draw()
        except Exception as e:
            print(f"Dashboard resize error: {e}")
    
    def update_data(self):
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_cores = psutil.cpu_count(logical=True)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            net_io = psutil.net_io_counters()
            boot_time = psutil.boot_time()
            uptime_seconds = psutil.time.time() - boot_time

            self.cpu_card.update_values(
                f"{cpu_percent:.1f}%",
                f"{cpu_cores} cores active",
                cpu_percent
            )

            self.memory_card.update_values(
                f"{memory.percent:.1f}%",
                f"{memory.used / (1024**3):.1f} GB / {memory.total / (1024**3):.1f} GB",
                memory.percent
            )

            self.disk_card.update_values(
                f"{disk.percent:.1f}%",
                f"{disk.used / (1024**3):.1f} GB / {disk.total / (1024**3):.1f} GB",
                disk.percent
            )

            self.network_card.update_values(
                f"{(net_io.bytes_sent + net_io.bytes_recv) / (1024**2):.0f} MB",
                f"↓ {net_io.bytes_recv / (1024**2):.0f} MB  ↑ {net_io.bytes_sent / (1024**2):.0f} MB"
            )

            self.process_card.update_values(
                str(len(psutil.pids())),
                "Active processes"
            )

            hours = int(uptime_seconds // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            self.uptime_card.update_values(
                f"{hours}h {minutes}m",
                "System running"
            )

            if self.ml_manager:
                self.update_ml_cards(memory.percent)

            self.timestamps.append(datetime.now())
            self.cpu_history.append(cpu_percent)
            self.memory_history.append(memory.percent)

            self.update_graphs()
            self.update_processes()

        except Exception as e:
            print(f"Dashboard update error: {e}")

    def update_ml_cards(self, current_memory_percent):
        try:
            if self.ml_manager.forecasting_active:
                forecast = self.ml_manager.get_memory_forecast()
                if forecast.prediction_memory_percent is not None:
                    trend_icon = "📈" if forecast.trend_direction == "up" else "📉" \
                               if forecast.trend_direction == "down" else "➡️"
                    self.memory_forecast_card.update_values(
                        f"{forecast.prediction_memory_percent:.1f}%",
                        f"Trend: {forecast.trend_direction} {trend_icon}"
                    )

                    if forecast.confidence_score is not None:
                        self.forecast_accuracy_card.update_values(
                            f"{forecast.confidence_score:.1f}",
                            f"Confidence: {forecast.confidence_score:.1f}"
                        )
                    else:
                        self.forecast_accuracy_card.update_values(
                            "Low",
                            "Confidence: Low"
                        )

            status = self.ml_manager.get_status_summary()
            if status["ml_available"]:
                training_status = "Training" if status["training_active"] else "Ready"
                model_count = sum([status["memory_forecaster_ready"], status["anomaly_detector_trained"]])
                self.model_status_card.update_values(
                    f"{model_count}/2",
                    f"{training_status} - {status.get('data_points_collected', 0)} samples"
                )
            else:
                self.model_status_card.update_values(
                    "No ML",
                    "Libraries not available"
                )

        except Exception as e:
            print(f"ML card update error: {e}")
            self.memory_forecast_card.update_values("--", "ML error")
            self.forecast_accuracy_card.update_values("--", "ML error")
            self.model_status_card.update_values("--", "ML error")
    
    def update_graphs(self):
        """Update performance graphs with forecast overlay"""
        if len(self.cpu_history) < 2:
            return

        cpu_list = list(self.cpu_history)
        memory_list = list(self.memory_history)
        x_data = list(range(len(cpu_list)))

        self.cpu_ax.clear()
        self.memory_ax.clear()

        # CPU Graph
        self.cpu_ax.set_facecolor(self.theme["bg_primary"])
        self.cpu_ax.plot(x_data, cpu_list, color=self.theme["accent_cyan"],
                        linewidth=2, antialiased=True, label="Current CPU")
        self.cpu_ax.fill_between(x_data, cpu_list, alpha=0.3,
                                color=self.theme["accent_cyan"])
        self.cpu_ax.set_title("CPU Usage", color=self.theme["text_primary"],
                             fontsize=11, fontweight='bold', pad=10)
        self.cpu_ax.set_ylim(0, 100)
        self.cpu_ax.grid(True, alpha=0.2, color=self.theme["text_secondary"])
        self.cpu_ax.tick_params(colors=self.theme["text_secondary"], labelsize=9)
        self.cpu_ax.legend(loc="upper right", fontsize=8)

        # Memory Graph with Forecast Overlay
        self.memory_ax.set_facecolor(self.theme["bg_primary"])

        # Plot actual memory data
        self.memory_ax.plot(x_data, memory_list, color=self.theme["accent_purple"],
                           linewidth=2, antialiased=True, label="Actual Memory")
        self.memory_ax.fill_between(x_data, memory_list, alpha=0.3,
                                    color=self.theme["accent_purple"])

        # Add forecast overlay if available
        if self.ml_manager and self.ml_manager.forecasting_active:
            try:
                forecast = self.ml_manager.get_memory_forecast()
                if forecast and forecast.forecast_values and len(forecast.forecast_values) > 0:
                    forecast_x = []
                    forecast_y = []

                    start_idx = len(memory_list) - 1
                    for i, forecast_val in enumerate(forecast.forecast_values[:10]):
                        forecast_x.append(start_idx + i + 1)
                        forecast_y.append(forecast_val)

                    self.memory_ax.plot(forecast_x, forecast_y,
                                       color="#ff6b9d", linestyle="--", linewidth=2,
                                       label="Memory Forecast")

                    self.memory_ax.axvline(x=start_idx + 0.5, color="#666666",
                                         linestyle=':', alpha=0.5)
            except Exception as e:
                pass

        self.memory_ax.set_title("Memory Usage & Forecast", color=self.theme["text_primary"],
                                fontsize=11, fontweight='bold', pad=10)
        self.memory_ax.set_ylim(0, 100)
        self.memory_ax.grid(True, alpha=0.2, color=self.theme["text_secondary"])
        self.memory_ax.tick_params(colors=self.theme["text_secondary"], labelsize=9)
        self.memory_ax.legend(loc="upper right", fontsize=8)

        try:
            try:
                self.matplotlib_canvas.draw_idle()
            except Exception:
                self.matplotlib_canvas.draw()
        except Exception as e:
            print(f"Graph draw error: {e}")
    
    def update_processes(self):
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                info = proc.info
                if info['cpu_percent'] is not None and info['memory_percent'] is not None:
                    processes.append({
                        'PID': info['pid'],
                        'Name': info['name'],
                        'CPU %': round(info['cpu_percent'], 2),
                        'Memory %': round(info['memory_percent'], 2)
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        self.all_processes = processes
        self.filter_processes()
    
    def filter_processes(self):
        for item in self.process_tree.get_children():
            self.process_tree.delete(item)
        
        search_term = self.search_var.get().lower()
        filtered = [p for p in self.all_processes 
                   if search_term in p['Name'].lower()]
        
        sort_by = self.sort_var.get()
        reverse = True if sort_by in ["CPU %", "Memory %"] else False
        filtered.sort(key=lambda x: x[sort_by], reverse=reverse)
        
        for proc in filtered[:50]:
            self.process_tree.insert("", tk.END, values=(
                proc['PID'],
                proc['Name'],
                f"{proc['CPU %']:.1f}%",
                f"{proc['Memory %']:.1f}%"
            ))
    
    def refresh_processes(self):
        self.update_processes()
    
    def on_process_select(self, event):
        selection = self.process_tree.selection()
        if selection:
            item = self.process_tree.item(selection[0])
            pid = item['values'][0]
            name = item['values'][1]
            self.selected_pid_label.config(
                text=f"Selected: {name} (PID: {pid})"
            )
    
    def terminate_process(self):
        selection = self.process_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a process first")
            return
        
        item = self.process_tree.item(selection[0])
        pid = item['values'][0]
        name = item['values'][1]
        
        if messagebox.askyesno("Confirm", f"Terminate {name} (PID: {pid})?"):
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                messagebox.showinfo("Success", f"Process {name} terminated")
                self.update_processes()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to terminate: {e}")
