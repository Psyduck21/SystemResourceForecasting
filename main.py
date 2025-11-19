import tkinter as tk
from tkinter import ttk
import platform
from datetime import datetime
import os
import threading
import time

# Import tab modules
from dashboard_tab import DashboardTab
from anomaly_tab import AnomalyTab
from battery_tab import BatteryTab
from ml_system_manager import MLSystemManager

# Modern theme configuration
THEME = {
    "bg_primary": "#0f0f23",
    "bg_secondary": "#1a1a2e",
    "bg_tertiary": "#16213e",
    "accent_cyan": "#00d4ff",
    "accent_purple": "#a855f7",
    "accent_pink": "#ec4899",
    "text_primary": "#ffffff",
    "text_secondary": "#94a3b8",
    "success": "#10b981",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "glass_bg": "#1a1a2e",
}


class NeuralSystemMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Smart System Monitor")

        # Responsive window sizing with intelligent defaults
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Calculate responsive size (90% of screen width/height, but reasonable bounds)
        window_width = min(int(screen_width * 0.9), 1680)  # Max 1680px
        window_height = min(int(screen_height * 0.85), 980) # Max 980px
        window_width = max(window_width, 1200)  # Min 1200px
        window_height = max(window_height, 700)  # Min 700px

        # Center the window
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        self.root.configure(bg=THEME["bg_primary"])

        # Maximize the window for full screen coverage
        try:
            self.root.state('zoomed')  # Windows
        except:
            try:
                self.root.attributes('-zoomed', 1)  # Linux/Mac alternative
            except:
                pass  # Fallback to original size if maximization fails

        # Store window size for responsive updates
        self.window_width = window_width
        self.window_height = window_height

        self.running = True

        # Bind window resize events for responsiveness
        self.root.bind('<Configure>', self.on_window_resize)

        # Initialize ML System Manager
        self.ml_manager = MLSystemManager()
        self.ml_manager.start_monitoring()
        self.ml_manager.start_forecasting()  # Enable forecasting by default
        self.ml_manager.start_anomaly_detection()  # Enable anomaly detection by default

        self.setup_ui()

        # Give some time for initial data collection before starting updates
        time.sleep(0.5)

        # Start background update thread
        self.update_thread = threading.Thread(target=self.background_update, daemon=True)
        self.update_thread.start()
    
    def setup_ui(self):
        """Setup main UI structure"""
        # Compact header
        self.create_header()
        
        # Modern notebook with custom styling
        self.create_notebook()
        
        # Compact status bar
        self.create_status_bar()
    
    def create_header(self):
        """Create compact animated gradient header"""
        header = tk.Frame(self.root, bg=THEME["bg_secondary"], height=70)
        header.pack(fill=tk.X, padx=0, pady=0)
        header.pack_propagate(False)
        
        # Title with glow effect
        title_frame = tk.Frame(header, bg=THEME["bg_secondary"])
        title_frame.pack(expand=True)
        
        title = tk.Label(
            title_frame,
            text="🤖 SMART SYSTEM MONITOR",
            font=("Segoe UI", 24, "bold"),
            bg=THEME["bg_secondary"],
            fg=THEME["accent_cyan"]
        )
        title.pack(pady=(10, 2))
        
        subtitle = tk.Label(
            title_frame,
            text="AI-Powered Real-Time System Analytics",
            font=("Segoe UI", 10),
            bg=THEME["bg_secondary"],
            fg=THEME["text_secondary"]
        )
        subtitle.pack()
        
        # Animated underline
        underline = tk.Canvas(header, height=2, bg=THEME["bg_secondary"], highlightthickness=0)
        underline.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Create gradient effect with dynamic width
        gradient = underline.create_rectangle(0, 0, self.window_width, 2, fill=THEME["accent_cyan"], width=0)
        self.animate_gradient(underline, gradient, self.window_width)
    
    def animate_gradient(self, canvas, item, width):
        """Animate gradient underline"""
        if not self.running:
            return

        # Pulse effect
        colors = [THEME["accent_cyan"], THEME["accent_purple"], THEME["accent_pink"]]
        current_color = colors[int(time.time()) % len(colors)]
        canvas.itemconfig(item, fill=current_color)

        self.root.after(2000, lambda: self.animate_gradient(canvas, item, width))
    
    def create_notebook(self):
        """Create modern tabbed interface that fills available space"""
        # Custom style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Notebook styling - compact tabs
        style.configure(
            "Custom.TNotebook",
            background=THEME["bg_primary"],
            borderwidth=0,
            tabmargins=[5, 5, 5, 0]
        )
        
        style.configure(
            "Custom.TNotebook.Tab",
            background=THEME["bg_tertiary"],
            foreground=THEME["text_secondary"],
            padding=[20, 10],
            font=("Segoe UI", 10, "bold"),
            borderwidth=0
        )
        
        style.map(
            "Custom.TNotebook.Tab",
            background=[("selected", THEME["bg_secondary"])],
            foreground=[("selected", THEME["accent_cyan"])],
            expand=[("selected", [1, 1, 1, 0])]
        )
        
        # Create notebook - no padding to maximize space
        self.notebook = ttk.Notebook(self.root, style="Custom.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Initialize tabs
        self.dashboard_tab = DashboardTab(self.notebook, THEME, self.ml_manager)
        self.anomaly_tab = AnomalyTab(self.notebook, THEME, self.ml_manager)
        # self.battery_tab = BatteryTab(self.notebook, THEME, self.ml_manager)

        # Configure tab frames to expand properly
        self.dashboard_tab.frame.rowconfigure(0, weight=1)
        self.dashboard_tab.frame.columnconfigure(0, weight=1)
        self.anomaly_tab.frame.rowconfigure(0, weight=1)
        self.anomaly_tab.frame.columnconfigure(0, weight=1)
        # self.battery_tab.frame.rowconfigure(0, weight=1)
        # self.battery_tab.frame.columnconfigure(0, weight=1)

        # Add tabs with full expansion
        self.notebook.add(self.dashboard_tab.frame, text="🖥️  DASHBOARD", sticky="nsew")
        self.notebook.add(self.anomaly_tab.frame, text="🤖  ANOMALY DETECTION", sticky="nsew")
        # self.notebook.add(self.battery_tab.frame, text="🔋  BATTERY & POWER", sticky="nsew")
    
    def create_status_bar(self):
        """Create compact modern status bar"""
        status_bar = tk.Frame(self.root, bg=THEME["bg_secondary"], height=30)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        status_bar.pack_propagate(False)
        
        # System info
        sys_info = tk.Label(
            status_bar,
            text=f"🖥️ {platform.system()} {platform.release()} | Python {platform.python_version()}",
            font=("Segoe UI", 8),
            bg=THEME["bg_secondary"],
            fg=THEME["text_secondary"]
        )
        sys_info.pack(side=tk.LEFT, padx=15)
        
        # Status indicator
        self.status_label = tk.Label(
            status_bar,
            text="● System Operational",
            font=("Segoe UI", 8, "bold"),
            bg=THEME["bg_secondary"],
            fg=THEME["success"]
        )
        self.status_label.pack(side=tk.RIGHT, padx=15)
        
        # Clock
        self.clock_label = tk.Label(
            status_bar,
            text=datetime.now().strftime("%H:%M:%S"),
            font=("Segoe UI", 8),
            bg=THEME["bg_secondary"],
            fg=THEME["text_primary"]
        )
        self.clock_label.pack(side=tk.RIGHT, padx=10)
        self.update_clock()
    
    def update_clock(self):
        """Update clock in status bar"""
        if not self.running:
            return
        self.clock_label.config(text=datetime.now().strftime("%H:%M:%S"))
        self.root.after(1000, self.update_clock)
    
    def background_update(self):
        """Background thread for updates"""
        while self.running:
            try:
                # Update each tab
                self.dashboard_tab.update_data()
                self.anomaly_tab.update_data()
                # self.battery_tab.update_data()
                
                time.sleep(2)  # Update every 2 seconds
            except Exception as e:
                print(f"Update error: {e}")

    def on_window_resize(self, event):
        """Handle window resize events to maintain responsiveness"""
        # Only respond to window resize, not widget-level configure events
        if event.widget == self.root:
            new_width = event.width
            new_height = event.height

            # Only update if size actually changed (avoid excessive updates)
            if abs(new_width - self.window_width) > 10 or abs(new_height - self.window_height) > 10:
                self.window_width = new_width
                self.window_height = new_height

                # Trigger layout updates in tabs that need responsiveness
                try:
                    if hasattr(self, 'dashboard_tab') and hasattr(self.dashboard_tab, 'on_resize'):
                        self.dashboard_tab.on_resize(new_width, new_height)
                    if hasattr(self, 'anomaly_tab') and hasattr(self.anomaly_tab, 'on_resize'):
                        self.anomaly_tab.on_resize(new_width, new_height)
                    # if hasattr(self, 'battery_tab') and hasattr(self.battery_tab, 'on_resize'):
                    #     self.battery_tab.on_resize(new_width, new_height)
                except Exception as e:
                    print(f"Resize handling error: {e}")

    def on_closing(self):
        """Cleanup on exit"""
        self.running = False

        # Stop ML system manager
        if hasattr(self, 'ml_manager'):
            self.ml_manager.cleanup()

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


def main():
    try:
        # Parse command line arguments to handle -text option properly
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == "-text":
            # Run in headless/text mode without GUI
            print("🤖 SMART SYSTEM MONITOR - TEXT MODE")
            print("===================================")

            # Initialize ML manager without GUI
            ml_manager = MLSystemManager()
            ml_manager.start_monitoring()
            ml_manager.start_anomaly_detection()

            print("✅ ML System initialized - monitoring active")

            try:
                while True:
                    # Check for anomalies every 2 seconds
                    anomalies = ml_manager.check_process_anomalies()
                    if anomalies:
                        print(f"🚨 [{datetime.now().strftime('%H:%M:%S')}] {len(anomalies)} anomalies detected:")
                        for anomaly in anomalies:
                            print(f"   • {anomaly.process_name}: score {anomaly.anomaly_score:.2f} ({anomaly.anomaly_type})")
                    time.sleep(2)
            except KeyboardInterrupt:
                print("\n🛑 Stopping monitoring...")
                ml_manager.cleanup()
                print("✅ Cleanup complete")

        else:
            # Run with GUI as normal
            root = tk.Tk()
            app = NeuralSystemMonitor(root)
            root.protocol("WM_DELETE_WINDOW", app.on_closing)
            root.mainloop()

    except Exception as e:
        print(f"❌ Application error: {e}")
        if "unknown option" in str(e):
            print("\n💡 Tip: If you're getting 'unknown option' errors, try:")
            print("   python3 main.py -text    # for text-only mode")
            print("   python3 main.py          # for GUI mode (normal)")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
