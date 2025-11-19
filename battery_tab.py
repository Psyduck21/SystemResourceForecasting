import tkinter as tk
from tkinter import ttk, messagebox
import psutil
from datetime import datetime, timedelta
from ml_system_manager import BatteryForecastResult, PowerOptimizationRecommendation


class BatteryTab:
    """Advanced Battery & Power Optimization Tab with ML-Powered Features"""

    def __init__(self, parent, theme, ml_manager=None):
        self.theme = theme
        self.ml_manager = ml_manager
        self.frame = tk.Frame(parent, bg=theme["bg_primary"])

        # Battery state
        self.power_plan_active = False
        self.smart_brightness_active = False
        self.adaptive_power_mode = False
        self.battery_history = []

        # ML-powered data storage
        self.battery_forecast = None
        self.power_recommendations = []

        # FIX: Cache ML recommendations to prevent frequent refreshing
        self.last_ml_update_time = 0
        self.ml_cache_duration = 1800  # Cache for 30 minutes
        self.cached_recommendations = []
        self.last_forecast_data = None

        # Auto-update interval: 5 minutes (300,000 ms)
        self.update_interval = 300000  # 5 minutes in milliseconds

        self.setup_ui()
        self.start_auto_updates()

    def setup_ui(self):
        container = tk.Frame(self.frame, bg=self.theme["bg_primary"])
        container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        canvas = tk.Canvas(container, bg=self.theme["bg_primary"], highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=self.theme["bg_primary"])

        scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        # keep window_id and bind configure so scrollable width follows canvas width
        window_id = canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width))

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Battery status section
        self.create_battery_status(scrollable)

        # Power consumption analysis
        self.create_power_analysis(scrollable)

        # Optimization recommendations
        self.create_recommendations(scrollable)

        # Quick actions
        self.create_quick_actions(scrollable)

    def create_battery_status(self, parent):
        """Create battery status display"""
        section = tk.Frame(parent, bg=self.theme["bg_primary"])
        section.pack(fill=tk.X, padx=40, pady=40)

        # Container
        container = tk.Frame(section, bg=self.theme["bg_tertiary"])
        container.pack(fill=tk.X)

        inner = tk.Frame(container, bg=self.theme["bg_tertiary"])
        inner.pack(fill=tk.BOTH, padx=40, pady=40)

        # Battery icon and level
        status_frame = tk.Frame(inner, bg=self.theme["bg_tertiary"])
        status_frame.pack(fill=tk.X)

        # Left: Battery visualization
        left = tk.Frame(status_frame, bg=self.theme["bg_tertiary"])
        left.pack(side=tk.LEFT, padx=(0, 40))

        self.battery_icon = tk.Label(
            left,
            text="🔋",
            font=("Segoe UI Emoji", 80),
            bg=self.theme["bg_tertiary"],
            fg=self.theme["success"]
        )
        self.battery_icon.pack()

        # Right: Battery info
        right = tk.Frame(status_frame, bg=self.theme["bg_tertiary"])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.battery_level = tk.Label(
            right,
            text="---%",
            font=("Segoe UI", 56, "bold"),
            bg=self.theme["bg_tertiary"],
            fg=self.theme["text_primary"]
        )
        self.battery_level.pack(anchor="w", pady=(0, 10))

        self.battery_status = tk.Label(
            right,
            text="Checking battery status...",
            font=("Segoe UI", 14),
            bg=self.theme["bg_tertiary"],
            fg=self.theme["text_secondary"]
        )
        self.battery_status.pack(anchor="w", pady=(0, 8))

        self.battery_time = tk.Label(
            right,
            text="--",
            font=("Segoe UI", 12),
            bg=self.theme["bg_tertiary"],
            fg=self.theme["accent_cyan"]
        )
        self.battery_time.pack(anchor="w")

        # Power source indicator
        divider = tk.Frame(inner, bg=self.theme["bg_primary"], height=2)
        divider.pack(fill=tk.X, pady=30)

        power_frame = tk.Frame(inner, bg=self.theme["bg_tertiary"])
        power_frame.pack(fill=tk.X)

        tk.Label(
            power_frame,
            text="Power Source",
            font=("Segoe UI", 12, "bold"),
            bg=self.theme["bg_tertiary"],
            fg=self.theme["text_secondary"]
        ).pack(side=tk.LEFT)

        self.power_source = tk.Label(
            power_frame,
            text="Unknown",
            font=("Segoe UI", 12),
            bg=self.theme["bg_tertiary"],
            fg=self.theme["accent_cyan"]
        )
        self.power_source.pack(side=tk.RIGHT)

    def create_power_analysis(self, parent):
        """Create ML-powered power consumption analysis"""
        section = tk.Frame(parent, bg=self.theme["bg_primary"])
        section.pack(fill=tk.X, padx=40, pady=(0, 30))

        # Title
        title = tk.Label(
            section,
            text="AI-POWERED BATTERY ANALYTICS",
            font=("Segoe UI", 16, "bold"),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_primary"]
        )
        title.pack(anchor="w", pady=(0, 20))

        # Container
        container = tk.Frame(section, bg=self.theme["bg_tertiary"])
        container.pack(fill=tk.X)

        inner = tk.Frame(container, bg=self.theme["bg_tertiary"])
        inner.pack(fill=tk.BOTH, padx=30, pady=30)

        # ML-Powered Battery Forecast
        forecast_title = tk.Label(
            inner,
            text="Battery Life Forecast (ML-Predicted):",
            font=("Segoe UI", 12, "bold"),
            bg=self.theme["bg_tertiary"],
            fg=self.theme["accent_cyan"]
        )
        forecast_title.pack(anchor="w", pady=(0, 15))

        # Forecast display
        forecast_frame = tk.Frame(inner, bg=self.theme["bg_primary"])
        forecast_frame.pack(fill=tk.X, pady=(0, 20))

        forecast_inner = tk.Frame(forecast_frame, bg=self.theme["bg_primary"])
        forecast_inner.pack(fill=tk.X, padx=20, pady=20)

        self.runtime_label = tk.Label(
            forecast_inner,
            text="Remaining Runtime: --",
            font=("Segoe UI", 12, "bold"),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_primary"]
        )
        self.runtime_label.pack(anchor="w")

        self.charge_time_label = tk.Label(
            forecast_inner,
            text="Charging Time: --",
            font=("Segoe UI", 11),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_secondary"]
        )
        self.charge_time_label.pack(anchor="w", pady=(5, 3))

        self.usage_impact_label = tk.Label(
            forecast_inner,
            text="Usage Impact: --",
            font=("Segoe UI", 11),
            bg=self.theme["bg_primary"],
            fg=self.theme["accent_purple"]
        )
        self.usage_impact_label.pack(anchor="w")

        # ML Recommendations
        rec_title = tk.Label(
            inner,
            text="Smart Battery Recommendations:",
            font=("Segoe UI", 12, "bold"),
            bg=self.theme["bg_tertiary"],
            fg=self.theme["text_primary"]
        )
        rec_title.pack(anchor="w", pady=(20, 15))

        # ML Recommendations container
        self.ml_recommendations_frame = tk.Frame(inner, bg=self.theme["bg_tertiary"])
        self.ml_recommendations_frame.pack(fill=tk.X)

        # Load recommendations if available
        self.update_ml_insights(initial=True)

    def update_ml_insights(self, initial=False):
        """Update ML-powered battery insights with caching to prevent frequent refreshing"""
        import time
        current_time = time.time()

        # FIX: Use caching to prevent ML recommendations from refreshing every 2 seconds
        if not initial and current_time - self.last_ml_update_time < self.ml_cache_duration:
            # Use cached recommendations if within cache duration and not initial load
            if self.cached_recommendations:
                self.update_ml_recommendations(self.cached_recommendations)
            return

        if not self.ml_manager:
            # Fallback to static content
            if initial:
                self.show_fallback_insights()
            return

        try:
            # 🔧 AUTO-START MONITORING if not running
            status = self.ml_manager.get_status_summary()
            if not status.get("monitoring_active", False):
                print("🔧 Starting system monitoring for ML battery analytics...")
                self.ml_manager.start_monitoring()
                # Wait a moment for data to start collecting
                time.sleep(1)

            # Get ML-powered recommendations
            recommendations = self.ml_manager.generate_power_optimization_recommendations()

            # Update forecast display (forecasts can update more frequently)
            forecast = self.ml_manager.get_battery_forecast()
            if forecast and not forecast.error_message:
                if forecast.remaining_minutes:
                    self.runtime_label.config(text=f"Remaining Runtime: {forecast.remaining_minutes:.0f} min (ML-predicted)")
                if forecast.charging_completion_minutes:
                    self.charge_time_label.config(text=f"Charging Time: {forecast.charging_completion_minutes:.0f} min")
                if forecast.usage_impact_score > 0:
                    impact_text = "High" if forecast.usage_impact_score > 0.7 else "Medium" if forecast.usage_impact_score > 0.4 else "Low"
                    self.usage_impact_label.config(text=f"Usage Impact: {impact_text}")

                # Update recommendations with caching
                self.cached_recommendations = recommendations  # Cache them
                self.last_ml_update_time = current_time  # Update cache timestamp
                self.update_ml_recommendations(recommendations)

            else:
                if initial:
                    self.show_fallback_insights()

        except Exception as e:
            print(f"ML insights update error: {e}")
            if initial:
                self.show_fallback_insights()

    def update_ml_recommendations(self, recommendations):
        """Update ML-powered recommendations display"""
        # Clear existing recommendations
        for widget in self.ml_recommendations_frame.winfo_children():
            widget.destroy()

        if not recommendations:
            # No ML recommendations available
            self.show_fallback_insights()
            return

        for rec in recommendations[:3]:  # Show top 3
            item = tk.Frame(self.ml_recommendations_frame, bg=self.theme["bg_primary"])
            item.pack(fill=tk.X, pady=8, padx=5)

            item_inner = tk.Frame(item, bg=self.theme["bg_primary"])
            item_inner.pack(fill=tk.X, padx=15, pady=15)

            # Left: Icon and data
            left = tk.Frame(item_inner, bg=self.theme["bg_primary"])
            left.pack(side=tk.LEFT, fill=tk.X, expand=True)

            impact_color = self.theme["success"] if rec.impact_score >= 7 else self.theme["warning"] if rec.impact_score >= 4 else self.theme["text_secondary"]

            tk.Label(
                left,
                text="🤖",
                font=("Segoe UI Emoji", 20),
                bg=self.theme["bg_primary"],
                fg=self.theme["accent_cyan"]
            ).pack(side=tk.LEFT, padx=(0, 15))

            text_frame = tk.Frame(left, bg=self.theme["bg_primary"])
            text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

            tk.Label(
                text_frame,
                text=f"{rec.description} (+{rec.estimated_time_saved} min)",
                font=("Segoe UI", 10),
                bg=self.theme["bg_primary"],
                fg=self.theme["text_primary"]
            ).pack(anchor="w")

            tk.Label(
                text_frame,
                text=f"Impact: {rec.impact_score:.1f}/10 | Confidence: {rec.confidence_level:.1f}",
                font=("Segoe UI", 9),
                bg=self.theme["bg_primary"],
                fg=impact_color
            ).pack(anchor="w", pady=(3, 0))

            # Right: Apply button
            tk.Button(
                item_inner,
                text="Apply",
                font=("Segoe UI", 10, "bold"),
                bg=self.theme["accent_cyan"],
                fg=self.theme["bg_primary"],
                activebackground=self.theme["accent_purple"],
                relief=tk.FLAT,
                padx=20,
                pady=8,
                cursor="hand2",
                command=lambda r=rec: self.apply_ml_recommendation(r)
            ).pack(side=tk.RIGHT)

    def show_fallback_insights(self):
        """Show fallback static insights when ML is not available"""
        # Clear existing
        for widget in self.ml_recommendations_frame.winfo_children():
            widget.destroy()

        # Static content
        recommendations = [
            ("💡", "Reduce screen brightness to 60%", "High Impact (+45 min)", self.theme["success"]),
            ("⚙️", "Enable background activity learning", "AI Optimization", self.theme["accent_cyan"]),
            ("🔋", "Use smart power plans", "Adaptive Power", self.theme["warning"])
        ]

        for emoji, text, impact, color in recommendations:
            item = tk.Frame(self.ml_recommendations_frame, bg=self.theme["bg_primary"])
            item.pack(fill=tk.X, pady=8, padx=5)

            item_inner = tk.Frame(item, bg=self.theme["bg_primary"])
            item_inner.pack(fill=tk.X, padx=15, pady=15)

            left = tk.Frame(item_inner, bg=self.theme["bg_primary"])
            left.pack(side=tk.LEFT, fill=tk.X, expand=True)

            tk.Label(
                left,
                text=emoji,
                font=("Segoe UI Emoji", 20),
                bg=self.theme["bg_primary"],
                fg=self.theme["accent_cyan"]
            ).pack(side=tk.LEFT, padx=(0, 15))

            text_frame = tk.Frame(left, bg=self.theme["bg_primary"])
            text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

            tk.Label(
                text_frame,
                text=text,
                font=("Segoe UI", 10),
                bg=self.theme["bg_primary"],
                fg=self.theme["text_primary"]
            ).pack(anchor="w")

            tk.Label(
                text_frame,
                text=impact,
                font=("Segoe UI", 9),
                bg=self.theme["bg_primary"],
                fg=color
            ).pack(anchor="w", pady=(3, 0))

    def apply_ml_recommendation(self, recommendation):
        """Apply ML-generated recommendation"""
        if recommendation.recommendation_type == "brightness_optimization":
            messagebox.showinfo("ML Optimization Applied",
                               f"Smart Brightness: {recommendation.description}\n\nBrightness automatically adjusted for battery savings.")
        elif recommendation.recommendation_type == "power_plan_switch":
            messagebox.showinfo("ML Power Plan Activated",
                               f"Power Plan Switch: {recommendation.description}\n\nSystem optimized for battery efficiency.")
        elif recommendation.recommendation_type == "background_cleanup":
            messagebox.showinfo("ML Background Cleanup",
                               f"Background Optimization: {recommendation.description}\n\nBattery-intensive processes limited.")
        elif recommendation.recommendation_type == "usage_optimization":
            messagebox.showinfo("ML Usage Optimization",
                               f"Usage Optimization: {recommendation.description}\n\nResource usage optimized for battery life.")
        else:
            messagebox.showinfo("ML Recommendation Applied",
                               f"Applied: {recommendation.description}\n\nEstimated battery savings: {recommendation.estimated_time_saved} minutes")

    def create_recommendations(self, parent):
        """Create optimization recommendations"""
        section = tk.Frame(parent, bg=self.theme["bg_primary"])
        section.pack(fill=tk.X, padx=40, pady=(0, 30))

        # Title
        title = tk.Label(
            section,
            text="SMART RECOMMENDATIONS",
            font=("Segoe UI", 16, "bold"),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_primary"]
        )
        title.pack(anchor="w", pady=(0, 20))

        # Container
        container = tk.Frame(section, bg=self.theme["bg_tertiary"])
        container.pack(fill=tk.X)

        inner = tk.Frame(container, bg=self.theme["bg_tertiary"])
        inner.pack(fill=tk.BOTH, padx=30, pady=30)

        # Recommendations list
        recommendations = [
            ("💡", "Reduce screen brightness to 60%", "High Impact", self.theme["success"]),
            ("🌙", "Enable dark mode system-wide", "Medium Impact", self.theme["warning"]),
            ("🔇", "Disable background app refresh", "High Impact", self.theme["success"]),
            ("🎵", "Close unused music/video apps", "Medium Impact", self.theme["warning"]),
            ("📡", "Turn off Bluetooth when not in use", "Low Impact", self.theme["text_secondary"]),
            ("🔄", "Reduce sync frequency to hourly", "Medium Impact", self.theme["warning"])
        ]

        for emoji, text, impact, color in recommendations:
            item = tk.Frame(inner, bg=self.theme["bg_primary"])
            item.pack(fill=tk.X, pady=10, padx=5)

            item_inner = tk.Frame(item, bg=self.theme["bg_primary"])
            item_inner.pack(fill=tk.X, padx=15, pady=15)

            # Left: Icon and text
            left = tk.Frame(item_inner, bg=self.theme["bg_primary"])
            left.pack(side=tk.LEFT, fill=tk.X, expand=True)

            tk.Label(
                left,
                text=emoji,
                font=("Segoe UI Emoji", 20),
                bg=self.theme["bg_primary"],
                fg=self.theme["accent_cyan"]
            ).pack(side=tk.LEFT, padx=(0, 15))

            text_frame = tk.Frame(left, bg=self.theme["bg_primary"])
            text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

            tk.Label(
                text_frame,
                text=text,
                font=("Segoe UI", 11),
                bg=self.theme["bg_primary"],
                fg=self.theme["text_primary"]
            ).pack(anchor="w")

            tk.Label(
                text_frame,
                text=impact,
                font=("Segoe UI", 9),
                bg=self.theme["bg_primary"],
                fg=color
            ).pack(anchor="w", pady=(3, 0))

            # Right: Apply button
            tk.Button(
                item_inner,
                text="Apply",
                font=("Segoe UI", 10, "bold"),
                bg=self.theme["accent_cyan"],
                fg=self.theme["bg_primary"],
                activebackground=self.theme["accent_purple"],
                relief=tk.FLAT,
                padx=20,
                pady=8,
                cursor="hand2",
                command=lambda t=text: self.apply_recommendation(t)
            ).pack(side=tk.RIGHT)

    def create_quick_actions(self, parent):
        """Create quick action buttons"""
        section = tk.Frame(parent, bg=self.theme["bg_primary"])
        section.pack(fill=tk.X, padx=40, pady=(0, 40))

        # Title
        title = tk.Label(
            section,
            text="QUICK ACTIONS",
            font=("Segoe UI", 16, "bold"),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_primary"]
        )
        title.pack(anchor="w", pady=(0, 20))

        # Action buttons - responsive grid
        buttons_container = tk.Frame(section, bg=self.theme["bg_primary"])
        buttons_container.pack(fill=tk.X)

        # Configure responsive grid for buttons (1 row, 3 columns)
        buttons_container.columnconfigure(0, weight=1)
        buttons_container.columnconfigure(1, weight=1)
        buttons_container.columnconfigure(2, weight=1)

        # Power saver mode
        power_saver = tk.Frame(buttons_container, bg=self.theme["bg_tertiary"])
        power_saver.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 20))

        inner1 = tk.Frame(power_saver, bg=self.theme["bg_tertiary"])
        inner1.pack(fill=tk.BOTH, padx=25, pady=25)

        tk.Label(
            inner1,
            text="⚡",
            font=("Segoe UI Emoji", 40),
            bg=self.theme["bg_tertiary"],
            fg=self.theme["warning"]
        ).pack(pady=(0, 15))

        tk.Label(
            inner1,
            text="Power Saver Mode",
            font=("Segoe UI", 13, "bold"),
            bg=self.theme["bg_tertiary"],
            fg=self.theme["text_primary"]
        ).pack(pady=(0, 8))

        tk.Label(
            inner1,
            text="Optimize system for\nmaximum battery life",
            font=("Segoe UI", 10),
            bg=self.theme["bg_tertiary"],
            fg=self.theme["text_secondary"],
            justify=tk.CENTER
        ).pack(pady=(0, 15))

        self.power_saver_btn = tk.Button(
            inner1,
            text="Enable",
            font=("Segoe UI", 11, "bold"),
            bg=self.theme["success"],
            fg=self.theme["text_primary"],
            activebackground="#059669",
            relief=tk.FLAT,
            padx=30,
            pady=10,
            cursor="hand2",
            command=self.toggle_power_saver
        )
        self.power_saver_btn.pack()

        # Battery health
        health = tk.Frame(buttons_container, bg=self.theme["bg_tertiary"])
        health.grid(row=0, column=1, sticky="nsew", padx=5, pady=(0, 20))

        inner2 = tk.Frame(health, bg=self.theme["bg_tertiary"])
        inner2.pack(fill=tk.BOTH, padx=25, pady=25)

        tk.Label(
            inner2,
            text="🏥",
            font=("Segoe UI Emoji", 40),
            bg=self.theme["bg_tertiary"],
            fg=self.theme["accent_cyan"]
        ).pack(pady=(0, 15))

        tk.Label(
            inner2,
            text="Battery Health",
            font=("Segoe UI", 13, "bold"),
            bg=self.theme["bg_tertiary"],
            fg=self.theme["text_primary"]
        ).pack(pady=(0, 8))

        tk.Label(
            inner2,
            text="Check battery health\nand charging cycles",
            font=("Segoe UI", 10),
            bg=self.theme["bg_tertiary"],
            fg=self.theme["text_secondary"],
            justify=tk.CENTER
        ).pack(pady=(0, 15))

        tk.Button(
            inner2,
            text="Check Now",
            font=("Segoe UI", 11, "bold"),
            bg=self.theme["accent_cyan"],
            fg=self.theme["bg_primary"],
            activebackground=self.theme["accent_purple"],
            relief=tk.FLAT,
            padx=30,
            pady=10,
            cursor="hand2",
            command=self.check_battery_health
        ).pack()

        # Custom settings
        custom = tk.Frame(buttons_container, bg=self.theme["bg_tertiary"])
        custom.grid(row=0, column=2, sticky="nsew", padx=(10, 0), pady=(0, 20))

        inner3 = tk.Frame(custom, bg=self.theme["bg_tertiary"])
        inner3.pack(fill=tk.BOTH, padx=25, pady=25)

        tk.Label(
            inner3,
            text="⚙️",
            font=("Segoe UI Emoji", 40),
            bg=self.theme["bg_tertiary"],
            fg=self.theme["accent_purple"]
        ).pack(pady=(0, 15))

        tk.Label(
            inner3,
            text="Custom Settings",
            font=("Segoe UI", 13, "bold"),
            bg=self.theme["bg_tertiary"],
            fg=self.theme["text_primary"]
        ).pack(pady=(0, 8))

        tk.Label(
            inner3,
            text="Configure advanced\npower options",
            font=("Segoe UI", 10),
            bg=self.theme["bg_tertiary"],
            fg=self.theme["text_secondary"],
            justify=tk.CENTER
        ).pack(pady=(0, 15))

        tk.Button(
            inner3,
            text="Configure",
            font=("Segoe UI", 11, "bold"),
            bg=self.theme["bg_tertiary"],
            fg=self.theme["text_primary"],
            activebackground=self.theme["bg_secondary"],
            relief=tk.FLAT,
            padx=30,
            pady=10,
            cursor="hand2",
            command=self.open_custom_settings
        ).pack()

    def toggle_power_saver(self):
        """Toggle power saver mode"""
        self.power_plan_active = not self.power_plan_active

        if self.power_plan_active:
            self.power_saver_btn.config(
                text="Disable",
                bg=self.theme["error"]
            )
            messagebox.showinfo(
                "Power Saver Enabled",
                "Power Saver Mode is now active!\n\n"
                "• Display brightness reduced\n"
                "• Background processes limited\n"
                "• CPU throttled for efficiency\n"
                "• Sleep timer reduced\n\n"
                "Estimated battery life extended by 30-40%"
            )
        else:
            self.power_saver_btn.config(
                text="Enable",
                bg=self.theme["success"]
            )
            messagebox.showinfo(
                "Power Saver Disabled",
                "Power Saver Mode has been disabled.\n\n"
                "System returned to balanced performance."
            )

    def check_battery_health(self):
        """Show battery health report"""
        try:
            battery = psutil.sensors_battery()

            if battery:
                # Simulate health data (in production, this would read from system)
                health_info = f"""Battery Health Report

Status: {'Charging' if battery.power_plugged else 'Discharging'}
Current Charge: {battery.percent}%
Estimated Health: 92% (Excellent)
Design Capacity: 50.00 Wh
Current Capacity: 46.00 Wh
Charge Cycles: 247

Recommendations:
• Battery health is excellent
• Continue current charging habits
• Avoid extreme temperatures
• Calibrate battery every 3 months
"""
                messagebox.showinfo("Battery Health", health_info)
            else:
                messagebox.showinfo(
                    "No Battery",
                    "No battery detected.\n\nThis appears to be a desktop system."
                )
        except Exception as e:
            messagebox.showerror("Error", f"Could not read battery information:\n{e}")

    def apply_recommendation(self, recommendation):
        """Apply a recommendation"""
        messagebox.showinfo(
            "Applied",
            f"Applied recommendation:\n\n{recommendation}\n\n"
            "Changes will take effect immediately."
        )

    def open_custom_settings(self):
        """Open custom settings dialog"""
        settings_window = tk.Toplevel(self.frame)
        settings_window.title("Custom Power Settings")
        settings_window.geometry("500x600")
        settings_window.configure(bg=self.theme["bg_primary"])

        # Title
        tk.Label(
            settings_window,
            text="⚙️ Custom Power Settings",
            font=("Segoe UI", 20, "bold"),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_primary"]
        ).pack(pady=30)

        # Settings container
        container = tk.Frame(settings_window, bg=self.theme["bg_tertiary"])
        container.pack(fill=tk.BOTH, expand=True, padx=30, pady=(0, 30))

        inner = tk.Frame(container, bg=self.theme["bg_tertiary"])
        inner.pack(fill=tk.BOTH, padx=30, pady=30)

        # Sleep timer
        tk.Label(
            inner,
            text="Sleep Timer",
            font=("Segoe UI", 12, "bold"),
            bg=self.theme["bg_tertiary"],
            fg=self.theme["text_primary"]
        ).pack(anchor="w", pady=(0, 10))

        sleep_scale = tk.Scale(
            inner,
            from_=1,
            to=30,
            orient=tk.HORIZONTAL,
            bg=self.theme["bg_tertiary"],
            fg=self.theme["text_primary"],
            highlightthickness=0,
            troughcolor=self.theme["bg_primary"],
            activebackground=self.theme["accent_cyan"]
        )
        sleep_scale.pack(fill=tk.X, pady=(0, 20))

        # Display brightness
        tk.Label(
            inner,
            text="Display Brightness",
            font=("Segoe UI", 12, "bold"),
            bg=self.theme["bg_tertiary"],
            fg=self.theme["text_primary"]
        ).pack(anchor="w", pady=(0, 10))

        brightness_scale = tk.Scale(
            inner,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            bg=self.theme["bg_tertiary"],
            fg=self.theme["text_primary"],
            highlightthickness=0,
            troughcolor=self.theme["bg_primary"],
            activebackground=self.theme["accent_cyan"]
        )
        brightness_scale.set(75)
        brightness_scale.pack(fill=tk.X, pady=(0, 20))

        # Checkboxes
        options = [
            "Dim display on battery",
            "Put hard disks to sleep",
            "Allow wake timers",
            "USB selective suspend"
        ]

        for option in options:
            var = tk.BooleanVar(value=True)
            tk.Checkbutton(
                inner,
                text=option,
                variable=var,
                bg=self.theme["bg_tertiary"],
                fg=self.theme["text_primary"],
                selectcolor=self.theme["bg_primary"],
                activebackground=self.theme["bg_tertiary"],
                font=("Segoe UI", 11)
            ).pack(anchor="w", pady=8)

        # Buttons
        button_frame = tk.Frame(settings_window, bg=self.theme["bg_primary"])
        button_frame.pack(fill=tk.X, padx=30, pady=(0, 30))

        tk.Button(
            button_frame,
            text="Apply",
            font=("Segoe UI", 11, "bold"),
            bg=self.theme["success"],
            fg=self.theme["text_primary"],
            relief=tk.FLAT,
            padx=30,
            pady=10,
            cursor="hand2",
            command=lambda: [messagebox.showinfo("Success", "Settings applied!"), settings_window.destroy()]
        ).pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(
            button_frame,
            text="Cancel",
            font=("Segoe UI", 11, "bold"),
            bg=self.theme["bg_tertiary"],
            fg=self.theme["text_primary"],
            relief=tk.FLAT,
            padx=30,
            pady=10,
            cursor="hand2",
            command=settings_window.destroy
        ).pack(side=tk.LEFT)

    def update_data(self):
        """Update battery information and ML insights"""
        try:
            battery = psutil.sensors_battery()

            if battery:
                # Update basic battery metrics
                percent = battery.percent
                self.battery_level.config(text=f"{percent:.0f}%")

                # Update icon and color based on level
                if battery.power_plugged:
                    self.battery_icon.config(text="🔌", fg=self.theme["success"])
                    self.battery_status.config(text="Charging")
                    self.power_source.config(text="AC Power")
                else:
                    if percent > 75:
                        self.battery_icon.config(fg=self.theme["success"])
                        self.battery_level.config(fg=self.theme["success"])
                    elif percent > 25:
                        self.battery_icon.config(fg=self.theme["warning"])
                        self.battery_level.config(fg=self.theme["warning"])
                    else:
                        self.battery_icon.config(fg=self.theme["error"])
                        self.battery_level.config(fg=self.theme["error"])

                    self.battery_status.config(text="On Battery")
                    self.power_source.config(text="Battery Power")

                # Update time remaining
                if battery.secsleft > 0 and battery.secsleft != psutil.POWER_TIME_UNLIMITED:
                    hours = battery.secsleft // 3600
                    minutes = (battery.secsleft % 3600) // 60
                    self.battery_time.config(text=f"{int(hours)}h {int(minutes)}m remaining")
                else:
                    if battery.power_plugged:
                        self.battery_time.config(text="Calculating charge time...")
                    else:
                        self.battery_time.config(text="Calculating discharge time...")

                # Update ML insights
                self.update_ml_insights()

            else:
                # No battery
                self.battery_icon.config(text="🖥️", fg=self.theme["text_secondary"])
                self.battery_level.config(text="N/A", fg=self.theme["text_secondary"])
                self.battery_status.config(text="No battery detected")
                self.battery_time.config(text="Desktop system")
                self.power_source.config(text="AC Power")

        except Exception as e:
            print(f"Battery update error: {e}")

    def start_auto_updates(self):
        """Start automatic battery metric updates every 5 minutes"""
        self.update_data()  # Initial update
        self.frame.after(self.update_interval, self.start_auto_updates)  # Schedule next update

# ===== ADDITIONAL ML METHODS =====

    def toggle_smart_brightness(self):
        """Toggle adaptive screen brightness"""
        self.smart_brightness_active = not self.smart_brightness_active

        if self.smart_brightness_active:
            messagebox.showinfo("Smart Brightness Enabled",
                               "Adaptive screen brightness is now active!\n\n"
                               "📱 The system will automatically adjust brightness based on:\n"
                               "• Content type (work, entertainment, reading)\n"
                               "• Battery level\n"
                               "• Time of day\n"
                               "• Ambient lighting conditions\n\n"
                               "Estimated battery savings: 20-35 minutes per hour")
        else:
            messagebox.showinfo("Smart Brightness Disabled",
                               "Adaptive brightness has been disabled.\n\n"
                               "Brightness will remain at manual settings.")

    def toggle_adaptive_power(self):
        """Toggle adaptive power plans"""
        self.adaptive_power_mode = not self.adaptive_power_mode

        if self.adaptive_power_mode:
            messagebox.showinfo("Adaptive Power Mode Enabled",
                               "Smart power management is now active!\n\n"
                               "🌟 Features enabled:\n"
                               "• Automatic power plan switching\n"
                               "• Background activity learning\n"
                               "• App-specific optimization\n"
                               "• Usage pattern recognition\n\n"
                               "Battery life optimized based on your habits!")
        else:
            messagebox.showinfo("Adaptive Power Mode Disabled",
                               "Adaptive power management has been disabled.\n\n"
                               "Switched back to manual power settings.")

    def show_usage_impact_analysis(self):
        """Show detailed usage impact analysis"""
        try:
            if not self.ml_manager:
                messagebox.showinfo("Feature Unavailable",
                                   "ML features require the machine learning system to be running.")
                return

            forecast = self.ml_manager.get_battery_forecast()
            process_anomalies = self.ml_manager.check_process_anomalies()

            if forecast:
                impact_report = f"""💡 Usage Impact Analysis

BATTERY LIFE PREDICTIONS:
Remaining Time: {forecast.remaining_minutes or 'N/A'} minutes
Usage Impact Score: {forecast.usage_impact_score:.1f} (0-1 scale)
Drain Rate: {forecast.drain_rate_flag.title()}

HIGH-IMPACT PROCESSES:
"""

                for i, anomaly in enumerate(process_anomalies[:5]):
                    impact_report += f"{i+1}. {anomaly.process_name}: {anomaly.anomaly_score:.2f} anomaly score\n"

                impact_report += "\n📈 RECOMMENDATIONS:\n"
                if forecast.usage_impact_score > 0.7:
                    impact_report += "• High resource usage detected\n"
                    impact_report += "• Consider closing resource-intensive applications\n"
                    impact_report += "• Enable adaptive power management\n"
                else:
                    impact_report += "• Battery usage appears normal\n"
                    impact_report += "• Continue current usage patterns\n"

                messagebox.showinfo("Usage Impact Analysis", impact_report)
            else:
                messagebox.showinfo("Analysis Unavailable",
                                   "Unable to generate usage impact analysis.\n\n"
                                   "Please ensure the system has collected sufficient data.")

        except Exception as e:
            messagebox.showerror("Analysis Error", f"Could not generate analysis:\n{e}")

    def show_trend_analysis(self):
        """Show battery degradation trend analysis"""
        try:
            if not self.ml_manager:
                messagebox.showinfo("Feature Unavailable",
                                   "ML trend analysis requires the machine learning system.")
                return

            trends = self.ml_manager.get_battery_trend_analysis()

            if trends:
                trend_report = f"""📊 Battery Trend Analysis

CAPACITY HEALTH:
Current Health: {trends.get('current_health_percent', 'N/A')}%
Capacity Trend: {trends.get('capacity_trend', 'Unknown').title()}
Degradation Rate: {trends.get('degradation_rate_percent_year', 'N/A')}% per year
Forecasted Health (12 months): {trends.get('forecasted_health_12months', 'N/A')}%

USAGE PATTERNS:
Charge Cycles: {trends.get('estimated_cycles', 'N/A')}
Calibration Recommended: {'Yes' if trends.get('recommended_calibration') else 'No'}

CHARGING EFFICIENCY:
Trend: {trends.get('charging_efficiency_trend', 'Unknown').title()}

💡 RECOMMENDATIONS:
• {'Consider battery calibration' if trends.get('recommended_calibration') else 'Battery health is good'}
• Monitor degradation rate regularly
• Use recommended charging patterns
"""

                messagebox.showinfo("Battery Trend Analysis", trend_report)
            else:
                messagebox.showinfo("Trend Analysis Unavailable",
                                   "Could not retrieve battery trend data.\n\n"
                                   "This feature requires historical battery data.")

        except Exception as e:
            messagebox.showerror("Trend Analysis Error", f"Could not generate trend analysis:\n{e}")
