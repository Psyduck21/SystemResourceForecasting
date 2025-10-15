# System Monitor with N-BEATS Forecasting

A real-time system monitoring application with memory usage forecasting using N-BEATS neural networks. Features a modern dark-themed GUI built with Tkinter and real-time performance graphs styled after GNOME System Monitor.

## Features

### 📊 System Monitoring
- **Real-time CPU usage** with smooth line graphs (GNOME blue)
- **Memory usage tracking** with forecasting capabilities (GNOME orange)
- **Process management** with search, filter, and priority control
- **System information** display (uptime, network, battery, disk usage)
- **Process termination** and priority adjustment (Windows/Linux support)
- **Auto-refresh** every 2 seconds with manual refresh option

### 🔮 Memory Forecasting
- **N-BEATS neural network** for memory usage prediction
- **Real-time forecasting** with next-step predictions (t+1)
- **Forecast statistics** and trend analysis (up/down/stable)
- **Visual comparison** between actual and predicted values
- **Purple dashed line** overlay showing forecasted values
- **Smart positioning** to avoid label overlap

### 🎨 User Interface
- **Dark theme** inspired by GNOME System Monitor
- **Responsive layout** with modern card-based design
- **Smooth animations** and anti-aliased graphs
- **Process filtering** and sorting capabilities
- **Fixed rolling window** (60 data points = 2 minutes)
- **Clean grid lines** and percentage labels

## Screenshots

The application displays:
- **System overview cards**: CPU, Memory, Disk, Battery, Processes, Uptime, Network, Platform
- **Real-time performance graphs**: CPU and Memory usage with GNOME-style dark theme
- **Memory forecasting overlay**: Purple dashed line showing predicted values
- **Process management**: Search, filter, sort, terminate, and priority control
- **Forecast statistics**: Trend analysis and prediction accuracy metrics

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd OS
```

2. **Create a virtual environment:**
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Ensure model files are present:**
   - `saved_model/config.json` - Model configuration
   - `saved_model/model.pth` - Trained N-BEATS weights
   - `saved_model/scaling.json` - Data normalization parameters
   
   **Note**: If model files are missing, the application will run without forecasting capabilities.

## Usage

### Running the Application

```bash
python interface.py
```

### Command Line Options

The application supports forecasting toggle:
```python
# Enable forecasting (default)
app = SystemMonitor(root, enable_forecasting=True)

# Disable forecasting
app = SystemMonitor(root, enable_forecasting=False)
```

### Features Overview

#### System Monitoring
- **Auto-refresh**: Updates every 2 seconds
- **Manual refresh**: Click "🔄 Refresh Now" button
- **Process management**: Select processes to terminate or change priority

#### Forecasting
- **Real-time predictions**: Memory usage forecasting every 2 seconds
- **Forecast statistics**: View mean, std deviation, trend analysis
- **Visual overlay**: Purple dashed line shows predicted values

#### Process Management
- **Search**: Filter processes by name
- **Sort**: By PID, Name, CPU%, Memory%, or Priority
- **Filter**: Show all, high CPU (>10%), or high memory (>5%) processes
- **Actions**: Terminate processes or change priority (requires elevated permissions)

## Project Structure

```
OS/
├── interface.py              # Main GUI application (Tkinter)
├── forecaster_mem.py         # N-BEATS forecasting engine + adapters
├── nbeats_utils.py          # Model loading utilities
├── saved_model/             # Pre-trained N-BEATS model
│   ├── config.json          # Model architecture config
│   ├── model.pth            # PyTorch model weights
│   └── scaling.json         # Data normalization params
├── data/                    # Training data and scripts
│   ├── system_metrics.csv   # Historical system data
│   ├── system_metrics_full.csv # Extended training dataset
│   ├── system_metrics_test.csv # Test dataset
│   ├── collect.py           # Data collection script
│   └── pretrained.ipynb     # Jupyter notebook for model training
├── realtime_data/           # Runtime data directory (auto-created)
├── myenv/                   # Virtual environment (gitignored)
├── phase2.docx              # Project documentation
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Configuration

### Model Configuration
The forecasting model uses N-BEATS architecture with:

#### Architecture Parameters
- **Backcast length**: 50 time steps (historical data window)
- **Forecast length**: 5 time steps (prediction horizon)
- **Hidden units**: 64 neurons per layer
- **Stack types**: ["trend", "seasonality"] (dual-stack architecture)
- **Blocks per stack**: 2 blocks per stack type
- **Total parameters**: ~50K trainable parameters

#### Runtime Configuration
- **Update interval**: 2 seconds
- **Smoothing window**: 3-7 points (moving average)
- **Buffer size**: 2000+ samples for model input
- **Sequence length**: 50 (matches backcast_length)
- **Prediction mode**: Next-step (t+1) forecasting

### UI Configuration
- **Graph window**: 60 data points (2 minutes at 2s intervals)
- **Colors**: GNOME-inspired dark theme (#1e1e1e background)
- **Refresh rate**: 2 seconds (configurable)
- **Process limit**: Top 50 processes displayed
- **Auto-cleanup**: CSV files deleted on exit

## Dependencies

### Core Dependencies
- **tkinter**: GUI framework (included with Python)
- **psutil**: System monitoring
- **matplotlib**: Graph plotting
- **numpy**: Numerical computations
- **pandas**: Data handling

### Machine Learning
- **torch**: PyTorch for N-BEATS model
- **nbeats-pytorch**: N-BEATS implementation

### Optional
- **scikit-learn**: For additional ML utilities
- **tensorflow**: Alternative Keras models

## Model Training

The N-BEATS model was trained on system memory usage data. To retrain:

1. **Collect training data:**
```bash
python data/collect.py
```

2. **Train the model:**
```python
# Use the Jupyter notebook: data/pretrained.ipynb
# Contains complete training pipeline and model evaluation
```

3. **Available datasets:**
   - `data/system_metrics.csv` - Basic training data
   - `data/system_metrics_full.csv` - Extended dataset
   - `data/system_metrics_test.csv` - Test dataset

4. **Save model artifacts:**
   - Model weights: `saved_model/model.pth` (~768 lines of PyTorch state dict)
   - Configuration: `saved_model/config.json` (architecture parameters)
   - Scaling parameters: `saved_model/scaling.json` (data normalization)

#### Model Architecture Details
```json
{
  "backcast_length": 50,
  "forecast_length": 5,
  "hidden_units": 64,
  "stack_types": ["trend", "seasonality"],
  "nb_blocks_per_stack": 2
}
```

## Troubleshooting

### Common Issues

1. **Import errors for torch/nbeats-pytorch:**
   ```bash
   pip install torch nbeats-pytorch
   ```
   **Note**: If ML dependencies are missing, the app runs without forecasting.

2. **Permission denied for process management:**
   - Run with elevated permissions (sudo on Linux)
   - Some processes may be protected by the system
   - Windows: Run as Administrator

3. **Model loading errors:**
   - Ensure `saved_model/` directory contains all required files
   - Check file permissions and paths
   - App will disable forecasting if model files are missing

4. **Graph not displaying:**
   - Verify matplotlib backend compatibility
   - Check for display issues in headless environments
   - Ensure tkinter is properly installed

5. **Forecasting not working:**
   - Check if `nbeats_utils.py` is present
   - Verify model files in `saved_model/`
   - Look for error messages in console output

### Performance Optimization

- **Reduce update frequency** for lower CPU usage
- **Disable forecasting** if not needed
- **Limit process list** to top N processes
- **Use smaller smoothing windows** for faster rendering

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **N-BEATS**: Neural basis expansion analysis for interpretable time series forecasting
- **GNOME System Monitor**: UI design inspiration
- **psutil**: Excellent system monitoring library
- **PyTorch**: Deep learning framework

## Future Enhancements

- [ ] CPU usage forecasting
- [ ] Network traffic prediction
- [ ] Disk usage forecasting
- [ ] Custom alert thresholds
- [ ] Export functionality for data
- [ ] Plugin system for custom metrics
- [ ] Web dashboard interface
- [ ] Mobile app companion

## Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed information
4. Include system information and error logs

---

**Note**: This application is designed for educational and monitoring purposes. Use process termination features with caution, especially on production systems.
