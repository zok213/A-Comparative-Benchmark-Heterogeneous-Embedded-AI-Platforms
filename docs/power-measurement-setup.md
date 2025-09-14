# Power Measurement Setup Guide

This guide provides detailed instructions for setting up accurate, hardware-based power measurement using the Yokogawa WT300E Digital Power Analyzer for the embedded AI benchmark suite.

## 🎯 Overview

Accurate power measurement is critical for calculating meaningful performance-per-watt metrics. This setup ensures:

- **Ground Truth Measurement:** External hardware measurement vs. software estimates
- **Total System Power:** Captures SoC, memory, peripherals, and power delivery losses
- **High Temporal Resolution:** 10Hz sampling rate captures transient power spikes
- **Automated Logging:** Continuous data collection during benchmark execution

## 📋 Required Equipment

### Primary Equipment
- **Yokogawa WT300E Digital Power Analyzer**
- **DC Power Supply** (platform-specific voltage/current)
- **DC Power Breakout Board** (for series current measurement)
- **High-gauge Wire** (12-14 AWG for current path)
- **Banana Plug Cables** (for analyzer connections)

### Optional Equipment
- **Current Shunt Resistor** (if higher precision needed)
- **Oscilloscope** (for transient analysis)
- **Ethernet Cable** (for remote monitoring)

## 🔌 Hardware Connection Diagram

```
[DC Power Supply] -----> [WT300E Current Input] -----> [WT300E Current Output] -----> [DUT Power Input]
                              |                                                            |
                              |                                                            |
                         [WT300E Voltage Input] <-------------------------------------- [DUT Power Input]
                              |
                              |
                         [WT300E Voltage Common]
```

### Detailed Connection Steps

1. **Power Supply to WT300E Current Input:**
   - Connect positive terminal of DC supply to WT300E current input (+)
   - This allows current measurement in series with the load

2. **WT300E Current Output to DUT:**
   - Connect WT300E current output (+) to DUT power input (+)
   - Connect WT300E current output (-) to DUT power input (-)

3. **Voltage Measurement (Parallel):**
   - Connect WT300E voltage input (+) to DUT power input (+)
   - Connect WT300E voltage common to DUT power input (-)

## ⚙️ WT300E Configuration

### Initial Setup
1. **Power On:** Press the power button and wait for initialization
2. **Reset to Defaults:** Press `SHIFT` + `PRESET` to reset all settings
3. **Select Measurement Mode:** Press `FUNC` → `MEAS` → Select "DC"

### Measurement Configuration

#### Voltage Settings
```
FUNC → VOLTAGE → RANGE → Manual → 15V (for 12V supplies) or 30V (for 19V supplies)
FUNC → VOLTAGE → COUPLING → DC
```

#### Current Settings
```
FUNC → CURRENT → RANGE → Manual → 1A (for low power) or 5A (for high power)
FUNC → CURRENT → COUPLING → DC
```

#### Power Calculation
```
FUNC → POWER → MODE → DC
FUNC → POWER → RANGE → Auto
```

### Data Logging Setup

#### Internal Logging
```
FUNC → DATA → LOGGING → ON
FUNC → DATA → INTERVAL → 100ms (10Hz)
FUNC → DATA → ITEMS → P (Power), U (Voltage), I (Current)
FUNC → DATA → START → Manual
```

#### Network Logging (Recommended)
```
FUNC → COMM → ETHERNET → ON
FUNC → COMM → IP → Set static IP (e.g., 192.168.1.100)
FUNC → COMM → PROTOCOL → Modbus/TCP
```

## 🖥️ Software Setup

### Python Data Logging Script

Create a Python script for automated data collection:

```python
#!/usr/bin/env python3
"""
Yokogawa WT300E Power Measurement Logger
Logs power, voltage, and current data during benchmark execution
"""

import time
import csv
import socket
import struct
from datetime import datetime
import argparse

class WT300E_Logger:
    def __init__(self, ip_address='192.168.1.100', port=502):
        self.ip = ip_address
        self.port = port
        self.socket = None
        self.transaction_id = 0
        
    def connect(self):
        """Connect to WT300E via Modbus/TCP"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.ip, self.port))
            print(f"Connected to WT300E at {self.ip}:{self.port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from WT300E"""
        if self.socket:
            self.socket.close()
            self.socket = None
    
    def read_measurements(self):
        """Read power, voltage, and current measurements"""
        try:
            # Modbus function code 3 (Read Holding Registers)
            # Register addresses for WT300E (consult manual for exact addresses)
            power_reg = 0x1000    # Power register
            voltage_reg = 0x1001  # Voltage register  
            current_reg = 0x1002  # Current register
            
            measurements = {}
            
            # Read each measurement (simplified - actual implementation would
            # need proper Modbus protocol handling)
            measurements['timestamp'] = datetime.now().isoformat()
            measurements['power'] = self._read_register(power_reg)
            measurements['voltage'] = self._read_register(voltage_reg)
            measurements['current'] = self._read_register(current_reg)
            
            return measurements
            
        except Exception as e:
            print(f"Measurement read failed: {e}")
            return None
    
    def _read_register(self, address):
        """Read a single Modbus register (simplified implementation)"""
        # This is a simplified placeholder - actual implementation would
        # need complete Modbus protocol handling
        return 0.0
    
    def log_continuous(self, filename, duration=None, interval=0.1):
        """Log measurements continuously to CSV file"""
        
        fieldnames = ['timestamp', 'power', 'voltage', 'current']
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            start_time = time.time()
            
            try:
                while True:
                    measurements = self.read_measurements()
                    if measurements:
                        writer.writerow(measurements)
                        csvfile.flush()  # Ensure data is written immediately
                    
                    # Check duration limit
                    if duration and (time.time() - start_time) > duration:
                        break
                    
                    time.sleep(interval)
                    
            except KeyboardInterrupt:
                print("\nLogging stopped by user")

def main():
    parser = argparse.ArgumentParser(description='WT300E Power Logger')
    parser.add_argument('--ip', default='192.168.1.100', help='WT300E IP address')
    parser.add_argument('--output', required=True, help='Output CSV filename')
    parser.add_argument('--duration', type=int, help='Logging duration in seconds')
    parser.add_argument('--interval', type=float, default=0.1, help='Sampling interval')
    
    args = parser.parse_args()
    
    logger = WT300E_Logger(args.ip)
    
    if logger.connect():
        print(f"Starting power logging to {args.output}")
        print("Press Ctrl+C to stop logging")
        
        logger.log_continuous(
            filename=args.output,
            duration=args.duration,
            interval=args.interval
        )
        
        logger.disconnect()
        print("Logging completed")
    else:
        print("Failed to connect to WT300E")

if __name__ == "__main__":
    main()
```

### Usage Example

```bash
# Start power logging
python3 power_logger.py --output benchmark_power.csv --duration 300 &
LOGGER_PID=$!

# Run benchmark
./run_benchmark.sh

# Stop power logging
kill $LOGGER_PID

# Analyze power data
python3 analyze_power.py benchmark_power.csv
```

## 🔧 Platform-Specific Setup

### NVIDIA Jetson Orin NX

#### Power Input Specifications
- **Voltage:** 19V DC
- **Current:** Up to 3.42A (65W)
- **Connector:** 5.5mm x 2.1mm DC barrel jack

#### Measurement Setup
1. **WT300E Range Settings:**
   - Voltage: 30V range
   - Current: 5A range

2. **Breakout Board Connection:**
   - Cut the positive wire of the DC barrel cable
   - Insert WT300E current measurement in series
   - Maintain voltage measurement in parallel

### Qualcomm QCS6490

#### Power Input Specifications
- **Voltage:** 12V DC
- **Current:** Up to 3A (36W)
- **Connector:** DC barrel jack (varies by dev kit)

#### Measurement Setup
1. **WT300E Range Settings:**
   - Voltage: 15V range
   - Current: 5A range

2. **Custom Power Cable:**
   - Create custom cable with current measurement breakout
   - Ensure adequate wire gauge for current handling

### Radxa X4

#### Power Input Specifications
- **Voltage:** USB-C PD (15V/20V)
- **Current:** Up to 2.25A (45W)
- **Connector:** USB-C

#### Measurement Setup
1. **USB-C PD Trigger Board:**
   - Use PD trigger board to negotiate fixed voltage
   - Break out VBUS and GND for measurement

2. **WT300E Range Settings:**
   - Voltage: 30V range
   - Current: 5A range

## 📊 Data Analysis

### Power Analysis Script

```python
#!/usr/bin/env python3
"""
Power measurement analysis script
Calculates statistics and generates visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def analyze_power_data(csv_file):
    """Analyze power measurement data"""
    
    # Load data
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate statistics
    stats = {
        'mean_power': df['power'].mean(),
        'std_power': df['power'].std(),
        'min_power': df['power'].min(),
        'max_power': df['power'].max(),
        'p95_power': df['power'].quantile(0.95),
        'p99_power': df['power'].quantile(0.99),
        'total_energy': df['power'].sum() * 0.1 / 3600,  # Wh (assuming 0.1s interval)
        'duration': (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
    }
    
    # Print statistics
    print("Power Measurement Analysis")
    print("=" * 30)
    print(f"Duration: {stats['duration']:.1f} seconds")
    print(f"Mean Power: {stats['mean_power']:.2f} W")
    print(f"Std Dev: {stats['std_power']:.2f} W")
    print(f"Min Power: {stats['min_power']:.2f} W")
    print(f"Max Power: {stats['max_power']:.2f} W")
    print(f"P95 Power: {stats['p95_power']:.2f} W")
    print(f"P99 Power: {stats['p99_power']:.2f} W")
    print(f"Total Energy: {stats['total_energy']:.2f} Wh")
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Power over time
    ax1.plot(df['timestamp'], df['power'])
    ax1.set_ylabel('Power (W)')
    ax1.set_title('Power Consumption Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Power histogram
    ax2.hist(df['power'], bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(stats['mean_power'], color='red', linestyle='--', label=f"Mean: {stats['mean_power']:.2f}W")
    ax2.set_xlabel('Power (W)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Power Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Voltage and current
    ax3.plot(df['timestamp'], df['voltage'], label='Voltage (V)', alpha=0.7)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(df['timestamp'], df['current'], label='Current (A)', color='orange', alpha=0.7)
    ax3.set_ylabel('Voltage (V)')
    ax3_twin.set_ylabel('Current (A)')
    ax3.set_xlabel('Time')
    ax3.set_title('Voltage and Current Over Time')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('power_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze power measurement data')
    parser.add_argument('csv_file', help='Input CSV file from power logger')
    
    args = parser.parse_args()
    analyze_power_data(args.csv_file)
```

## 🔍 Troubleshooting

### Common Issues

#### 1. No Communication with WT300E
- **Check:** Ethernet cable connection
- **Check:** IP address configuration
- **Check:** Firewall settings
- **Solution:** Use direct Ethernet connection, configure static IP

#### 2. Incorrect Power Readings
- **Check:** Current measurement polarity
- **Check:** Voltage measurement connections
- **Check:** Measurement range settings
- **Solution:** Verify all connections match the diagram

#### 3. Noisy Measurements
- **Check:** Ground loops
- **Check:** Cable routing (away from switching supplies)
- **Check:** Measurement averaging settings
- **Solution:** Use twisted pair cables, enable averaging

#### 4. Data Logging Interruption
- **Check:** Network stability
- **Check:** Disk space
- **Check:** USB connection (if using USB logging)
- **Solution:** Use internal logging as backup

### Verification Procedures

#### 1. Accuracy Verification
```bash
# Test with known load (e.g., resistor)
# Calculate expected power: P = V²/R
# Compare with WT300E reading
```

#### 2. Timing Verification
```bash
# Sync WT300E clock with host PC
# Verify timestamp accuracy in logs
# Check for timing drift over long measurements
```

#### 3. Range Verification
```bash
# Test at minimum expected power
# Test at maximum expected power
# Verify no clipping or saturation
```

## 📝 Best Practices

### Setup Best Practices
1. **Cable Management:** Use shortest possible cables for current path
2. **Grounding:** Ensure proper grounding to avoid ground loops
3. **Calibration:** Perform regular calibration checks
4. **Documentation:** Record all connection details and settings

### Measurement Best Practices
1. **Warm-up:** Allow 30 minutes for WT300E warm-up
2. **Baseline:** Measure idle power before benchmark
3. **Synchronization:** Sync measurement start with benchmark start
4. **Redundancy:** Use multiple measurement methods when possible

### Data Quality Best Practices
1. **Sampling Rate:** Use 10Hz minimum for transient capture
2. **Duration:** Measure complete benchmark duration plus margins
3. **Filtering:** Apply appropriate filtering for noise reduction
4. **Validation:** Cross-check results with platform power estimates

This setup ensures accurate, repeatable power measurements that form the foundation for meaningful performance-per-watt analysis across all benchmark platforms.
