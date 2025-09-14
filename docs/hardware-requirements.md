# Hardware Requirements

This document provides detailed hardware specifications and requirements for running the embedded AI benchmark suite.

## 🎯 Target Platforms

### NVIDIA Jetson Orin NX 8GB/16GB

**Recommended Model:** NVIDIA Jetson Orin NX 16GB Developer Kit

#### Core Specifications
- **CPU:** 6-core ARM Cortex-A78AE @ 2.0 GHz
- **GPU:** 1024-core NVIDIA Ampere architecture with 32 Tensor Cores
- **AI Accelerator:** 1x NVDLA v2.0 (Deep Learning Accelerator)
- **Memory:** 8GB or 16GB 128-bit LPDDR5 @ 102.4 GB/s
- **Peak AI Performance:** 70 TOPS (sparse INT8)
- **Power Envelope:** 10W - 20W (configurable)

#### Required Accessories
- **Cooling:** NVIDIA Jetson Orin NX/Nano Active Heatsink (mandatory)
- **Power Supply:** 19V/65W DC adapter
- **Storage:** microSD card (64GB+) or NVMe SSD
- **Connectivity:** Ethernet cable, USB-C for setup

### Qualcomm QCS6490 Development Kit

**Recommended Model:** Thundercomm TurboX C6490 Development Kit

#### Core Specifications
- **CPU:** 8-core Qualcomm Kryo 670 (1x A78 @ 2.7GHz, 3x A78 @ 2.4GHz, 4x A55 @ 1.9GHz)
- **GPU:** Qualcomm Adreno 643 @ 812 MHz
- **AI Accelerator:** Qualcomm Hexagon NPU with Tensor Accelerator
- **Memory:** 8GB LPDDR5 @ ~51.2 GB/s
- **Peak AI Performance:** 12-13 TOPS (INT8)
- **Power Envelope:** 5W - 12W (typical)

#### Required Accessories
- **Cooling:** Generic 40mm 5V fan with custom mounting (mandatory)
- **Power Supply:** 12V/3A DC adapter
- **Storage:** eUFS or microSD card (64GB+)
- **Connectivity:** Ethernet cable, USB-C for setup

### Radxa X4 (Intel N100)

**Recommended Model:** Radxa X4 with 16GB LPDDR5

#### Core Specifications
- **CPU:** 4-core Intel N100 (Alder Lake-N) @ 3.4 GHz (Turbo)
- **GPU:** Intel UHD Graphics (24 Execution Units)
- **AI Accelerator:** Intel GNA 3.0 (Gaussian & Neural Accelerator)
- **Memory:** 8GB or 16GB LPDDR5 @ ~38.4 GB/s
- **Peak AI Performance:** Not specified (varies by workload)
- **Power Envelope:** 6W TDP (configurable)

#### Required Accessories
- **Cooling:** Official Radxa Heatsink for X4 (recommended)
- **Power Supply:** USB-C PD 45W adapter
- **Storage:** microSD card (64GB+) or eMMC
- **Connectivity:** Ethernet cable, USB-C for power/data

## 🔌 Power Measurement Equipment

### Primary Recommendation: Yokogawa WT300E

**Model:** Yokogawa WT300E Digital Power Meter

#### Specifications
- **Accuracy:** 0.1% reading + 0.05% range
- **Update Rate:** 100ms (10 Hz)
- **Current Range:** 1mA to 20A
- **Voltage Range:** 15V to 600V
- **Connectivity:** Ethernet, USB, RS-232
- **Data Logging:** Built-in memory + PC software

#### Required Accessories
- **Current Shunt:** High-precision current measurement
- **Voltage Probes:** Differential voltage measurement
- **DC Power Breakout Board:** For series current measurement
- **Ethernet Cable:** For remote monitoring and logging

### Alternative Options

#### Budget Option: INA219/INA226 Modules
- **Accuracy:** ~1% (sufficient for relative comparisons)
- **Interface:** I2C (requires microcontroller/Raspberry Pi)
- **Cost:** <$20 per module
- **Limitation:** Lower accuracy, requires custom logging setup

#### Mid-Range Option: Keysight E36312A
- **Type:** Programmable DC power supply with built-in measurement
- **Accuracy:** 0.03% + 2mV
- **Current Range:** 3A max
- **Interface:** USB, LAN, GPIB

## 🌡️ Thermal Management (Mandatory)

### NVIDIA Jetson Orin NX
- **Requirement:** Active cooling mandatory
- **Recommended:** Official NVIDIA Active Heatsink
- **Alternative:** Noctua NF-A4x10 5V fan + heatsink
- **Mounting:** Standard mounting holes on carrier board

### Qualcomm QCS6490
- **Requirement:** Active cooling mandatory
- **Recommended:** 40mm 5V fan with custom mount
- **Heat Sink:** Aluminum heatsink with thermal interface material
- **Mounting:** Custom bracket or adhesive mounting

### Radxa X4
- **Requirement:** Heatsink recommended, fan optional
- **Recommended:** Official Radxa Heatsink for X4
- **Alternative:** Low-profile aluminum heatsink
- **Mounting:** Thermal pads included with official heatsink

### Thermal Monitoring
All platforms must maintain SoC temperatures below 85°C during sustained benchmarks to prevent thermal throttling.

## 💾 Storage Requirements

### Minimum Storage per Platform
- **System OS:** 32GB
- **Datasets:** 50GB
  - EuRoC MAV: ~1.2GB
  - KITTI: ~12GB
  - Cityscapes: ~11GB
- **Models:** 5GB
- **Results/Logs:** 10GB
- **Total Minimum:** 100GB

### Recommended Storage
- **microSD Card:** 128GB Class 10 (minimum)
- **Preferred:** NVMe SSD 256GB+ (where supported)
- **Performance Impact:** SSD significantly improves dataset loading times

## 🖥️ Host PC Requirements

A host PC is required for:
- SDK installation and configuration
- Dataset downloading and preparation
- Cross-compilation (for ARM platforms)
- Results analysis and visualization

### Minimum Host PC Specs
- **OS:** Ubuntu 20.04 LTS (x86_64)
- **CPU:** 4-core Intel/AMD processor
- **RAM:** 16GB
- **Storage:** 200GB free space
- **GPU:** Discrete NVIDIA GPU (for CUDA SDK setup)
- **Network:** Ethernet connection for device communication

## 🔗 Connectivity Requirements

### Network Setup
- **Ethernet:** Gigabit Ethernet recommended for all platforms
- **WiFi:** Available on most platforms but Ethernet preferred for stability
- **SSH Access:** Required for remote execution and monitoring

### Cables and Adapters
- **Ethernet Cables:** Cat6, various lengths
- **USB-C Cables:** For power and data (platform-specific)
- **HDMI/DisplayPort:** For initial setup (monitor connection)
- **Power Cables:** Platform-specific DC adapters

## ⚡ Power Supply Specifications

### NVIDIA Jetson Orin NX
- **Input:** 19V DC
- **Current:** 3.42A (65W)
- **Connector:** DC barrel jack (5.5mm x 2.1mm)
- **Efficiency:** >80%

### Qualcomm QCS6490
- **Input:** 12V DC
- **Current:** 3A (36W)
- **Connector:** DC barrel jack (varies by dev kit)
- **Efficiency:** >85%

### Radxa X4
- **Input:** USB-C PD
- **Power:** 45W minimum
- **Voltage:** 15V/20V (PD negotiation)
- **Connector:** USB-C

## 🧪 Laboratory Setup

### Environmental Conditions
- **Temperature:** 20-25°C ambient
- **Humidity:** <60% RH
- **Ventilation:** Adequate airflow around devices
- **Vibration:** Minimal (stable mounting surface)

### Safety Considerations
- **Electrical Safety:** Proper grounding, GFCI protection
- **Thermal Safety:** Temperature monitoring, thermal shutdown
- **Mechanical Safety:** Secure mounting, cable management

### Workspace Requirements
- **Bench Space:** 2m x 1m minimum
- **Power Outlets:** 6+ outlets with surge protection
- **Network Access:** Ethernet switch with 4+ ports
- **Lighting:** Adequate for cable identification and monitoring

## 📋 Pre-Purchase Checklist

### Essential Items
- [ ] Target platform(s) with appropriate memory configuration
- [ ] Active cooling solution for each platform
- [ ] High-precision power analyzer (Yokogawa WT300E recommended)
- [ ] DC power supplies with correct specifications
- [ ] Storage media (microSD cards/SSDs) with sufficient capacity
- [ ] Host PC meeting minimum requirements

### Optional but Recommended
- [ ] Backup storage media
- [ ] Additional cooling fans
- [ ] Thermal interface materials
- [ ] Cable management solutions
- [ ] Spare power adapters
- [ ] Network switch for multi-platform testing

### Budget Considerations
| Component | Budget Option | Recommended | High-End |
|-----------|---------------|-------------|----------|
| Power Analyzer | INA219 (~$20) | Yokogawa WT300E (~$2000) | Keysight E36312A (~$1500) |
| Cooling | Generic fans (~$10) | Platform-specific (~$50) | Custom liquid cooling (~$200) |
| Storage | microSD 128GB (~$20) | NVMe SSD 256GB (~$50) | NVMe SSD 1TB (~$150) |
| **Total per Platform** | **~$500** | **~$800** | **~$1200** |

## 🔧 Setup Verification

### Power System Verification
1. Verify power analyzer accuracy with known loads
2. Test current measurement in series configuration
3. Validate voltage measurement in parallel configuration
4. Confirm data logging functionality

### Thermal System Verification
1. Verify fan operation and airflow direction
2. Test thermal monitoring and alert systems
3. Confirm thermal interface material application
4. Validate temperature sensor readings

### Platform Verification
1. Verify boot sequence and OS installation
2. Test network connectivity and SSH access
3. Confirm SDK installation and functionality
4. Validate storage performance and capacity

This hardware setup ensures reliable, repeatable, and scientifically valid benchmark results across all target platforms.
