# Unified Analysis Scripts

This directory contains platform-agnostic analysis scripts that work across all embedded AI platforms in the benchmark suite.

## Available Scripts

### ORB-SLAM3 Analysis

#### `orb_slam3_verify_metrics.py`
**Purpose**: Comprehensive verification of all ORB-SLAM3 metrics
- Validates dataset integrity and ground truth data
- Verifies trajectory accuracy calculations (RMSE, ATE, RPE)
- Checks performance metrics and statistical correctness
- Ensures benchmark compliance and reproducibility

**Usage**: 
```bash
# From any platform directory
python ../../analysis/orb_slam3_verify_metrics.py
```

#### `orb_slam3_calculate_metrics.py`
**Purpose**: Calculate standard scientific metrics for publication
- Trajectory accuracy metrics (RMSE, ATE, RPE)
- Processing performance (FPS, latency, resource usage)
- Platform-specific optimizations and bottlenecks
- Standardized benchmark scores (0-100 scale)

**Usage**:
```bash
# From any platform directory  
python ../../analysis/orb_slam3_calculate_metrics.py
```

#### `orb_slam3_advanced_analysis.py` 
**Purpose**: Advanced statistical analysis for research publication
- Rigorous statistical analysis with confidence intervals
- Cross-platform comparative analysis
- Publication-ready insights and recommendations
- Research contribution assessment

**Usage**:
```bash
# From any platform directory
python ../../analysis/orb_slam3_advanced_analysis.py
```

## Platform Detection

All scripts automatically detect the current platform based on the working directory:

| Platform | Detection | Emoji | Architecture |
|----------|-----------|-------|--------------|
| NVIDIA Jetson | `nvidia-jetson` in path | 🚀 | ARM64 + Ampere GPU |
| Qualcomm QCS6490 | `qualcomm-qcs6490` in path | 🔥 | ARM64 + Adreno GPU |
| Radxa CM5 | `radxa-cm5` in path | ⚡ | ARM64 + Mali GPU + NPU |

## Benefits of Unified Approach

### Before (Duplicated Scripts)
- ❌ 3 copies of `verify_all_metrics.py` (one per platform)
- ❌ 1 copy of `calculate_metrics.py` (Radxa only)
- ❌ 1 copy of `advanced_metrics_analysis.py` (Radxa only)
- ❌ Inconsistent analysis across platforms
- ❌ Manual synchronization of bug fixes

### After (Unified Scripts)
- ✅ Single source of truth for all analysis logic
- ✅ Consistent metrics calculation across platforms
- ✅ Automatic platform detection and adaptation
- ✅ Bug fixes and improvements apply to all platforms
- ✅ Easier maintenance and updates

## Platform-Specific Outputs

While the scripts are unified, they provide platform-specific:
- Performance characteristics and bottlenecks
- Hardware optimization recommendations  
- Power efficiency comparisons
- Research significance and novel contributions

## Research Applications

These scripts support:
- **Comparative Studies**: Direct comparison across embedded AI platforms
- **Scientific Publication**: Publication-ready metrics with statistical rigor
- **Performance Optimization**: Platform-specific bottleneck identification
- **Reproducible Research**: Consistent methodology across all platforms

## Integration with Benchmark Suite

The unified analysis scripts integrate seamlessly with:
- Platform-specific benchmark runs in `platforms/*/orb-slam3/`
- Centralized dataset management in `datasets/`
- Overall benchmark orchestration via `run_all_benchmarks.sh`

## Future Extensions

The unified approach enables easy addition of:
- New embedded AI platforms
- Additional SLAM algorithms (beyond ORB-SLAM3)
- Extended metrics and analysis techniques
- Automated report generation

## Example Workflow

```bash
# 1. Run ORB-SLAM3 benchmark on specific platform
cd platforms/nvidia-jetson/orb-slam3/
./run_benchmark.sh

# 2. Verify all metrics are correct
python ../../analysis/orb_slam3_verify_metrics.py

# 3. Calculate comprehensive metrics
python ../../analysis/orb_slam3_calculate_metrics.py

# 4. Generate advanced analysis for publication
python ../../analysis/orb_slam3_advanced_analysis.py
```

This approach eliminates duplication while ensuring consistent, high-quality analysis across all embedded AI platforms.