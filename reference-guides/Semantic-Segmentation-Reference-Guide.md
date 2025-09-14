

# **Guide to Benchmarking Semantic Segmentation on Heterogeneous Accelerators**

## **Introduction**

**Objective:** This guide presents a definitive, technically rigorous, and reproducible methodology for benchmarking the performance and efficiency of the DDRNet-23-slim semantic segmentation model. The protocol is designed to provide engineers, researchers, and system architects with a reliable framework for evaluating AI inference capabilities on diverse embedded hardware.

**Scope:** The experimental protocol covers four distinct embedded AI compute targets, representing a heterogeneous landscape of modern edge accelerators. The platforms under evaluation are the NVIDIA Jetson Orin NX 8GB (targeting both its Ampere architecture GPU and its NVDLA v2.0 accelerator), the Qualcomm QCS6490 (targeting its Hexagon NPU), and the Radxa X4 with an Intel N100 processor (targeting its integrated Intel UHD Graphics).

**Motivation:** The proliferation of specialized AI hardware has created an urgent need for standardized, reliable benchmarking practices. Superficial metrics and inconsistent testing methodologies often lead to misleading conclusions, hindering effective technology selection and deployment.1 This guide seeks to address this challenge by establishing a "benchmark that doesn't lie"—a protocol founded on principles of methodological rigor, ground-truth measurement, and controlled variables. By providing a holistic performance profile that encompasses throughput, latency, power consumption, and energy efficiency, this framework enables a nuanced and realistic comparison of hardware capabilities, thereby informing critical, real-world deployment decisions.3

---

## **🔬 Core Principles for a Reliable Benchmark**

A credible benchmark is built upon a foundation of principles that ensure fairness, accuracy, and reproducibility. The following core tenets are non-negotiable for generating meaningful and comparable results across the diverse hardware and software ecosystems under evaluation.

### **The Imperative of Native SDKs for Peak Performance**

Modern AI accelerators are not monolithic compute units; they are complex, heterogeneous systems whose full potential can only be realized through vendor-specific Software Development Kits (SDKs).4 Attempting to benchmark these platforms using a generic runtime, such as a vanilla implementation of ONNX Runtime, would fail to engage the hardware-specific optimizations that are the primary differentiators of these platforms.

SDKs like NVIDIA's TensorRT, Qualcomm's Neural Processing (NP) SDK, and Intel's OpenVINO are sophisticated compiler toolchains, not simple model loaders.6 They perform a series of critical, hardware-aware optimizations during a "build" or "compilation" phase, which may include:

* **Layer and Tensor Fusion:** Merging sequential operations (e.g., a convolution followed by a bias addition and a ReLU activation) into a single, highly optimized kernel. This reduces memory bandwidth requirements and kernel launch overhead.  
* **Kernel Auto-Tuning:** Selecting the fastest CUDA or OpenCL kernel implementation for a given operation and set of parameters from a library of hardware-specific kernels.  
* **Precision Calibration:** Analyzing the model with sample data to determine the optimal scaling factors for INT8 quantization, minimizing accuracy loss while maximizing performance gains.  
* **Hardware-Aware Data Layout Transformations:** Reordering tensor data in memory (e.g., from NCHW to a channel-blocked format like NHWC8) to match the accelerator's preferred data access patterns, thereby maximizing memory throughput.

These optimizations are unique to each hardware architecture and are inaccessible through generic APIs.7 Consequently, the exclusive use of each platform's native SDK is a fundamental requirement of this protocol. To do otherwise would constitute a methodological flaw, resulting in an unfair and inaccurate assessment of the hardware's true inference capabilities.

### **Ground Truth Measurement: The Necessity of Hardware-Based Power Analysis**

Software-based power estimation tools, while convenient, are often insufficient for rigorous benchmarking. They may rely on incomplete models, fail to capture the total power draw of the entire System-on-Chip (SoC), or have a sampling rate too low to accurately represent the transient power spikes characteristic of AI workloads.

To establish a ground truth for power consumption and energy efficiency, this protocol mandates the use of an external, high-precision digital power analyzer. This instrument must be connected in series with the DC power supply input of the development board, measuring the total voltage and current consumed by the entire system during the workload.10 This method provides a holistic and indisputable measurement of the power required to perform the inference task.

Furthermore, the nature of this measurement must be sustained over time. Research has demonstrated that "short-duration measurement is not reliable" due to the dynamic behavior of embedded systems, where clock speeds ramp, memory controllers activate, and accelerators draw power in bursts.11 A brief measurement might capture a peak, a trough, or a transient state, leading to a highly variable and non-representative result. By executing the inference task in a loop for a significant number of iterations (e.g., 1000\) and recording power continuously throughout this period, it is possible to average out the noise and capture the true, steady-state power consumption of the workload. This principle directly informs the design of the benchmarking scripts in this guide, which are structured to facilitate stable and reliable power measurement.

### **Establishing a Fair Baseline: Consistent INT8 Quantization**

INT8 quantization is a cornerstone technique for optimizing model performance on edge devices. By representing weights and activations with 8-bit integers instead of 32-bit floating-point numbers, it reduces the model's memory footprint, decreases memory bandwidth requirements, and allows the hardware to leverage specialized, high-throughput integer arithmetic units.12

To ensure a fair comparison, this guide enforces a consistent Post-Training Quantization (PTQ) strategy across all platforms. PTQ is selected as it allows for the optimization of a pre-trained model without requiring access to the original training pipeline, which is a common scenario in deployment.

A critical, and often overlooked, variable in the PTQ process is the calibration dataset. The quality and representativeness of the data used to calibrate the quantization process are paramount for minimizing accuracy degradation.12 Each platform's SDK provides its own tools for PTQ, but the input to these tools must be controlled. If one platform is calibrated with a different set of images than another, any resulting differences in performance or accuracy could be attributed to the calibration data itself, introducing a confounding variable that invalidates the comparison.

Therefore, to create a scientifically sound comparison of the platforms' respective hardware and software stacks, the calibration input must be identical. This protocol mandates the creation of a single, shared calibration dataset—a specific subset of 500 images from the Cityscapes validation set. This same set of images will be used as input for the PTQ step on all four target platforms. This ensures that the model optimization process for each device begins from an identical baseline, effectively isolating the hardware and its native SDK as the primary variables under test.

---

## **🛠️ Required Setup and Prerequisites**

Reproducibility begins with a precise and unambiguous definition of the required hardware, software, and data assets. The following tables provide a complete manifest for this experiment.

### **Table 1: Hardware Bill of Materials**

| Item | Specific Model/Part Number | Purpose | Notes |
| :---- | :---- | :---- | :---- |
| NVIDIA Platform | NVIDIA Jetson Orin NX 8GB Developer Kit | Target Device 1 & 2 | Includes the Orin NX 8GB module and a carrier board.15 |
| Qualcomm Platform | Thundercomm TurboX C6490 Development Kit | Target Device 3 | A representative development kit for the QCS6490 SoC.18 |
| Intel Platform | Radxa X4 (Intel N100, 16GB RAM variant) | Target Device 4 | A compact Single-Board Computer (SBC) featuring the Alder Lake-N processor.21 |
| Power Analyzer | Yokogawa WT300E Digital Power Meter | Power Measurement | High-precision instrument for DC power analysis, capable of measuring low currents and integrating energy over time.24 |
| Active Cooling | \- NVIDIA: Official NVIDIA Jetson ORIN NX/ORIN Nano Active Heatsink \- Qualcomm: Generic 40mm 5V Fan (requires custom mounting) \- Radxa: Official Radxa Heatsink for X4 | Thermal Management | **Mandatory.** Prevents thermal throttling during sustained benchmarks. Various third-party solutions are available.27 |
| Host PC | x86-64 PC with a discrete NVIDIA GPU | Flashing & SDK Host | Required for running NVIDIA SDK Manager and for cross-compilation tasks for other platforms. |
| Peripherals | Monitor, Keyboard, Mouse, Ethernet Cables, USB-C Cables, DC Power Breakout Board | System Operation | Standard laboratory equipment for setup, operation, and power measurement. |

### **Table 2: Software and Asset Manifest**

| Component | Version | Source/Download Link | Purpose |
| :---- | :---- | :---- | :---- |
| Host/Target OS | Ubuntu 20.04 LTS | ([https://releases.ubuntu.com/20.04/](https://releases.ubuntu.com/20.04/)) | A consistent operating system environment for all platforms to minimize OS-level performance variations. |
| NVIDIA SDK | JetPack 5.1.1 | ([https://developer.nvidia.com/embedded/jetpack-sdk-511](https://developer.nvidia.com/embedded/jetpack-sdk-511)) | Includes Linux for Tegra (L4T) 35.3.1, CUDA 11.4, and TensorRT 8.5.2.7 |
| Qualcomm SDK | Neural Processing SDK v2.10+ | ([https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk-ai](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk-ai)) | Provides the necessary tools (snpe-onnx-to-dlc, snpe-dlc-quantize) and runtime libraries for the Hexagon NPU.8 |
| Intel SDK | OpenVINO Toolkit 2023.3+ | ([https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)) | Provides the Model Optimizer (mo) and runtime libraries, including the Neural Network Compression Framework (NNCF) for quantization.9 |
| AI Model | DDRNet-23-slim (ONNX) | ([https://huggingface.co/qualcomm/DDRNet23-Slim](https://huggingface.co/qualcomm/DDRNet23-Slim)) | The semantic segmentation model under test. The ONNX format serves as the common starting point for all platform-specific optimizations.33 |
| Dataset | Cityscapes (leftImg8bit, gtFine) | ([https://www.cityscapes-dataset.com/](https://www.cityscapes-dataset.com/)) | Provides the images and ground-truth labels for calibration and accuracy validation.35 |

### **Dataset Preparation**

1. **Download Cityscapes:** Register on the official Cityscapes website and download the leftImg8bit\_trainvaltest.zip (approx. 11 GB) and gtFine\_trainvaltest.zip (approx. 241 MB) packages.  
2. **Extract Data:** Unzip the packages into a common directory structure.  
3. **Create a Representative Subset:** A 500-image subset from the val split is required for both INT8 calibration and post-quantization accuracy validation. To ensure reproducibility, this subset must be identical for every run of the experiment. Execute a script to randomly select 500 images from the leftImg8bit/val directory and their corresponding ground-truth labels from gtFine/val, copying them to a dedicated calibration\_validation\_set directory. Using a fixed random seed in the selection script is crucial for consistency.

---

## **⚙️ Step-by-Step Experimental Protocol**

This section provides the detailed, command-level instructions for executing the benchmark. It is divided into three phases: preparing the optimized models, running the performance tests, and analyzing the collected data.

### **Phase I: Model Preparation and Optimization**

This phase converts the source ONNX model into a platform-specific, performance-optimized format using INT8 precision.

#### **General Workflow**

1. **Download Model:** Obtain the DDRNet23-Slim.onnx model from the Hugging Face repository specified in Table 2\.33  
2. **Prepare Calibration Set:** Ensure the 500-image calibration/validation subset from the Cityscapes dataset has been created as described in Section 2\.

#### **Platform: NVIDIA Jetson Orin NX (TensorRT)**

The primary tool for this platform is trtexec, a command-line utility included with TensorRT that serves as a powerful wrapper for building and benchmarking engines.36 The process involves two steps: generating a calibration cache and then using that cache to build two separate engines, one for the GPU and one for the DLA.

Step 1: Generate INT8 Calibration Cache  
Before building an INT8 engine, TensorRT must analyze the distribution of activation values to determine appropriate scaling factors. This is done by running inference on a representative dataset and storing the resulting tensor dynamic ranges in a cache file.  
Create a text file named calibration\_files.txt that lists the absolute paths to the 500 calibration images. Then, run trtexec in calibration mode:

Bash

/usr/src/tensorrt/bin/trtexec \--onnx=DDRNet23-Slim.onnx \\  
                            \--int8 \\  
                            \--calib=calibration.cache \\  
                            \--calibData=/path/to/calibration\_files.txt \\  
                            \--buildOnly

This command will not produce a final engine but will generate a calibration.cache file containing the necessary dynamic range information.

Step 2: Build the Ampere GPU Engine  
With the calibration cache ready, build the INT8 engine optimized for the Orin NX's Ampere architecture GPU.

Bash

/usr/src/tensorrt/bin/trtexec \--onnx=DDRNet23-Slim.onnx \\  
                            \--saveEngine=ddrnet\_gpu.engine \\  
                            \--int8 \\  
                            \--calib=calibration.cache \\  
                            \--workspace=4096

* \--onnx: Specifies the input ONNX model file.  
* \--saveEngine: Defines the path for the output serialized TensorRT engine.  
* \--int8: Enables INT8 precision mode for the builder.  
* \--calib: Provides the pre-generated calibration cache to guide the quantization process.36  
* \--workspace: Allocates a generous 4096 MB of GPU memory for the builder to explore various optimization tactics and algorithms.38

Step 3: Build the NVDLA v2.0 Engine  
Next, build a separate engine specifically targeting one of the two NVDLA (NVIDIA Deep Learning Accelerator) cores on the Orin NX SoC.

Bash

/usr/src/tensorrt/bin/trtexec \--onnx=DDRNet23-Slim.onnx \\  
                            \--saveEngine=ddrnet\_dla.engine \\  
                            \--int8 \\  
                            \--calib=calibration.cache \\  
                            \--useDLACore=0 \\  
                            \--allowGPUFallback \\  
                            \--workspace=4096

* \--useDLACore=0: This crucial flag instructs TensorRT to compile the network layers for the first available DLA core (indexed from 0).39  
* \--allowGPUFallback: This is a mandatory flag for robust DLA deployment. If the DLA hardware does not support a specific layer or operation in the model, this flag permits TensorRT to assign that layer to run on the GPU instead, preventing a build failure and enabling a hybrid execution plan.41

#### **Platform: Qualcomm QCS6490 (Qualcomm NP SDK)**

The Qualcomm Neural Processing (NP) SDK employs a two-stage toolchain. First, the ONNX model is converted to the proprietary Deep Learning Container (.dlc) format. Second, this .dlc file is quantized using a separate tool.8

Step 1: Convert ONNX to FP32 DLC  
The initial conversion translates the model graph into the DLC format while retaining its original FP32 precision.

Bash

snpe-onnx-to-dlc \--input\_network DDRNet23-Slim.onnx \\  
                 \--output\_path ddrnet\_fp32.dlc

This command parses the ONNX file and creates an unquantized ddrnet\_fp32.dlc file.43

Step 2: Prepare Calibration Data List  
The quantization tool requires a list of pre-processed input files in a raw binary format. A helper script must be written to iterate through the 500 calibration images, pre-process them according to the DDRNet model's requirements (e.g., resizing, normalization), and save each resulting tensor as a .raw file. Subsequently, create a text file named calibration\_list.txt, where each line contains the absolute path to one of these .raw files.  
Step 3: Quantize DLC for Hexagon NPU  
Using the FP32 DLC and the calibration list, perform post-training quantization to generate the final INT8 model optimized for the Hexagon NPU (often referred to as the DSP in the SDK documentation).

Bash

snpe-dlc-quantize \--input\_dlc ddrnet\_fp32.dlc \\  
                  \--input\_list calibration\_list.txt \\  
                  \--output\_dlc ddrnet\_quantized.dlc

This tool will run the model on the calibration data, compute quantization parameters (scales and zero-points), and embed them into the new ddrnet\_quantized.dlc file, making it ready for execution on the Hexagon processor.45

#### **Platform: Radxa X4 with Intel N100 (OpenVINO)**

The Intel OpenVINO workflow involves using the Model Optimizer (mo) tool to convert the ONNX model to OpenVINO's Intermediate Representation (IR) format, which consists of an .xml (topology) and a .bin (weights) file. Quantization is then applied programmatically using the Neural Network Compression Framework (NNCF) library.47

Step 1: Convert ONNX to FP16 IR  
While direct conversion to FP32 is possible, converting to FP16 is often a more robust intermediate step for targeting Intel GPUs, which have native hardware support for 16-bit floating-point arithmetic. This can improve performance even before INT8 quantization.

Bash

mo \--input\_model DDRNet23-Slim.onnx \\  
   \--output\_dir FP16\_IR \\  
   \--data\_type FP16

* \--input\_model: Specifies the source ONNX file.  
* \--output\_dir: Designates the directory for the output IR files.  
* \--data\_type FP16: This flag instructs the Model Optimizer to cast the model's weights and biases to the FP16 data type, creating an IR optimized for half-precision floating-point execution.50

Step 2: Quantize IR to INT8 using NNCF  
Post-training quantization is performed using a Python script that leverages the openvino and nncf libraries. This approach provides fine-grained control and uses the same calibration dataset as the other platforms for consistency.

Python

import openvino as ov  
import nncf  
import numpy as np  
from PIL import Image  
from pathlib import Path

\# \--- Configuration \---  
FP16\_MODEL\_XML \= "FP16\_IR/DDRNet23-Slim.xml"  
INT8\_MODEL\_XML \= "INT8\_IR/DDRNet23-Slim.xml"  
CALIBRATION\_DIR \= Path("/path/to/calibration\_validation\_set/images")  
INPUT\_HEIGHT, INPUT\_WIDTH \= 1024, 2048

\# \--- 1\. Load the FP16 model \---  
core \= ov.Core()  
model \= core.read\_model(FP16\_MODEL\_XML)

\# \--- 2\. Create the calibration dataset \---  
\# Transformation function to preprocess each image  
def transform\_fn(image\_path):  
    image \= Image.open(image\_path).resize((INPUT\_WIDTH, INPUT\_HEIGHT))  
    image \= np.array(image).astype(np.float32)  
    image \= image / 255.0  \# Normalize to   
    \# Add normalization specific to DDRNet if required  
    image \= np.transpose(image, (2, 0, 1))  \# HWC to CHW  
    image \= np.expand\_dims(image, axis=0)   \# Add batch dimension  
    \# The key of the dictionary must match the model's input layer name  
    return {"input.1": image}

\# Get list of calibration image paths  
calibration\_data \= list(CALIBRATION\_DIR.glob("\*.png"))

\# Create NNCF Dataset object  
calibration\_dataset \= nncf.Dataset(calibration\_data, transform\_fn)

\# \--- 3\. Quantize the model \---  
\# The nncf.quantize() function applies the default PTQ algorithm  
quantized\_model \= nncf.quantize(model, calibration\_dataset)

\# \--- 4\. Save the INT8 model \---  
Path("INT8\_IR").mkdir(exist\_ok=True)  
ov.save\_model(quantized\_model, INT8\_MODEL\_XML)

print(f"INT8 model saved to {INT8\_MODEL\_XML}")

This script programmatically loads the FP16 IR, defines a data loading and preprocessing pipeline for the 500 calibration images, applies the NNCF post-training quantization algorithm, and saves the resulting INT8 IR model to a new directory.49

### **Phase II: Benchmarking Execution**

This phase involves running inference on the optimized models using a standardized script structure to measure performance metrics.

#### **Conceptual Script Structure**

A consistent structure should be used for all platforms to ensure comparable measurements. The core logic, implemented in Python, is as follows:

Python

import time  
import numpy as np

def benchmark(load\_model\_func, model\_path, target\_device, num\_iterations=1000, warmup\_iterations=100):  
    """  
    A generic benchmarking function.  
      
    Args:  
        load\_model\_func: A platform-specific function to load and compile the model.  
        model\_path: Path to the optimized model file.  
        target\_device: String identifier for the target hardware.  
        num\_iterations: Number of timed inference runs.  
        warmup\_iterations: Number of untimed runs to stabilize the system.  
    """  
      
    \# 1\. Load and compile the model for the specified target device  
    print(f"Loading model {model\_path} for {target\_device}...")  
    inference\_engine \= load\_model\_func(model\_path, target\_device)  
      
    \# 2\. Prepare a sample input tensor (e.g., a random tensor with the correct shape)  
    input\_shape \= inference\_engine.get\_input\_shape()  
    sample\_input \= np.random.rand(\*input\_shape).astype(np.float32)  
      
    \# 3\. Warm-up phase (not timed)  
    print(f"Running {warmup\_iterations} warm-up iterations...")  
    for \_ in range(warmup\_iterations):  
        inference\_engine.infer(sample\_input)  
      
    \# 4\. Timed inference loop  
    print(f"Running {num\_iterations} timed iterations...")  
    latencies \=  
    for \_ in range(num\_iterations):  
        start\_time \= time.perf\_counter()  
        inference\_engine.infer(sample\_input)  
        end\_time \= time.perf\_counter()  
        latencies.append((end\_time \- start\_time) \* 1000) \# Convert to milliseconds

    \# 5\. Log latencies to a file for later analysis  
    results\_filename \= f"results\_{model\_path.stem}\_{target\_device}.csv"  
    np.savetxt(results\_filename, np.array(latencies), delimiter=",")  
    print(f"Results saved to {results\_filename}")  
      
    return latencies

This structure standardizes the crucial elements: loading the optimized model, a warm-up phase to ensure system clocks are at maximum frequency, a long timed loop for stable measurements, use of a high-resolution timer (time.perf\_counter), and logging of raw latency data for post-processing.

#### **Platform-Specific Runtime Implementation**

NVIDIA (TensorRT):  
The load\_model\_func will use the tensorrt Python library. Two separate benchmark runs are required.

* **For GPU:** The script will load ddrnet\_gpu.engine. The engine file itself contains the targeting information, so no explicit device selection is needed at runtime.  
* **For DLA:** The script will load ddrnet\_dla.engine. The TensorRT runtime will automatically dispatch the DLA-compatible portions of the graph to the NVDLA hardware as defined during the engine build process.54

Python

\# Example TensorRT loader function  
import tensorrt as trt  
\#... (common imports)

def load\_tensorrt\_model(engine\_path, \_): \# target\_device is implicit  
    TRT\_LOGGER \= trt.Logger(trt.Logger.WARNING)  
    with open(engine\_path, "rb") as f, trt.Runtime(TRT\_LOGGER) as runtime:  
        engine \= runtime.deserialize\_cuda\_engine(f.read())  
    context \= engine.create\_execution\_context()  
    \#... create an inference wrapper class around this context...  
    return InferenceWrapper(context, engine)

Qualcomm (SNPE):  
The load\_model\_func will use the snpe-python bindings. The key step is specifying the correct runtime target when building the network instance to ensure the workload is directed to the Hexagon NPU.

Python

\# Example SNPE loader function  
from snpe import zdl  
\#... (common imports)

def load\_snpe\_model(dlc\_path, target\_runtime):  
    if target\_runtime \== "DSP":  
        runtime \= zdl.DlSystem.Runtime\_t.DSP  
    \#... add cases for GPU, CPU etc.  
    else:  
        raise ValueError("Invalid SNPE runtime")  
          
    builder \= zdl.Snpe.SnpeBuilder(dlc\_path).setRuntime(runtime)  
    snpe\_engine \= builder.build()  
    \#... create an inference wrapper class around this snpe\_engine...  
    return InferenceWrapper(snpe\_engine)

The benchmark script would call this function with target\_device="DSP".56

Intel (OpenVINO):  
The load\_model\_func will use the openvino.Core object. The target device is specified explicitly during the model compilation step.

Python

\# Example OpenVINO loader function  
import openvino as ov  
\#... (common imports)

def load\_openvino\_model(xml\_path, target\_device):  
    core \= ov.Core()  
    model \= core.read\_model(xml\_path)  
    \# The 'GPU' device name targets the integrated Intel UHD Graphics  
    compiled\_model \= core.compile\_model(model, device\_name=target\_device)  
    \#... create an inference wrapper class around this compiled\_model...  
    return InferenceWrapper(compiled\_model)

The benchmark script would call this function with target\_device="GPU" to target the integrated graphics.58

### **Phase III: Data Analysis**

After running the benchmark scripts for all four targets, the raw data (latency CSVs and power logs) must be processed to derive the final metrics.

#### **Parsing Performance Logs**

A Python script using libraries like pandas and numpy should be used to load each results\_\*.csv file into a data structure for analysis.

#### **Calculating Performance Metrics**

1. Throughput (FPS): This metric represents the average number of inferences the system can perform per second under a sustained load. It is calculated from the mean latency.

   Throughput (FPS)=mean(latenciesms​)1000​  
2. **99th Percentile Latency (ms):** This is a critical metric for real-time applications, as it represents a practical worst-case latency, ignoring extreme outliers. It indicates that 99% of all inferences completed faster than this value.  
   Python  
   import numpy as np  
   \# latencies\_ms is a NumPy array of all recorded latencies  
   p99\_latency \= np.percentile(latencies\_ms, 99)

   The numpy.percentile function is the standard tool for this calculation.60

#### **Correlating with Power Data**

The power data, logged externally by the Yokogawa WT300E, must be correlated with the performance logs.

1. **Synchronization:** The benchmark execution script should log a start timestamp just before the timed loop begins and an end timestamp immediately after it finishes.  
2. **Data Extraction:** The power analyzer's data log (typically a CSV file with timestamp and power readings in Watts) is imported.  
3. **Windowing:** The power readings are filtered to include only those that fall between the start and end timestamps of the timed inference loop.  
4. **Calculation:** The **Average Power (W)** is calculated by taking the mean of all power readings within this synchronized window. To determine the power consumption attributable to the workload itself, a separate measurement of the system's idle power should be taken and subtracted from the average power during the run.

#### **Calculating Efficiency**

The final metric, Efficiency (FPS/Watt), provides the most holistic view of performance for power-constrained embedded systems. It quantifies how many frames per second of processing can be achieved for each watt of power consumed.

Efficiency (FPS/W)=Average Power (W)Throughput (FPS)​

---

## **Final Table and Best Practices**

The culmination of this protocol is the population of a standardized results table and adherence to a checklist that ensures the integrity and reliability of the findings.

### **Table 3: Final Benchmark Results (Template)**

This table serves as the final output of the experiment, providing a clear, at-a-glance comparison of the performance and efficiency of the DDRNet-23-slim model across the four target accelerator platforms. The inclusion of the post-quantization accuracy metric (mIoU) is essential for validating that performance gains were not achieved at the cost of unacceptable degradation in model quality.

| Platform | Accelerator | Throughput (FPS) | 99th Percentile Latency (ms) | Average Power (W) | Efficiency (FPS/W) | Post-Quantization mIoU (%) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| NVIDIA Jetson Orin NX 8GB | Ampere GPU |  |  |  |  |  |
| NVIDIA Jetson Orin NX 8GB | NVDLA v2.0 |  |  |  |  |  |
| Qualcomm QCS6490 | Hexagon NPU |  |  |  |  |  |
| Radxa X4 (Intel N100) | Intel UHD Graphics |  |  |  |  |  |

### **Checklist for Reliability**

To ensure the validity and reproducibility of the benchmark results, the following conditions must be met and verified for every experimental run.

* **✅ Verify Post-Quantization Accuracy:** Before commencing performance benchmarking, the quantized model for each platform must be validated for accuracy. Run inference on the entire 500-image validation subset and calculate the mean Intersection over Union (mIoU) score. This score should be compared against the mIoU of the original FP32 model. A drop in mIoU greater than 2-3 percentage points may indicate a suboptimal calibration or a fundamental limitation of the platform's quantization toolchain, and this degradation must be noted alongside the performance results.12  
* **✅ Ensure System Quiescence:** Prior to initiating each benchmark script, the target device must be in a quiescent (idle) state. Close all non-essential applications and services. Use system monitoring tools (e.g., htop, tegrastats) to confirm that CPU, GPU, and accelerator utilization has returned to a baseline low before starting the measurement. This prevents background processes from consuming resources and interfering with the results.  
* **✅ Mandate Active Cooling and Monitor Thermals:** All target platforms must be equipped with an active cooling solution (e.g., a fan and heatsink). During the sustained benchmark run, system temperatures must be actively monitored. If the SoC temperature exceeds its specified thermal limits (a conservative threshold is 85°C), the run should be considered invalid. Exceeding this limit indicates that thermal throttling has likely occurred, artificially depressing the performance metrics and not reflecting the hardware's true sustained capability.  
* **✅ Use a Warm-up Phase:** Every benchmark script must include an initial, untimed loop of at least 100 inference iterations. This "warm-up" period is critical for allowing the device's dynamic voltage and frequency scaling (DVFS) to ramp up processor clocks to their maximum sustained levels and for populating various system caches. The subsequent timed measurements will therefore reflect steady-state performance, not the variable performance of a "cold" system.  
* **✅ Isolate DC Power Measurement:** The external power analyzer must be configured to measure the DC voltage and current being supplied directly *to the development board's power input jack*. Measuring the AC power from the wall outlet would incorrectly include the inefficiency of the AC-to-DC power adapter, confounding the measurement of the device's actual power consumption. A DC power breakout board is often required to facilitate this connection.

#### **Works cited**

1. Benchmarks That Don't Lie: A 90-Day Blueprint for Responsible AI \- ibex., accessed September 13, 2025, [https://www.ibex.co/resources/blogs/benchmarks-that-dont-lie-a-90-day-blueprint-for-responsible-ai/](https://www.ibex.co/resources/blogs/benchmarks-that-dont-lie-a-90-day-blueprint-for-responsible-ai/)  
2. AI Benchmarking Best Practices: A Framework for CX Leaders \- Quiq, accessed September 13, 2025, [https://quiq.com/blog/ai-benchmarking-best-practices/](https://quiq.com/blog/ai-benchmarking-best-practices/)  
3. Benchmarking AI \- ML Systems Textbook, accessed September 13, 2025, [https://www.mlsysbook.ai/contents/core/benchmarking/benchmarking](https://www.mlsysbook.ai/contents/core/benchmarking/benchmarking)  
4. AI SDK for Cloud Computing | Cloud Inference \- Qualcomm, accessed September 13, 2025, [https://www.qualcomm.com/developer/cloud-ai-sdk/overview](https://www.qualcomm.com/developer/cloud-ai-sdk/overview)  
5. AI ML SDK: A Comprehensive Guide \- BytePlus, accessed September 13, 2025, [https://www.byteplus.com/en/topic/536972](https://www.byteplus.com/en/topic/536972)  
6. Cloud AI SDK \- Qualcomm, accessed September 13, 2025, [https://www.qualcomm.com/developer/software/cloud-ai-sdk](https://www.qualcomm.com/developer/software/cloud-ai-sdk)  
7. JetPack SDK 5.1.1 | NVIDIA Developer, accessed September 13, 2025, [https://developer.nvidia.com/embedded/jetpack-sdk-511](https://developer.nvidia.com/embedded/jetpack-sdk-511)  
8. Qualcomm Neural Processing SDK for AI Documentation, accessed September 13, 2025, [https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2?product=1601111740010412](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2?product=1601111740010412)  
9. OpenVINO™ is an open source toolkit for optimizing and deploying AI inference \- GitHub, accessed September 13, 2025, [https://github.com/openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino)  
10. Power Estimation and Energy Efficiency of AI Accelerators on Embedded Systems \- MDPI, accessed September 13, 2025, [https://www.mdpi.com/1996-1073/18/14/3840](https://www.mdpi.com/1996-1073/18/14/3840)  
11. Data-driven Software-based Power Estimation for Embedded Devices \- arXiv, accessed September 13, 2025, [https://arxiv.org/html/2407.02764v2](https://arxiv.org/html/2407.02764v2)  
12. What is the impact of INT8 and INT4 precision on AI model accuracy? \- Massed Compute, accessed September 13, 2025, [https://massedcompute.com/faq-answers/?question=What%20is%20the%20impact%20of%20INT8%20and%20INT4%20precision%20on%20AI%20model%20accuracy?](https://massedcompute.com/faq-answers/?question=What+is+the+impact+of+INT8+and+INT4+precision+on+AI+model+accuracy?)  
13. What Is int8 Quantization and Why Is It Popular for Deep Neural Networks? \- MathWorks, accessed September 13, 2025, [https://fr.mathworks.com/company/technical-articles/what-is-int8-quantization-and-why-is-it-popular-for-deep-neural-networks.html](https://fr.mathworks.com/company/technical-articles/what-is-int8-quantization-and-why-is-it-popular-for-deep-neural-networks.html)  
14. Understanding LLM Quantization: A Deep Dive into Performance Optimization \- Sam Ozturk, accessed September 13, 2025, [https://themeansquare.medium.com/understanding-llm-quantization-a-deep-dive-into-performance-optimization-c27d63857faa](https://themeansquare.medium.com/understanding-llm-quantization-a-deep-dive-into-performance-optimization-c27d63857faa)  
15. NVIDIA® Jetson Orin™ NX 8GB-EDOM Technology \- Your Best Solutions Partner, accessed September 13, 2025, [https://www.edomtech.com/en/product-detail/jetson-orin-nx-8gb/](https://www.edomtech.com/en/product-detail/jetson-orin-nx-8gb/)  
16. NVIDIA Jetson Orin NX 8 GB Specs \- GPU Database \- TechPowerUp, accessed September 13, 2025, [https://www.techpowerup.com/gpu-specs/jetson-orin-nx-8-gb.c4081](https://www.techpowerup.com/gpu-specs/jetson-orin-nx-8-gb.c4081)  
17. NVIDIA® Jetson Orin™ NX Edge AI Computing \- Specifications \- NEXCOM, accessed September 13, 2025, [https://www.nexcom.com/Products/multi-media-solutions/ai-edge-computer/nvidia-solutions/aiedge-x-80/Specifications](https://www.nexcom.com/Products/multi-media-solutions/ai-edge-computer/nvidia-solutions/aiedge-x-80/Specifications)  
18. Hardware \- Qualcomm, accessed September 13, 2025, [https://www.qualcomm.com/developer/hardware/rb3-gen-2-development-kit/hardware](https://www.qualcomm.com/developer/hardware/rb3-gen-2-development-kit/hardware)  
19. Hardware \- Qualcomm, accessed September 13, 2025, [https://www.qualcomm.com/products/internet-of-things/consumer/smart-homes/hardware](https://www.qualcomm.com/products/internet-of-things/consumer/smart-homes/hardware)  
20. C6490 Development Kit \- Thundercomm, accessed September 13, 2025, [https://www.thundercomm.com/product/c6490-development-kit/](https://www.thundercomm.com/product/c6490-development-kit/)  
21. Radxa X4, accessed September 13, 2025, [https://docs.radxa.com/en/x/x4](https://docs.radxa.com/en/x/x4)  
22. Intel N100 and RP2040 Powered Tiny SBC: Meet the Compact Powerhouse, Radxa X4, accessed September 13, 2025, [https://www.youtube.com/watch?v=1SunQG4yMIA](https://www.youtube.com/watch?v=1SunQG4yMIA)  
23. Radxa X4, accessed September 13, 2025, [https://www.radxa.com/products/x/x4/](https://www.radxa.com/products/x/x4/)  
24. WT300E Digital Power Analyzer \- Yokogawa Test & Measurement, accessed September 13, 2025, [https://tmi.yokogawa.com/us/solutions/products/power-analyzers/digital-power-meter-wt300e/](https://tmi.yokogawa.com/us/solutions/products/power-analyzers/digital-power-meter-wt300e/)  
25. Yokogawa WT300E \- Digital Power Analyzer \- Instru-Measure, accessed September 13, 2025, [https://instru-measure.com/yokogawa-wt300e-digital-power-analyzer/](https://instru-measure.com/yokogawa-wt300e-digital-power-analyzer/)  
26. Yokogawa WT300E Digital Power Analyzer Yokogawa Power Analyzer & Power Meter Test & Measurement Malaysia, Penang, Singapore, Indonesia Supplier, Suppliers, Supply, Supplies | Hexo Industries (M) Sdn Bhd, accessed September 13, 2025, [https://m.hexoind.com/index.php?ws=showproducts\&products\_id=4407695](https://m.hexoind.com/index.php?ws=showproducts&products_id=4407695)  
27. Cooling Solutions \- Silicon Highway, accessed September 13, 2025, [https://www.siliconhighwaydirect.com/category-s/1820.htm](https://www.siliconhighwaydirect.com/category-s/1820.htm)  
28. Vision AI-KIT 6490 \- Tria Technologies, accessed September 13, 2025, [https://www.tria-technologies.com/product/vision-ai-kit-6490/](https://www.tria-technologies.com/product/vision-ai-kit-6490/)  
29. Radxa Heatsink for X4, accessed September 13, 2025, [https://www.radxa.com/products/accessories/heatsink-for-x4/](https://www.radxa.com/products/accessories/heatsink-for-x4/)  
30. Qualcomm Neural Processing SDK \- Qualcomm® Linux ..., accessed September 13, 2025, [https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/building\_and\_executing\_tutorial.html?vproduct=1601111740013072\&version=1.5\&facet=Qualcomm%20Neural%20Processing%20SDK](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/building_and_executing_tutorial.html?vproduct=1601111740013072&version=1.5&facet=Qualcomm+Neural+Processing+SDK)  
31. Neural Processing SDK \- Qualcomm ID, accessed September 13, 2025, [https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/setup\_linux.html](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/setup_linux.html)  
32. Installation Guides for the OpenVINO™ Toolkit \- Intel, accessed September 13, 2025, [https://www.intel.com/content/www/us/en/support/articles/000057226/software/development-software.html](https://www.intel.com/content/www/us/en/support/articles/000057226/software/development-software.html)  
33. qualcomm/DDRNet23-Slim at cf7fde462212e9a414f63b809157f5f862a2f065 \- Hugging Face, accessed September 13, 2025, [https://huggingface.co/qualcomm/DDRNet23-Slim/blob/cf7fde462212e9a414f63b809157f5f862a2f065/DDRNet23-Slim.onnx](https://huggingface.co/qualcomm/DDRNet23-Slim/blob/cf7fde462212e9a414f63b809157f5f862a2f065/DDRNet23-Slim.onnx)  
34. qualcomm/DDRNet23-Slim \- Hugging Face, accessed September 13, 2025, [https://huggingface.co/qualcomm/DDRNet23-Slim](https://huggingface.co/qualcomm/DDRNet23-Slim)  
35. Chris1/cityscapes · Datasets at Hugging Face, accessed September 13, 2025, [https://huggingface.co/datasets/Chris1/cityscapes](https://huggingface.co/datasets/Chris1/cityscapes)  
36. Command-Line Programs — NVIDIA TensorRT Documentation, accessed September 13, 2025, [https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html)  
37. TensorRT's Capabilities \- NVIDIA Documentation, accessed September 13, 2025, [https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/capabilities.html](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/capabilities.html)  
38. trtexec \--help \- GitHub Gist, accessed September 13, 2025, [https://gist.github.com/apivovarov/efe528348588408615496e2c5b4280ae](https://gist.github.com/apivovarov/efe528348588408615496e2c5b4280ae)  
39. Working with DLA — NVIDIA TensorRT Documentation, accessed September 13, 2025, [https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html)  
40. TensorRT Command-Line Wrapper: trtexec \- ccoderun.ca, accessed September 13, 2025, [https://www.ccoderun.ca/programming/doxygen/tensorrt/md\_TensorRT\_samples\_opensource\_trtexec\_README.html](https://www.ccoderun.ca/programming/doxygen/tensorrt/md_TensorRT_samples_opensource_trtexec_README.html)  
41. TensorRT trtexec的用法说明 \- 博客园, accessed September 13, 2025, [https://www.cnblogs.com/michaelcjl/p/16643306.html](https://www.cnblogs.com/michaelcjl/p/16643306.html)  
42. Neural Processing SDK, accessed September 13, 2025, [https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2)  
43. aleshem/yolov5\_snpe\_conversion: YOLOv5 in PyTorch \> ONNX \> DLC \- GitHub, accessed September 13, 2025, [https://github.com/aleshem/yolov5\_snpe\_conversion](https://github.com/aleshem/yolov5_snpe_conversion)  
44. AI Developer Workflow \- Qualcomm® Linux Documentation, accessed September 13, 2025, [https://docs.qualcomm.com/bundle/publicresource/topics/80-70020-15B/appx-export-yolov8.html?vproduct=1601111740013072\&version=1.5](https://docs.qualcomm.com/bundle/publicresource/topics/80-70020-15B/appx-export-yolov8.html?vproduct=1601111740013072&version=1.5)  
45. SNPE Building and Executing Your Model for Windows Host \- Neural Processing SDK, accessed September 13, 2025, [https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/building\_and\_executing\_tutorial\_windows\_host.html](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/building_and_executing_tutorial_windows_host.html)  
46. rakelkar/snpe\_convert: Convert and run a TF model using Qualcomm SNPE tools \- GitHub, accessed September 13, 2025, [https://github.com/rakelkar/snpe\_convert](https://github.com/rakelkar/snpe_convert)  
47. Intel OpenVINO Model Optimizer \- C\# Corner, accessed September 13, 2025, [https://www.c-sharpcorner.com/article/intel-openvino-model-optimizer/](https://www.c-sharpcorner.com/article/intel-openvino-model-optimizer/)  
48. AI Model Optimization using OpenVINO \- Ignitarium, accessed September 13, 2025, [https://ignitarium.com/ai-model-optimisation-using-openvino/](https://ignitarium.com/ai-model-optimisation-using-openvino/)  
49. Quantizing Models Post-training \- OpenVINO™ documentation, accessed September 13, 2025, [https://docs.openvino.ai/2023.3/ptq\_introduction.html](https://docs.openvino.ai/2023.3/ptq_introduction.html)  
50. Running OpenVINO Models on Intel Integrated GPU \- LearnOpenCV, accessed September 13, 2025, [https://learnopencv.com/running-openvino-models-on-intel-integrated-gpu/](https://learnopencv.com/running-openvino-models-on-intel-integrated-gpu/)  
51. \[LEGACY\] Compressing a Model to FP16 \- OpenVINO™ documentation, accessed September 13, 2025, [https://docs.openvino.ai/2023.3/openvino\_docs\_MO\_DG\_FP16\_Compression.html](https://docs.openvino.ai/2023.3/openvino_docs_MO_DG_FP16_Compression.html)  
52. How to read onnx model on GPU and set its precision as fp16? · Issue \#12448 · openvinotoolkit/openvino \- GitHub, accessed September 13, 2025, [https://github.com/openvinotoolkit/openvino/issues/12448](https://github.com/openvinotoolkit/openvino/issues/12448)  
53. Accelerate Big Transfer (BiT) Model Even More with Quantization using OpenVINO and Neural Network Compression Framework (NNCF) \- Medium, accessed September 13, 2025, [https://medium.com/openvino-toolkit/accelerate-big-transfer-bit-model-even-more-with-quantization-using-openvino-and-neural-network-6d653fee05bd](https://medium.com/openvino-toolkit/accelerate-big-transfer-bit-model-even-more-with-quantization-using-openvino-and-neural-network-6d653fee05bd)  
54. Quick Start Guide — NVIDIA TensorRT Documentation, accessed September 13, 2025, [https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html)  
55. TensorRT Python Inference \- Lei Mao's Log Book, accessed September 13, 2025, [https://leimao.github.io/blog/TensorRT-Python-Inference/](https://leimao.github.io/blog/TensorRT-Python-Inference/)  
56. Port a model using Qualcomm Neural Processing Engine SDK, accessed September 13, 2025, [https://docs.qualcomm.com/bundle/publicresource/topics/80-70017-15B/port-models.html](https://docs.qualcomm.com/bundle/publicresource/topics/80-70017-15B/port-models.html)  
57. Run models \- AI Developer Workflow, accessed September 13, 2025, [https://docs.qualcomm.com/bundle/publicresource/topics/80-70018-15B/run-models.html](https://docs.qualcomm.com/bundle/publicresource/topics/80-70018-15B/run-models.html)  
58. Working with GPUs in OpenVINO, accessed September 13, 2025, [https://docs.openvino.ai/2024/notebooks/gpu-device-with-output.html](https://docs.openvino.ai/2024/notebooks/gpu-device-with-output.html)  
59. OpenVINO \- ALCF User Guides \- Argonne National Laboratory, accessed September 13, 2025, [https://docs.alcf.anl.gov/aurora/data-science/inference/openvino/](https://docs.alcf.anl.gov/aurora/data-science/inference/openvino/)  
60. How to calculate percentiles in NumPy \- Educative.io, accessed September 13, 2025, [https://www.educative.io/answers/how-to-calculate-percentiles-in-numpy](https://www.educative.io/answers/how-to-calculate-percentiles-in-numpy)  
61. numpy.percentile — NumPy v2.0 Manual, accessed September 13, 2025, [https://numpy.org/doc/2.0/reference/generated/numpy.percentile.html](https://numpy.org/doc/2.0/reference/generated/numpy.percentile.html)  
62. numpy.percentile — NumPy v2.1 Manual, accessed September 13, 2025, [https://numpy.org/doc/2.1/reference/generated/numpy.percentile.html](https://numpy.org/doc/2.1/reference/generated/numpy.percentile.html)