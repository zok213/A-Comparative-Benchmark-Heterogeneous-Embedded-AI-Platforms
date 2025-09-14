

# **A Definitive Guide to Reproducing the Pseudo-LiDAR \+ PointPillars 3D Object Detection Benchmark on Embedded Systems**

## **1\. Benchmark Overview**

### **1.1. Objective and Significance**

This document provides a comprehensive, publication-quality guide for the rigorous benchmarking of a vision-based, two-stage 3D object detection pipeline on a diverse set of embedded computing platforms. The primary objective is to meticulously measure and compare the end-to-end performance, power efficiency, and final detection accuracy of this pipeline. The key metrics for evaluation are total pipeline latency (p99), throughput in frames per second (FPS), average power consumption (W), power efficiency (FPS/W), and 3D mean Average Precision (mAP) on the KITTI dataset.

The significance of this benchmark is rooted in the increasing demand for robust 3D perception in autonomous systems, such as robotics and automated driving. While LiDAR sensors provide direct 3D measurements, they are often costly, have active components, and can be bulky. This benchmark evaluates the practical viability of a low-cost, passive alternative: using stereo cameras to generate a "pseudo-LiDAR" point cloud, which is then processed by a highly efficient, LiDAR-based detection algorithm.1 By executing this identical pipeline on three distinct classes of embedded hardware—NVIDIA Jetson (GPU-centric), Qualcomm (NPU-centric), and Intel (iGPU-centric)—this guide facilitates a holistic and scientifically valid comparison. The resulting data provides critical insights into the trade-offs between computational performance, power consumption, and the ultimate application-level accuracy, informing hardware selection for real-world, resource-constrained deployments.

### **1.2. Pipeline Architecture: From Stereo Images to 3D Detections**

The end-to-end pipeline is architecturally deconstructed into two primary neural network inference stages, connected by an intermediate geometric processing step. This design allows for the modular evaluation of each component's contribution to the overall system performance.

Stage 1: Stereo Depth Estimation  
The pipeline ingests a synchronized pair of stereo images (left and right) from the KITTI dataset. These images are fed into the CREStereo network, a state-of-the-art model for stereo matching.2 CREStereo is selected for its high accuracy, which it achieves through a sophisticated cascaded recurrent architecture. This architecture employs a hierarchical, coarse-to-fine strategy, iteratively refining its estimate of the disparity between the two images.2 The final output of this stage is a dense disparity map, where the value of each pixel represents the horizontal displacement of that point between the left and right images. According to the principles of stereo vision, disparity is inversely proportional to depth, following the relation  
D=f⋅B/d, where D is depth, f is focal length, B is the stereo baseline, and d is disparity.4

Intermediate Step: Pseudo-LiDAR Point Cloud Generation  
This critical processing step is not a neural network inference but a purely algorithmic geometric transformation. The dense disparity map produced by CREStereo is converted into a three-dimensional point cloud. This process effectively simulates the data structure of a LiDAR sensor, which is essential for compatibility with the subsequent detection network.1 The transformation relies on projecting each 2D pixel  
(u, v) with its corresponding disparity value into 3D space. This reprojection is accomplished using the camera's intrinsic parameters (focal length, principal point) and extrinsic parameters (stereo baseline), which are provided in the KITTI dataset's calibration files.1 This step is executed on the host CPU of the embedded platform.

Stage 2: 3D Object Detection  
The generated pseudo-LiDAR point cloud serves as the input to the PointPillars network. This model is specifically chosen for its exceptional computational efficiency, making it highly suitable for embedded systems.7 PointPillars achieves its speed by avoiding computationally expensive 3D convolutions. Instead, it voxelizes the 3D point cloud into a grid of vertical columns, or "pillars." A simplified PointNet-like feature encoder is applied to the points within each pillar to learn a representative feature vector. These pillar features are then scattered back onto a 2D grid, creating a compact "pseudo-image" representation. This pseudo-image is then processed by a standard, highly optimized 2D CNN backbone and detection head to predict 3D bounding boxes for objects such as cars, pedestrians, and cyclists.7

## **2\. Prerequisites and Environment Setup**

### **2.1. Hardware Requirements**

Reproducibility of this benchmark necessitates a standardized hardware environment. The following components are mandatory.

* **Compute Platforms:** The benchmark must be executed on the following three systems:  
  * **NVIDIA Jetson Orin NX:** The 16GB memory variant is recommended to avoid potential memory bottlenecks. This platform features an Ampere architecture GPU with 1024 CUDA cores and 32 Tensor Cores.9  
  * **Qualcomm QCS6490 Development Kit:** A platform based on the QCS6490 SoC, which includes a Kryo 670 CPU, Adreno 643 GPU, and a Hexagon processor with vector extensions and a tensor accelerator, rated for up to 12 TOPS.11  
  * **Radxa X4:** A single-board computer featuring the Intel Processor N100 (Alder Lake-N architecture) with integrated Intel UHD Graphics. The 16GB LPDDR5 RAM variant is recommended.13  
* **Power Measurement:** An external, high-precision power analyzer capable of data logging is **mandatory**. On-board or on-chip power monitoring solutions are insufficient as they often fail to capture the total system-level power draw, which includes the SoC, memory, and other peripherals. The **Yokogawa WT300E series** is the reference instrument for this guide, providing accurate measurements and logging capabilities via Ethernet.15  
* **Thermal Management:** **Active cooling** (e.g., a fan affixed to a heatsink) is a non-negotiable requirement for all three platforms. The sustained, high-intensity computational load of the end-to-end pipeline will induce thermal throttling on passively cooled systems. This throttling leads to inconsistent and artificially degraded performance measurements, rendering the benchmark results invalid.

### **2.2. Software Stack**

A consistent software baseline is critical for isolating hardware performance differences.

* **Base Operating System:** All platforms must be provisioned with **Ubuntu 20.04 (Focal Fossa)** LTS. This includes AArch64 builds for the Jetson and Qualcomm platforms, and an x86-64 build for the Radxa X4.  
* **Platform-Specific SDKs:** The correct installation and configuration of the following vendor-specific SDKs are paramount.  
  * **NVIDIA:** Install **NVIDIA JetPack 5.1.1**. This comprehensive package provides the entire required software stack, including Linux for Tegra (L4T) 35.3.1, CUDA 11.4, cuDNN 8.6.0, and, most importantly, **TensorRT 8.5.2**, which is the inference optimization and runtime engine used for this platform.16  
  * **Qualcomm:** Install the **Qualcomm Neural Processing SDK for AI (SNPE)**, version 2.x or later. The setup process requires sourcing the envsetup.sh script within the SDK directory in a terminal session to configure the necessary environment variables ($SNPE\_ROOT, $PATH, etc.) for accessing the conversion and execution tools.18  
  * **Intel:** Install the **Intel Distribution of OpenVINO Toolkit** (2023.x or a more recent version). The recommended installation method is via PyPI: pip install openvino openvino-dev. For GPU offloading to function correctly, the Intel graphics drivers for Linux must be properly installed and configured as per the OpenVINO documentation.20

### **2.3. Models and Dataset Acquisition**

* **Pre-trained Models:** Download pre-trained versions of the CREStereo and PointPillars models. For maximum cross-platform compatibility, these models should be obtained in the ONNX (Open Neural Network Exchange) format, which will serve as the common starting point for optimization on each platform.  
* **KITTI Dataset:** Download the **KITTI Stereo 2015** 23 and  
  **KITTI 3D Object Detection** 6 datasets. The data must be organized into the following directory structure to ensure compatibility with the provided data loaders and evaluation scripts. This structure separates training and testing data and organizes images, calibration files, and ground-truth labels into their respective subdirectories.1  
  KITTI/object/  
  ├── training/  
  │   ├── calib/      \# Camera calibration files (\*.txt)  
  │   ├── image\_2/    \# Left color images (\*.png)  
  │   ├── image\_3/    \# Right color images (\*.png)  
  │   └── label\_2/    \# Ground truth 3D object labels (\*.txt)  
  └── testing/  
      ├── calib/  
      ├── image\_2/  
      └── image\_3/

## **3\. Phase I: Platform-Specific Model Optimization (CRITICAL)**

This phase is the most technically demanding and crucial for a valid benchmark. The process of optimizing a neural network for deployment is highly specific to the target hardware and its corresponding software toolchain. A simple "one-size-fits-all" model is not feasible. Furthermore, achieving INT8 (8-bit integer) precision is not merely an optional optimization but a functional requirement. The specialized hardware accelerators on these platforms—NVIDIA's Tensor Cores 10, Qualcomm's Hexagon Tensor Accelerator 11, and Intel's GPU execution units—are designed to deliver maximum performance and efficiency when operating on low-precision data types. An FP32-based comparison would fail to engage these critical silicon features, resulting in a fundamentally flawed and unrepresentative benchmark of the hardware's intended capabilities. Therefore, the post-training quantization and calibration process, which is unique to each toolchain, must be executed with precision.

### **3.1. NVIDIA Jetson Orin NX: TensorRT Engine Generation**

The goal for the Jetson platform is to convert the ONNX models into optimized TensorRT engine files (.engine).

* **Step 1: Export to ONNX:** If the source models are in PyTorch, convert them to the .onnx format using the torch.onnx.export function. It is critical to specify a fixed opset version (e.g., 13\) that is well-supported by the target TensorRT version (8.5.2) to ensure maximum operator compatibility.26  
* **Step 2: Generate INT8 Calibration Cache:** TensorRT's INT8 quantization requires a calibration step to determine the dynamic range of activations for each layer. This process requires a representative set of input data. The trtexec tool consumes a pre-generated calibration cache file. A Python script utilizing the TensorRT API (tensorrt.IInt8EntropyCalibrator2) must be used to create this cache. This script will iterate over a subset of the KITTI training dataset (approximately 300-500 images/point clouds), feeding them through the network to collect activation statistics and generate a calibration.cache file.29  
* **Step 3: Build the INT8 Engine with trtexec:** With the ONNX model and calibration cache prepared, use the trtexec command-line tool to build the final, optimized, and deployable TensorRT engine. The arguments must be specified precisely.  
  Bash  
  \# Command for CREStereo model  
  trtexec \--onnx=crestereo.onnx \\  
          \--saveEngine=crestereo\_int8.engine \\  
          \--int8 \\  
          \--calib=crestereo\_calibration.cache \\  
          \--minShapes=left:1x3x375x1242,right:1x3x375x1242 \\  
          \--optShapes=left:1x3x375x1242,right:1x3x375x1242 \\  
          \--maxShapes=left:1x3x375x1242,right:1x3x375x1242

  \# Command for PointPillars model  
  trtexec \--onnx=pointpillars.onnx \\  
          \--saveEngine=pointpillars\_int8.engine \\  
          \--int8 \\  
          \--calib=pointpillars\_calibration.cache \\  
          \--minShapes=points:1x10000x4 \\  
          \--optShapes=points:1x60000x4 \\  
          \--maxShapes=points:1x120000x4

  **Argument Explanation:**  
  * \--onnx: Specifies the input ONNX model file.  
  * \--saveEngine: Defines the file path for the output TensorRT engine.  
  * \--int8: Enables the INT8 precision mode for inference.  
  * \--calib: Provides the path to the essential calibration cache file generated in the previous step.31  
  * \--minShapes, \--optShapes, \--maxShapes: These arguments are critical for models with dynamic input shapes. For PointPillars, the number of points in a pseudo-LiDAR scan varies, so these flags define the minimum, optimal, and maximum input tensor dimensions the engine should be optimized for.31

### **3.2. Qualcomm QCS6490: DLC Generation and Quantization**

For the Qualcomm platform, the ONNX models must be converted to the proprietary Deep Learning Container (.dlc) format and quantized for the Hexagon NPU.

* **Step 1: Convert to FP32 DLC:** Use the snpe-onnx-to-dlc tool from the SNPE SDK. This initial conversion creates a baseline 32-bit floating-point DLC file.18  
  Bash  
  snpe-onnx-to-dlc \--input\_network model.onnx \--output\_path model\_fp32.dlc

* **Step 2: Prepare Calibration Data:** The SNPE quantization tool, snpe-dlc-quantize, requires a text file that lists the absolute paths to raw input files.33 A Python script must be used to process a subset of the KITTI dataset (300-500 samples). For each sample, the image or point cloud data must be preprocessed (e.g., resized, normalized) and saved as a flat binary (  
  .raw) file. The script will then generate a calibration\_list.txt file containing one path per line.  
* **Step 3: Quantize to INT8 DLC for Hexagon NPU:** Use the snpe-dlc-quantize tool to perform post-training quantization on the FP32 DLC.  
  Bash  
  snpe-dlc-quantize \--input\_dlc model\_fp32.dlc \\  
                    \--input\_list calibration\_list.txt \\  
                    \--output\_dlc model\_quantized\_htp.dlc \\  
                    \--enable\_htp

  **Argument Explanation:**  
  * \--input\_dlc: The source FP32 DLC file.  
  * \--input\_list: The path to the text file containing the list of raw calibration data files.35  
  * \--output\_dlc: The file path for the final quantized INT8 DLC.  
  * \--enable\_htp: This is a critical flag that enables optimizations specifically for the Hexagon Tensor Processor (HTP), ensuring the model is compiled to run on the most efficient hardware block within the SoC.33

### **3.3. Radxa X4 (Intel): OpenVINO IR Generation and Quantization**

The workflow for the Intel platform involves converting the ONNX models to OpenVINO's Intermediate Representation (IR) format and then using the Post-Training Optimization Tool (POT) for quantization.

* **Step 1: Convert to FP32 Intermediate Representation (IR):** Use OpenVINO's Model Optimizer (mo) command-line tool to convert the .onnx models into the FP32 IR format, which consists of an .xml file (describing the network topology) and a .bin file (containing the weights).37  
  Bash  
  mo \--input\_model model.onnx \--output\_dir fp32\_model/

* **Step 2: Quantize to INT8 IR using the Post-Training Optimization Tool (POT):** The POT provides a structured and reproducible approach to quantization via a JSON configuration file.39 This method is preferred for its clarity and ease of modification. A complete  
  quantization\_config.json file must be created.  
  JSON  
  {  
    "model": {  
      "model\_name": "pointpillars\_kitti",  
      "model": "/path/to/fp32\_model/model.xml",  
      "weights": "/path/to/fp32\_model/model.bin"  
    },  
    "engine": {  
      "type": "simplified",  
      "data\_source": "/path/to/KITTI/object/training/image\_2\_calibration\_subset/"  
    },  
    "compression": {  
      "target\_device": "GPU",  
      "algorithms":  
    }  
  }

  **JSON Configuration Explanation:**  
  * model: Specifies the paths to the input FP32 IR (.xml and .bin) files.  
  * engine.data\_source: Defines the directory containing the subset of KITTI images to be used for calibration.  
  * compression.target\_device: This is the most crucial parameter. It must be set to "GPU" to ensure that the quantization parameters and model optimizations are tailored for execution on the integrated Intel UHD Graphics.38 Setting this to "CPU" would result in a sub-optimal model for the target hardware.  
* **Step 3: Execute POT:** Run the POT from the command line, pointing it to the configuration file. This will generate the final INT8 IR files.  
  Bash  
  pot \-c quantization\_config.json \--output-dir INT8\_model/

## **4\. Phase II: End-to-End Pipeline Implementation & Execution**

This phase details the construction and execution of the main application that integrates the optimized models and processing steps into a cohesive pipeline. A key consideration in this pipeline is that the total end-to-end latency is not merely the sum of the two model inference times. The intermediate step of converting the depth map to a pseudo-LiDAR point cloud is a geometric transformation that runs entirely on the platform's CPU. The performance of this step is dependent on the host CPU's architecture and clock speed (e.g., Arm Cortex-A78AE on Orin NX, Kryo 670 on QCS6490, Intel N100 on Radxa X4).10 A platform with a powerful CPU might achieve a lower total latency even with a slightly slower AI accelerator, or vice-versa. To conduct a fair and insightful benchmark, this CPU-bound conversion step must be timed independently to diagnose its contribution to the overall system latency.

### **4.1. Application Logical Flow**

The main application should be implemented in a high-performance language such as C++ or Python. The control flow must include precise timestamping at key stages to isolate the performance of each component. In C++, std::chrono::high\_resolution\_clock is recommended; in Python, time.perf\_counter() provides the necessary precision.42

1. **Initialization:**  
   * Load the platform-specific optimized models (TensorRT .engine, Qualcomm .dlc, OpenVINO .xml/.bin).  
   * Load the list of image IDs for the KITTI validation set.  
   * Initialize data structures for logging latencies.  
2. **Main Processing Loop:** Iterate through each image ID in the validation set.  
   * Load the corresponding left (image\_2) and right (image\_3) stereo images, and the camera calibration file (calib).  
   * Start the total pipeline timer: t\_pipeline\_start \= time.perf\_counter().  
   * **Stage 1 Inference:**  
     * Start the stereo model timer: t\_stereo\_start \= time.perf\_counter().  
     * Run inference with the CREStereo model on the stereo image pair.  
     * Stop the stereo model timer: t\_stereo\_stop \= time.perf\_counter().  
   * **Intermediate Conversion:**  
     * Start the conversion timer: t\_conversion\_start \= time.perf\_counter().  
     * Execute the pseudo-LiDAR generation algorithm on the output disparity map.  
     * Stop the conversion timer: t\_conversion\_stop \= time.perf\_counter().  
   * **Stage 2 Inference:**  
     * Start the PointPillars timer: t\_pp\_start \= time.perf\_counter().  
     * Run inference with the PointPillars model on the generated point cloud.  
     * Stop the PointPillars timer: t\_pp\_stop \= time.perf\_counter().  
   * Stop the total pipeline timer: t\_pipeline\_stop \= time.perf\_counter().  
   * **Logging:**  
     * Calculate and log the latency for each stage:  
       * stereo\_latency \= (t\_stereo\_stop \- t\_stereo\_start) \* 1000 (in ms)  
       * conversion\_latency \= (t\_conversion\_stop \- t\_conversion\_start) \* 1000 (in ms)  
       * pointpillars\_latency \= (t\_pp\_stop \- t\_pp\_start) \* 1000 (in ms)  
       * total\_latency \= (t\_pipeline\_stop \- t\_pipeline\_start) \* 1000 (in ms)  
   * **Output Generation:**  
     * Format the 3D bounding box detections from PointPillars into the official KITTI label format.  
     * Save the formatted detections to a unique .txt file (e.g., 000001.txt) in a designated results directory.  
3. **Termination:** After processing all validation images, write the complete list of logged latencies for all four categories to a summary .csv file for later analysis.

### **4.2. Core Logic: Pseudo-LiDAR Point Cloud Generation**

This section provides the core algorithm for converting the disparity map from CREStereo into a 3D point cloud, using the KITTI calibration data. A Python/NumPy implementation is provided for clarity.

Python

import numpy as np

def disparity\_to\_pointcloud(disparity\_map, calib\_file):  
    \# 1\. Parse calibration file  
    with open(calib\_file, 'r') as f:  
        lines \= f.readlines()  
        p2\_line \= \[line for line in lines if line.startswith('P2:')\]  
        p2\_matrix \= np.array(p2\_line.strip().split(' ')\[1:\], dtype=np.float32).reshape(3, 4)

    \# 2\. Extract intrinsic parameters  
    fx \= p2\_matrix  
    fy \= p2\_matrix  
    cx \= p2\_matrix  
    cy \= p2\_matrix  
    baseline \= \-p2\_matrix / fx

    \# 3\. Create pixel coordinate grid  
    h, w \= disparity\_map.shape  
    u\_coords, v\_coords \= np.meshgrid(np.arange(w), np.arange(h))

    \# 4\. Calculate depth from disparity  
    \# Avoid division by zero  
    valid\_mask \= disparity\_map \> 0  
    depth \= np.zeros\_like(disparity\_map)  
    depth\[valid\_mask\] \= (fx \* baseline) / disparity\_map\[valid\_mask\]

    \# 5\. Unproject 2D pixels to 3D camera coordinates  
    x\_cam \= (u\_coords \- cx) \* depth / fx  
    y\_cam \= (v\_coords \- cy) \* depth / fy  
    z\_cam \= depth

    \# 6\. Filter points outside a reasonable range (e.g., \> 80 meters deep)  
    range\_mask \= z\_cam \< 80.0  
    final\_mask \= np.logical\_and(valid\_mask, range\_mask)

    \# 7\. Stack into a point cloud (N, 3\)  
    points\_3d\_camera \= np.dstack((x\_cam, y\_cam, z\_cam))\[final\_mask\]

    \# Note: For PointPillars, points are typically used in Velodyne coordinates.  
    \# The transformation from camera to Velodyne coordinates is also needed  
    \# using the Tr\_velo\_to\_cam matrix from the calibration file.  
    \# This step is omitted here for brevity but is required for the pipeline.  
    \# The final output should be an (N, 4\) array with intensity as the 4th dim.  
      
    \# For this pipeline, we can add a dummy intensity channel  
    intensity \= np.ones((points\_3d\_camera.shape, 1), dtype=np.float32)  
    pointcloud \= np.hstack((points\_3d\_camera, intensity))

    return pointcloud

### **4.3. Core Logic: Formatting Detections for KITTI Evaluation**

The validity of the entire benchmark's accuracy measurement depends on producing detection files that are perfectly compliant with the KITTI evaluation script's strict format. The PointPillars model outputs raw tensor data representing bounding boxes (center coordinates, dimensions, yaw angle) in the Velodyne coordinate system. These must be transformed back into the camera coordinate system and serialized into a specific 15-column text format. A single formatting error can invalidate the mAP score.

The required format for each detected object per line in the output .txt file is 6:

type truncation occlusion alpha left top right bottom height width length x y z rotation\_y score  
A Python function must be implemented to perform this conversion, which involves applying the inverse coordinate transformations used during data preparation and formatting the floating-point numbers to the required precision.

### **4.4. Execution Command**

The end-to-end pipeline application should be executable from the command line. A template command is as follows:

Bash

python./run\_benchmark.py \\  
    \--platform nvidia \\  
    \--stereo\_model /path/to/crestereo\_int8.engine \\  
    \--detector\_model /path/to/pointpillars\_int8.engine \\  
    \--kitti\_dir /path/to/KITTI/object/ \\  
    \--val\_split /path/to/kitti\_splits/val.txt \\  
    \--results\_dir./results/nvidia/ \\  
    \--log\_file./logs/nvidia\_latencies.csv

## **5\. Phase III: Data Analysis and Metric Calculation**

This final phase details the procedures for processing the raw data generated by the benchmark application to calculate the final, comparable metrics.

### **5.1. Performance Metrics: Latency and Throughput**

A post-processing script (e.g., parse\_logs.py) is required to analyze the .csv file containing the logged latencies.

* **Total Latency (ms, p99):** The 99th percentile (p99) latency is a robust metric that reflects the worst-case performance experienced by the vast majority of frames, while being insensitive to extreme, unrepresentative outliers. To calculate it, load the 'Total Latency' column from the log file into an array, sort the array in ascending order, and select the value at the ceil(0.99 \* N)-th index, where N is the total number of frames processed.44  
* **Throughput (FPS):** Throughput is calculated as the inverse of the average total latency. Compute the arithmetic mean of all 'Total Latency' values from the log file and then calculate throughput as:  
  Throughput(FPS)=Mean Total Latency (ms)1000​

### **5.2. Power Efficiency Metric**

* **Data Acquisition:** During the entire benchmark run on the validation set, log power data from the Yokogawa WT300E power analyzer. This can be automated by connecting the analyzer to the network and using a Python script with a library such as pyModbusTCP to poll the relevant power measurement register via the Modbus/TCP protocol at a regular interval (e.g., every 100-500 ms).15  
* **Calculation:**  
  * **Average Power (W):** Calculate the arithmetic mean of all power samples collected from the Yokogawa analyzer during the run.  
  * **Power Efficiency (FPS/W):** This metric provides a holistic view of performance relative to power consumption. It is calculated as:  
    Power Efficiency=Average Power (W)Throughput (FPS)​

### **5.3. Accuracy Metric: 3D mAP (Moderate)**

The final detection accuracy must be calculated using an implementation of the official KITTI evaluation protocol to ensure the results are valid and comparable to published literature. The kitti-object-eval-python repository provides a widely accepted Python implementation.47

* **Step 1: Prepare Inputs:** Ensure you have three components:  
  1. The directory containing the ground truth labels from the KITTI dataset (KITTI/object/training/label\_2/).  
  2. The directory containing your pipeline's generated detection files (./results/\[platform\]/).  
  3. A val.txt file that lists the frame IDs of the validation split.  
* **Step 2: Execute Evaluation Script:** Run the evaluation from the command line. The following command evaluates the 'Car' class (class ID 0\) for the 3D detection task.  
  Bash  
  python /path/to/kitti-object-eval-python/evaluate.py evaluate \\  
      \--label\_path=/path/to/KITTI/object/training/label\_2 \\  
      \--result\_path=./results/\[platform\]/ \\  
      \--label\_split\_file=/path/to/kitti\_splits/val.txt \\  
      \--current\_class=0

* **Step 3: Interpret Results:** The script will output a table of Average Precision (AP) values for three difficulty levels (Easy, Moderate, Hard) and three IoU (Intersection over Union) thresholds (0.7 for cars, 0.5 for others). For this benchmark, record the **3D AP** for the **'Car'** class at the **'Moderate'** difficulty setting. This is the standard metric reported in most 3D object detection papers on the KITTI benchmark.

## **6\. Final Results Table**

Upon completion of all phases for each platform, the measured results should be consolidated into the following table. This table serves as the final deliverable of the benchmark, providing a clear, concise, and comprehensive comparison of the platforms across all key evaluation axes. The inclusion of sub-component latencies allows for a deeper analysis of where each platform's performance strengths and weaknesses lie.

| Metric | NVIDIA Jetson Orin NX | Qualcomm QCS6490 | Radxa X4 (Intel N100) |
| :---- | :---- | :---- | :---- |
| **Performance** |  |  |  |
| Stereo Latency (ms) |  |  |  |
| Conversion Latency (ms) |  |  |  |
| PointPillars Latency (ms) |  |  |  |
| **Total Latency (p99, ms)** |  |  |  |
| **Throughput (FPS)** |  |  |  |
| **Power** |  |  |  |
| Average Power (W) |  |  |  |
| **Power Efficiency (FPS/W)** |  |  |  |
| **Accuracy** |  |  |  |
| **3D mAP (Moderate, Car)** |  |  |  |

#### **Works cited**

1. mileyan/pseudo\_lidar: (CVPR 2019\) Pseudo-LiDAR from ... \- GitHub, accessed September 13, 2025, [https://github.com/mileyan/pseudo\_lidar](https://github.com/mileyan/pseudo_lidar)  
2. CREStereo | Luxonis \- Model Zoo, accessed September 13, 2025, [https://models.luxonis.com/luxonis/crestereo/4729a8bd-54df-467a-92ca-a8a5e70b52ab?backTo=%2F%3FbackTo%3D%252F%253FbackTo%253D%25252F%25253FbackTo%25253D%2525252F%2525253FbackTo%2525253D%252525252F](https://models.luxonis.com/luxonis/crestereo/4729a8bd-54df-467a-92ca-a8a5e70b52ab?backTo=/?backTo%3D%252F%253FbackTo%253D%25252F%25253FbackTo%25253D%2525252F%2525253FbackTo%2525253D%252525252F)  
3. Cross-spectral Gated-RGB Stereo Depth Estimation \- Princeton Computational Imaging Lab, accessed September 13, 2025, [https://light.princeton.edu/publication/gatedrccbstereo/](https://light.princeton.edu/publication/gatedrccbstereo/)  
4. Depth Estimation From Stereo Images Using Deep Learning | by Satya \- Medium, accessed September 13, 2025, [https://medium.com/@satya15july\_11937/depth-estimation-from-stereo-images-using-deep-learning-314952b8eaf9](https://medium.com/@satya15july_11937/depth-estimation-from-stereo-images-using-deep-learning-314952b8eaf9)  
5. mileyan/Pseudo\_Lidar\_V2: (ICLR) Pseudo-LiDAR++: Accurate Depth for 3D Object Detection in Autonomous Driving \- GitHub, accessed September 13, 2025, [https://github.com/mileyan/Pseudo\_Lidar\_V2](https://github.com/mileyan/Pseudo_Lidar_V2)  
6. KITTI 3D \- Supervisely, accessed September 13, 2025, [https://docs.supervisely.com/import-and-export/import/supported-annotation-formats/pointclouds/kitti3d](https://docs.supervisely.com/import-and-export/import/supported-annotation-formats/pointclouds/kitti3d)  
7. ExistenceMap-PointPillars: A Multifusion Network for Robust 3D Object Detection with Object Existence Probability Map \- MDPI, accessed September 13, 2025, [https://www.mdpi.com/1424-8220/23/20/8367](https://www.mdpi.com/1424-8220/23/20/8367)  
8. \[2509.05780\] 3DPillars: Pillar-based two-stage 3D object detection \- arXiv, accessed September 13, 2025, [https://arxiv.org/abs/2509.05780](https://arxiv.org/abs/2509.05780)  
9. NVIDIA Jetson Orin NX 16 GB Specs | TechPowerUp GPU Database, accessed September 13, 2025, [https://www.techpowerup.com/gpu-specs/jetson-orin-nx-16-gb.c4086](https://www.techpowerup.com/gpu-specs/jetson-orin-nx-16-gb.c4086)  
10. TEK6100-ORIN-NX \- TechNexion, accessed September 13, 2025, [https://www.technexion.com/products/embedded-computing/aivision/tek6100-orin-nx/](https://www.technexion.com/products/embedded-computing/aivision/tek6100-orin-nx/)  
11. Aikri QCS6490 System on Module (SoM) | eInfochips, accessed September 13, 2025, [https://www.einfochips.com/wp-content/uploads/2025/04/qualcomm-qcs6490-aikri-aikri-64x-90as-8-som-technical-datasheet.pdf](https://www.einfochips.com/wp-content/uploads/2025/04/qualcomm-qcs6490-aikri-aikri-64x-90as-8-som-technical-datasheet.pdf)  
12. MSC SM2S-QCS6490 \- Avnet Embedded, accessed September 13, 2025, [https://embedded.avnet.com/product/msc-sm2s-qcs6490/](https://embedded.avnet.com/product/msc-sm2s-qcs6490/)  
13. Radxa X4, accessed September 13, 2025, [https://docs.radxa.com/en/x/x4](https://docs.radxa.com/en/x/x4)  
14. Radxa X4, accessed September 13, 2025, [https://www.radxa.com/products/x/x4/](https://www.radxa.com/products/x/x4/)  
15. WT300E Digital Power Analyzer \- Yokogawa Test & Measurement, accessed September 13, 2025, [https://tmi.yokogawa.com/us/solutions/products/power-analyzers/digital-power-meter-wt300e/](https://tmi.yokogawa.com/us/solutions/products/power-analyzers/digital-power-meter-wt300e/)  
16. JetPack SDK 5.1.1 \- NVIDIA Developer, accessed September 13, 2025, [https://developer.nvidia.com/embedded/jetpack-sdk-511](https://developer.nvidia.com/embedded/jetpack-sdk-511)  
17. JetPack SDK 5.1 \- NVIDIA Developer, accessed September 13, 2025, [https://developer.nvidia.com/embedded/jetpack-sdk-51](https://developer.nvidia.com/embedded/jetpack-sdk-51)  
18. Qualcomm Neural Processing SDK \- Qualcomm® Linux ..., accessed September 13, 2025, [https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/building\_and\_executing\_tutorial.html?vproduct=1601111740013072\&version=1.5\&facet=Qualcomm%20Neural%20Processing%20SDK](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/building_and_executing_tutorial.html?vproduct=1601111740013072&version=1.5&facet=Qualcomm+Neural+Processing+SDK)  
19. Neural Processing SDK \- Qualcomm ID, accessed September 13, 2025, [https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/setup\_linux.html](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/setup_linux.html)  
20. OpenVINO™ is an open source toolkit for optimizing and deploying AI inference \- GitHub, accessed September 13, 2025, [https://github.com/openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino)  
21. Configurations for Intel® Processor Graphics (GPU) with OpenVINO, accessed September 13, 2025, [https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html](https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html)  
22. Configurations for Intel® Processor Graphics (GPU) with OpenVINO, accessed September 13, 2025, [https://docs.openvino.ai/2023.3/openvino\_docs\_install\_guides\_configurations\_for\_intel\_gpu.html](https://docs.openvino.ai/2023.3/openvino_docs_install_guides_configurations_for_intel_gpu.html)  
23. The KITTI Vision Benchmark Suite \- Jimmy S. Ren, accessed September 13, 2025, [https://www.jimmyren.com/papers/kitti\_stereo2015\_crl.pdf](https://www.jimmyren.com/papers/kitti_stereo2015_crl.pdf)  
24. KITTI stereo 2015 \- The KITTI Vision Benchmark Suite, accessed September 13, 2025, [https://www.cvlibs.net/datasets/kitti/eval\_scene\_flow.php?benchmark=stereo](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)  
25. Object Detection Evaluation \- The KITTI Vision Benchmark Suite, accessed September 13, 2025, [https://www.cvlibs.net/datasets/kitti/eval\_3dobject.php](https://www.cvlibs.net/datasets/kitti/eval_3dobject.php)  
26. ONNX \- Hugging Face, accessed September 13, 2025, [https://huggingface.co/docs/transformers/serialization](https://huggingface.co/docs/transformers/serialization)  
27. PyTorch \+ ONNX Runtime, accessed September 13, 2025, [https://onnxruntime.ai/pytorch](https://onnxruntime.ai/pytorch)  
28. Overview — NVIDIA TensorRT Documentation, accessed September 13, 2025, [https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html)  
29. TensorRT: Performing Inference In INT8 Using Custom Calibration \- ccoderun.ca, accessed September 13, 2025, [https://www.ccoderun.ca/programming/doxygen/tensorrt/md\_TensorRT\_samples\_opensource\_sampleINT8\_README.html](https://www.ccoderun.ca/programming/doxygen/tensorrt/md_TensorRT_samples_opensource_sampleINT8_README.html)  
30. INT8 calibration fails with trtexec · Issue \#4092 · NVIDIA/TensorRT \- GitHub, accessed September 13, 2025, [https://github.com/NVIDIA/TensorRT/issues/4092](https://github.com/NVIDIA/TensorRT/issues/4092)  
31. Command-Line Programs — NVIDIA TensorRT Documentation, accessed September 13, 2025, [https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html)  
32. Qualcomm Neural Processing SDK for AI Documentation, accessed September 13, 2025, [https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2?product=1601111740010412](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2?product=1601111740010412)  
33. Port a model using Qualcomm Neural Processing Engine SDK, accessed September 13, 2025, [https://docs.qualcomm.com/bundle/publicresource/topics/80-70017-15B/port-models.html](https://docs.qualcomm.com/bundle/publicresource/topics/80-70017-15B/port-models.html)  
34. Model Porting using SNPE \- Qualcomm ID, accessed September 13, 2025, [https://docs.qualcomm.com/bundle/publicresource/topics/80-70015-15B/snpe-port-model.html](https://docs.qualcomm.com/bundle/publicresource/topics/80-70015-15B/snpe-port-model.html)  
35. How could I draw bbox from output of YOLOV8 Quantized from Qualcomm SNPE (Inferenced by Hexagon DSP under uint8) \- Stack Overflow, accessed September 13, 2025, [https://stackoverflow.com/questions/79546256/how-could-i-draw-bbox-from-output-of-yolov8-quantized-from-qualcomm-snpe-infere](https://stackoverflow.com/questions/79546256/how-could-i-draw-bbox-from-output-of-yolov8-quantized-from-qualcomm-snpe-infere)  
36. rakelkar/snpe\_convert: Convert and run a TF model using Qualcomm SNPE tools \- GitHub, accessed September 13, 2025, [https://github.com/rakelkar/snpe\_convert](https://github.com/rakelkar/snpe_convert)  
37. Intel® Distribution of OpenVINO™ Toolkit Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors Based Platform, accessed September 13, 2025, [https://cdrdv2-public.intel.com/686401/openvino-toolkit-tuning-guide-on-xeon-v1.1.pdf](https://cdrdv2-public.intel.com/686401/openvino-toolkit-tuning-guide-on-xeon-v1.1.pdf)  
38. Intel OpenVINO Export \- Ultralytics YOLO Docs, accessed September 13, 2025, [https://docs.ultralytics.com/integrations/openvino/](https://docs.ultralytics.com/integrations/openvino/)  
39. Intel® Distribution of OpenVINO™ Toolkit Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors Based Platform, accessed September 13, 2025, [https://cdrdv2-public.intel.com/686401/openvino-toolkit-tuning-guide-on-xeon.pdf](https://cdrdv2-public.intel.com/686401/openvino-toolkit-tuning-guide-on-xeon.pdf)  
40. Post Training Quantization with OpenVINO Toolkit \- LearnOpenCV, accessed September 13, 2025, [https://learnopencv.com/post-training-quantization-with-openvino-toolkit/](https://learnopencv.com/post-training-quantization-with-openvino-toolkit/)  
41. Working with GPUs in OpenVINO, accessed September 13, 2025, [https://docs.openvino.ai/2024/notebooks/gpu-device-with-output.html](https://docs.openvino.ai/2024/notebooks/gpu-device-with-output.html)  
42. How to Measure Elapsed Time in C++? \- GeeksforGeeks, accessed September 13, 2025, [https://www.geeksforgeeks.org/cpp/how-to-measure-elapsed-time-in-cpp/](https://www.geeksforgeeks.org/cpp/how-to-measure-elapsed-time-in-cpp/)  
43. Measure execution time of a function in C++ \- GeeksforGeeks, accessed September 13, 2025, [https://www.geeksforgeeks.org/cpp/measure-execution-time-function-cpp/](https://www.geeksforgeeks.org/cpp/measure-execution-time-function-cpp/)  
44. controlplane.com, accessed September 13, 2025, [https://controlplane.com/community-blog/post/4-tips-to-improve-p99-latency\#:\~:text=To%20figure%20out%20the%20P99,of%20requests%20are%20faster%20than.](https://controlplane.com/community-blog/post/4-tips-to-improve-p99-latency#:~:text=To%20figure%20out%20the%20P99,of%20requests%20are%20faster%20than.)  
45. 4 Tips to Improve P99 Latency \- Control Plane, accessed September 13, 2025, [https://controlplane.com/community-blog/post/4-tips-to-improve-p99-latency](https://controlplane.com/community-blog/post/4-tips-to-improve-p99-latency)  
46. Quick start guide — pyModbusTCP 0.3.1.dev0 documentation, accessed September 13, 2025, [https://pymodbustcp.readthedocs.io/en/latest/quickstart/index.html](https://pymodbustcp.readthedocs.io/en/latest/quickstart/index.html)  
47. traveller59/kitti-object-eval-python: Fast kitti object detection ... \- GitHub, accessed September 13, 2025, [https://github.com/traveller59/kitti-object-eval-python](https://github.com/traveller59/kitti-object-eval-python)