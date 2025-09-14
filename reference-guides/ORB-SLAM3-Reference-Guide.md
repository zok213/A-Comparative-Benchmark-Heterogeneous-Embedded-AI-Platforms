

# **A Definitive Guide to Benchmarking CPU Performance with ORB-SLAM3**

## **Foundational Principles for Rigorous Benchmarking**

Benchmarking the performance of embedded systems is a scientific endeavor that demands rigor, precision, and a clear understanding of the underlying principles. The objective is not merely to generate numbers, but to produce results that are repeatable, comparable, and reflective of real-world performance. This guide establishes a standardized protocol for evaluating the CPU performance of diverse hardware platforms using ORB-SLAM3, a state-of-the-art Simultaneous Localization and Mapping algorithm, as the benchmark workload.

### **The Primacy of Real-World Workloads**

Computational benchmarks can be broadly categorized into two types: synthetic and application-level. Synthetic benchmarks, such as EEMBC's CoreMark, are designed to test specific, isolated functions of a processor core, like list processing, matrix manipulation, and cyclic redundancy checks.2 While valuable for assessing raw core functionality, they often fail to capture the complex interplay of a complete system under a realistic load. They do not adequately stress memory subsystems, cache hierarchies, I/O performance, or the operating system's scheduler, all of which are critical factors in the performance of a real application.1

For robotics and autonomous systems, an application-level benchmark is far more representative. ORB-SLAM3 is an ideal choice for this purpose. As a feature-based, visual-inertial SLAM library, it is computationally demanding and multi-threaded, exercising a wide range of system capabilities.4 Its workload includes:

* **Intensive Floating-Point and Integer Operations:** For feature extraction (ORB), geometric calculations, and non-linear optimization (Bundle Adjustment).  
* **Significant Memory Bandwidth Usage:** For manipulating image data and managing large map data structures.  
* **Complex CPU-Level Parallelism:** Involving distinct threads for tracking, local mapping, and loop closing, which stresses the CPU's multi-core management and the OS scheduler.6

By using ORB-SLAM3, the benchmark moves beyond measuring a CPU's theoretical peak performance to assessing its practical, sustained performance on a task that is directly relevant to the target domain of robotics and computer vision.

### **The Pillars of a Valid Benchmark**

To ensure that the results are meaningful and defensible, the experimental protocol must be built upon four foundational pillars: consistency, workload isolation, thermal stability, and accurate power measurement.

#### **Consistency and Repeatability**

The core goal of benchmarking is comparison. Therefore, the only variable under test should be the hardware platform itself. All other factors must be held constant. This requires an absolutely consistent software environment, including the same operating system version, C++ libraries, compiler, and build flags.7 The workload must also be deterministic; for this, a standardized, publicly available dataset—the EuRoC MAV dataset—is used for every test run.8 Even minor deviations in software can introduce performance variations that obscure the true hardware capabilities and invalidate the comparison.

#### **Workload Isolation**

Modern, multi-tasking operating systems like Linux are inherently non-deterministic. Background services, kernel tasks, and other user processes can be scheduled to run at any time, consuming CPU cycles and memory bandwidth, thereby interfering with the benchmark process.7 This interference introduces noise and variability into the measurements, making results difficult to reproduce. To achieve a scientifically valid result, the benchmark workload must be isolated from this OS-level activity as much as possible. A standard practice of simply closing other applications is insufficient for rigorous testing. The Linux kernel provides mechanisms for CPU core isolation (

isolcpus kernel parameter) and process affinity (taskset command). By dedicating specific CPU cores exclusively to the benchmark process, interference from the OS scheduler can be virtually eliminated, dramatically improving the consistency and reliability of the results.

#### **Thermal Stability**

The performance of a modern processor is inextricably linked to its thermal state. Under sustained high load, a CPU's temperature will rise. To prevent physical damage from overheating, processors employ a self-protection mechanism called thermal throttling, where the clock frequency is automatically reduced to lower power consumption and heat output.10 A benchmark conducted without adequate cooling will measure the system's burst performance before it overheats, not its sustained performance, which is the critical metric for long-running robotics applications. Therefore,

**active cooling is a mandatory component of the testbed**. An external fan and heatsink ensure that the CPU can operate at its maximum sustained frequency throughout the benchmark, preventing thermal throttling from becoming a confounding variable in the results.11

#### **Accurate Power Measurement**

For battery-powered robotic systems, performance-per-watt is often more important than raw performance. While many embedded platforms, such as NVIDIA's Jetson series, provide on-chip power sensors and command-line tools like tegrastats for monitoring power consumption, these internal sensors are not suitable for high-integrity benchmarking.13 Research has shown that these sensors can be inaccurate and often measure power at intermediate points in the power delivery network, failing to capture the total system draw.14 They also tend to report averaged values, potentially missing transient current peaks that are critical to understanding system behavior.14 For a definitive benchmark, the ground truth must be established using a high-precision, external power analyzer. This instrument is placed in series between the DC power supply and the device under test, measuring the total power entering the board with high accuracy and a fast update rate.15 This provides the authoritative data needed to calculate true power consumption and energy efficiency.

## **Required Testbed Hardware and Software Stack**

To adhere to the principle of consistency, this protocol specifies a precise bill of materials for the testbed. Any deviation from this list should be explicitly documented in the final results.

### **Hardware Platforms (Devices Under Test \- DUTs)**

The following platforms have been selected to provide a representative comparison across common embedded architectures.

* **ARM64 (NVIDIA):** NVIDIA Jetson Orin NX 16GB Developer Kit.  
  * **CPU:** 8-core Arm® Cortex®-A78AE v8.2 64-bit CPU (2MB L2 \+ 4MB L3).17  
  * **Memory:** 16GB 128-bit LPDDR5 (102.4 GB/s).18  
* **ARM64 (Qualcomm):** Qualcomm Robotics RB5 Development Kit.  
  * **CPU:** Qualcomm® Kryo™ 585 Octa-core 64-bit CPU (up to 2.84 GHz).19  
  * **Memory:** 8GB LPDDR5 (2750 MHz).20  
* **x86\_64:** ODROID-H4 Ultra.  
  * **CPU:** Intel® Processor N305 (8-Core, up to 3.8 GHz).  
  * **Memory:** User-provided DDR5 SO-DIMM (e.g., 16GB DDR5-4800). This platform is selected as a modern, power-efficient x86 competitor to high-end ARM SoCs.22

### **Measurement and Support Hardware**

* **Power Analyzer:** Yokogawa WT300E Digital Power Analyzer. This model is specified for its high basic accuracy of 0.1% reading \+0.05% range, fast data update rate of 100 ms, and its ability to accurately measure DC power, making it ideal for this application.15  
* **DC Power Supply:** A stable, lab-grade variable DC power supply capable of delivering at least 12V and 5A.  
* **Active Cooling:** A dedicated fan for each DUT is mandatory. This can be a purpose-built solution like the Raspberry Pi Active Cooler or a generic 40mm 5V fan securely mounted to a heatsink on the DUT's main processor.10 The fan should be powered externally or from the SBC's GPIO pins and run at 100% duty cycle throughout the test.  
* **Cabling and Connectors:** Appropriate DC barrel jack connectors for each DUT and high-gauge wire to create a series connection for current measurement through the power analyzer.

### **Software Stack**

A uniform software environment is critical for a valid comparison. All tests must be conducted using the following stack.

* **Operating System:** A fresh installation of **Ubuntu 20.04 LTS (Focal Fossa)** desktop image on each DUT. This specific version is chosen for its widespread use in robotics and known compatibility with the required dependencies.24  
* **Build Tools:** The essential tools for compiling C++ projects from source.  
  Bash  
  sudo apt update  
  sudo apt install build-essential cmake git g++

* **ORB-SLAM3 C++ Dependencies:**  
  * **Eigen3:** A C++ template library for linear algebra. The version in the Ubuntu 20.04 repositories is sufficient.  
    Bash  
    sudo apt install libeigen3-dev

    24  
  * **OpenCV:** An open-source computer vision library. The default version provided with the Ubuntu 20.04 desktop image (4.2.0) is compatible and sufficient for this benchmark, simplifying the setup process.4  
  * **Pangolin:** A library for managing OpenGL display and interaction. This dependency is a common point of failure and **must be built from source** to ensure compatibility across all platforms. The exact build procedure is detailed in the next section.4  
* **Analysis Software:** Python 3 and its scientific computing libraries are used for the data analysis phase.  
  Bash  
  sudo apt install python3-numpy python3-matplotlib

## **Step-by-Step Experimental Protocol**

This section provides the precise, sequential commands and procedures required to prepare the system, collect data, and analyze the results. These steps should be followed meticulously on each DUT.

### **Phase I: System Preparation and Compilation**

This phase establishes a consistent build environment and compiles the necessary libraries and the ORB-SLAM3 application from source. The process of compilation itself can introduce performance variations if not standardized; therefore, this canonical build process must be followed exactly.

#### **1\. Install Core Dependencies**

Execute the following commands on a fresh installation of Ubuntu 20.04 to install all required system packages:

Bash

sudo apt update  
sudo apt install build-essential cmake git g++ libeigen3-dev python3-numpy python3-matplotlib

#### **2\. Build Pangolin from Source**

Pangolin must be built from source to ensure a consistent version and feature set across all DUTs.

Bash

\# Navigate to your home directory or a preferred development folder  
cd \~

\# Clone the Pangolin repository and its required submodules  
git clone \--recursive https://github.com/stevenlovegrove/Pangolin.git  
cd Pangolin

\# Create a build directory and configure the project  
cmake \-B build

\# Compile the library using all available CPU cores  
cmake \--build build \-j$(nproc)

\# Install the library to the system path (e.g., /usr/local/lib)  
sudo cmake \--build build \--target install

27

#### **3\. Build ORB-SLAM3 from Source**

This is the procedure for building the benchmark application itself.

Bash

\# Navigate back to your home directory or development folder  
cd \~

\# Clone the official ORB-SLAM3 repository  
git clone https://github.com/UZ-SLAMLab/ORB\_SLAM3.git  
cd ORB\_SLAM3

\# CRITICAL STEP: Modify the CMakeLists.txt file to use the C++14 standard,  
\# which is required for compatibility with the g++ compiler on Ubuntu 20.04.  
sed \-i 's/c++11/c++14/g' CMakeLists.txt

\# Make the build script executable  
chmod \+x build.sh

\# Execute the build script. This will build the internal DBoW2 and g2o  
\# libraries, followed by the main ORB\_SLAM3 library and examples.  
./build.sh

25

#### **4\. Architecture-Specific Build Considerations (ARM64)**

Embedded ARM64 platforms like the NVIDIA Jetson and Qualcomm Robotics boards often have more limited RAM and may not have swap configured by default compared to x86 systems. The build.sh script for ORB-SLAM3 invokes make \-j by default, which attempts to use all available CPU cores for compilation. On an 8-core system with limited memory, this can lead to an out-of-memory condition, causing the compiler to crash and the build to fail.29

To mitigate this, edit the build.sh script before running it:

1. Open the script: nano build.sh  
2. Locate all instances of the make \-j command.  
3. Change them to a more conservative value, such as make \-j4. This limits the build to four parallel jobs, reducing peak memory usage.

### **Phase II: Data Collection**

This phase details the physical setup of the measurement hardware and the execution of the benchmark run.

#### **1\. Download and Prepare the EuRoC Dataset**

The benchmark uses the "Machine Hall 01" sequence from the EuRoC MAV dataset, which provides a consistent visual-inertial input.

Bash

\# Create a directory to store datasets  
mkdir \-p \~/Datasets/EuRoc  
cd \~/Datasets/EuRoc

\# Download the dataset sequence (approx. 1.2 GB)  
wget \-c http://robotics.ethz.ch/\~asl-datasets/ijrr\_euroc\_mav\_dataset/machine\_hall/MH\_01\_easy/MH\_01\_easy.zip

\# Create a directory for the sequence and unzip the data  
mkdir MH01  
unzip MH\_01\_easy.zip \-d MH01/

8

#### **2\. Configure the Power Measurement Hardware**

The power analyzer must be connected correctly to measure the total DC power consumed by the DUT.

1. **Wiring:** Connect the DC power supply to the **current input** terminals of the Yokogawa WT300E. Connect the DUT's power input (e.g., its DC barrel jack) to the **current output** terminals of the power analyzer. This places the analyzer's current shunt in series with the DUT. Connect the analyzer's **voltage input** terminals in parallel across the DUT's power input terminals.  
   \+-----------+      \+----------------+      \+----------------+      \+-----+

| DC Supply |-----\>| Current IN (A) |-----\>| Current OUT(A) |-----\>| DUT |  
\+-----------+ | Yokogawa | | Yokogawa | \+-----+  
| | | | |  
| Voltage IN (V) |\<-----+----------------+--------+  
\+----------------+  
\`\`\`

30

2\. Analyzer Setup:  
\* Power on the Yokogawa WT300E.  
\* Set the measurement mode to DC.  
\* Set the voltage range appropriately (e.g., the 15 V range for a 12V supply).  
\* Set the current range appropriately (e.g., the 1 A or 5 A range).  
\* Configure the display to show Average Power (W).  
\* Set the data update rate to its fastest setting, typically 100 ms.15

\* Begin logging the power data either via the device's internal memory or through a connected PC using Yokogawa's software. Start the logging just before executing the benchmark command and stop it immediately after the program finishes.

#### **3\. Execute the Benchmark Run**

Before execution, ensure the system is in a quiescent state with no unnecessary applications running. Ensure the active cooling fan is operating at 100%.

1. Navigate to the ORB-SLAM3 directory:  
   Bash  
   cd \~/ORB\_SLAM3

2. Execute the following command. This command runs the monocular-inertial example on the prepared EuRoC dataset. The \> and 2\>&1 operators redirect all standard output and standard error streams into a single log file named run\_log.txt for subsequent automated analysis.

./Examples/Monocular-Inertial/mono\_inertial\_euroc./Vocabulary/ORBvoc.txt./Examples/Monocular-Inertial/EuRoC.yaml \~/Datasets/EuRoc/MH01./Examples/Monocular-Inertial/EuRoC\_TimeStamps/MH01.txt \> run\_log.txt 2\>&1  
\`\`\`

4

### **Phase III: Data Analysis & Scripting**

This phase involves extracting the key performance indicators from the collected data.

#### **1\. Defining Key Performance Metrics**

* **Average Power (W):** This metric is calculated from the data logged by the Yokogawa power analyzer. It is the mean of all power readings recorded between the start and end of the benchmark execution.  
* **Throughput (FPS):** A measure of overall processing speed, calculated as the total number of frames processed divided by the total execution time. A higher FPS indicates better performance.3  
* **99th Percentile (p99) Latency (ms):** This is a robust statistical measure of frame processing time. It represents the value below which 99% of all frame processing times fall. Unlike a simple average, p99 latency is less sensitive to extreme outliers and provides a much better indication of consistent, predictable real-time performance, which is critical for robotics applications where occasional long delays can be catastrophic.3

#### **2\. Log File Parsing and Metric Calculation**

The run\_log.txt file contains the frame-by-frame processing times needed to calculate throughput and latency. The following Python script automates the extraction and calculation of these metrics.

Python

\# Filename: parse\_orb\_log.py  
\# Description: Parses the output log from an ORB-SLAM3 run to calculate  
\#              performance metrics: throughput (FPS) and 99th percentile latency.

import re  
import numpy as np  
import sys

def parse\_log\_file(log\_file\_path):  
    """  
    Parses an ORB-SLAM3 log file to extract frame processing times.

    Args:  
        log\_file\_path (str): The path to the run\_log.txt file.

    Returns:  
        list: A list of frame processing times in milliseconds (float).  
              Returns an empty list if the file cannot be read or no times are found.  
    """  
    processing\_times\_ms \=  
    \# Regular expression to find lines containing the frame processing time.  
    \# It captures the floating-point number representing the time in milliseconds.  
    \# Example line: "Frame processing time: 25.34 ms"  
    time\_regex \= re.compile(r"Frame processing time: (\\d+\\.\\d+)\\s+ms")

    try:  
        with open(log\_file\_path, 'r') as f:  
            for line in f:  
                match \= time\_regex.search(line)  
                if match:  
                    \# Convert the captured string to a float and append to the list  
                    time\_ms \= float(match.group(1))  
                    processing\_times\_ms.append(time\_ms)  
    except FileNotFoundError:  
        print(f"Error: Log file not found at '{log\_file\_path}'")  
        return

    return processing\_times\_ms

def calculate\_metrics(processing\_times\_ms):  
    """  
    Calculates performance metrics from a list of processing times.

    Args:  
        processing\_times\_ms (list): A list of frame processing times in ms.

    Returns:  
        dict: A dictionary containing the calculated metrics.  
    """  
    if not processing\_times\_ms:  
        return {  
            "total\_frames": 0,  
            "total\_duration\_s": 0.0,  
            "throughput\_fps": 0.0,  
            "p99\_latency\_ms": 0.0,  
            "mean\_latency\_ms": 0.0,  
            "std\_dev\_latency\_ms": 0.0  
        }

    \# Convert list to a NumPy array for efficient computation  
    times\_array \= np.array(processing\_times\_ms)

    \# Calculate metrics  
    total\_frames \= len(times\_array)  
    total\_duration\_s \= np.sum(times\_array) / 1000.0  \# Convert total ms to seconds  
    throughput\_fps \= total\_frames / total\_duration\_s if total\_duration\_s \> 0 else 0.0  
    p99\_latency\_ms \= np.percentile(times\_array, 99)  
    mean\_latency\_ms \= np.mean(times\_array)  
    std\_dev\_latency\_ms \= np.std(times\_array)

    return {  
        "total\_frames": total\_frames,  
        "total\_duration\_s": total\_duration\_s,  
        "throughput\_fps": throughput\_fps,  
        "p99\_latency\_ms": p99\_latency\_ms,  
        "mean\_latency\_ms": mean\_latency\_ms,  
        "std\_dev\_latency\_ms": std\_dev\_latency\_ms  
    }

if \_\_name\_\_ \== "\_\_main\_\_":  
    if len(sys.argv)\!= 2:  
        print("Usage: python3 parse\_orb\_log.py \<path\_to\_run\_log.txt\>")  
        sys.exit(1)

    log\_file \= sys.argv  
    times \= parse\_log\_file(log\_file)

    if times:  
        metrics \= calculate\_metrics(times)  
        print("--- ORB-SLAM3 Performance Analysis \---")  
        print(f"Log File:           {log\_file}")  
        print(f"Total Frames:       {metrics\['total\_frames'\]}")  
        print(f"Total Duration:     {metrics\['total\_duration\_s'\]:.2f} s")  
        print("-" \* 36)  
        print(f"Throughput (FPS):   {metrics\['throughput\_fps'\]:.2f}")  
        print(f"P99 Latency (ms):   {metrics\['p99\_latency\_ms'\]:.2f}")  
        print("-" \* 36)  
        print(f"Mean Latency (ms):  {metrics\['mean\_latency\_ms'\]:.2f}")  
        print(f"Std Dev Latency(ms):{metrics\['std\_dev\_latency\_ms'\]:.2f}")  
        print("--------------------------------------")  
    else:  
        print("No frame processing times found in the log file.")

To use the script, save it as parse\_orb\_log.py and run it from the terminal, passing the path to the log file as an argument:

Bash

python3 parse\_orb\_log.py run\_log.txt

## **Final Table and Best Practices**

The final step is to collate the results into a standardized format and review a checklist of best practices to ensure the quality and defensibility of the findings.

### **Results Summary Table**

Present the final, averaged results from multiple runs in a clear, tabular format. This facilitates direct comparison between the platforms. The "Performance/Watt" metric is a crucial derived value for assessing energy efficiency, calculated as Throughput (FPS) / Average Power (W).

**Table: ORB-SLAM3 Performance Benchmark Results**

| Hardware Platform | CPU Architecture | Average Power (W) | Throughput (FPS) | p99 Latency (ms) | Performance/Watt (FPS/W) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| NVIDIA Jetson Orin NX 16GB | ARM64 |  |  |  |  |
| Qualcomm Robotics RB5 | ARM64 |  |  |  |  |
| ODROID-H4 Ultra | x86\_64 |  |  |  |  |

### **Pro Tips for Reliable and Defensible Results**

Adhering to the following best practices will significantly enhance the quality and reproducibility of the benchmark results.

* **Run, Re-run, and Average:** A single measurement is not a reliable result. Execute the entire benchmark protocol (Phase II and III) at least 3 to 5 times for each DUT. Report the mean and standard deviation for each metric. A low standard deviation indicates a stable and repeatable measurement.7  
* **Ensure a Quiescent State:** Before each run, use a system monitoring tool like htop to verify that the system is idle and that no unexpected background processes are consuming significant CPU resources.  
* **Achieve Thermal Equilibrium:** Allow the DUT to idle for at least five minutes after power-on before starting the first benchmark run. This allows the system to reach a stable idle temperature. Maintain a consistent time interval (e.g., 2-3 minutes) between consecutive runs to ensure each test starts from a similar thermal state.  
* **Lock Power States and Clocks:** On platforms that support it, disable dynamic voltage and frequency scaling (DVFS) to prevent the CPU/GPU clocks from changing during the run. For NVIDIA Jetson devices, this can be accomplished by setting a fixed power mode with nvpmodel and then running sudo jetson\_clocks to lock the frequencies to their maximum values for that mode.13 This removes a major source of performance variability.  
* **Document Everything:** Meticulously record the complete experimental context. This includes the exact OS image version, versions of all key libraries (OpenCV, Eigen3), the specific git commit hash of the ORB-SLAM3 and Pangolin repositories used for compilation, and the ambient room temperature. This level of documentation is essential for others to be able to reproduce the results.

#### **Works cited**

1. Benchmark (computing) \- Wikipedia, accessed September 13, 2025, [https://en.wikipedia.org/wiki/Benchmark\_(computing)](https://en.wikipedia.org/wiki/Benchmark_\(computing\))  
2. CoreMark \- CPU Benchmark \- EEMBC, accessed September 13, 2025, [https://www.eembc.org/coremark/](https://www.eembc.org/coremark/)  
3. How to Read and Understand CPU Benchmarks \- Intel, accessed September 13, 2025, [https://www.intel.com/content/www/us/en/gaming/resources/read-cpu-benchmarks.html](https://www.intel.com/content/www/us/en/gaming/resources/read-cpu-benchmarks.html)  
4. UZ-SLAMLab/ORB\_SLAM3: ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM \- GitHub, accessed September 13, 2025, [https://github.com/UZ-SLAMLab/ORB\_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)  
5. ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM \- arXiv, accessed September 13, 2025, [https://arxiv.org/abs/2007.11898](https://arxiv.org/abs/2007.11898)  
6. ORB-SLAM3 Monocular Testing and Future Ideas \- HILTI Challenge, accessed September 13, 2025, [https://hilti-challenge.com/submissions/ORB-SLAM3%20Mono%20Testing%20-%20Andreu%20Gimenez%20and%20Daniel%20Casado/report.pdf](https://hilti-challenge.com/submissions/ORB-SLAM3%20Mono%20Testing%20-%20Andreu%20Gimenez%20and%20Daniel%20Casado/report.pdf)  
7. how do you properly benchmark? : r/cpp \- Reddit, accessed September 13, 2025, [https://www.reddit.com/r/cpp/comments/1179ho8/how\_do\_you\_properly\_benchmark/](https://www.reddit.com/r/cpp/comments/1179ho8/how_do_you_properly_benchmark/)  
8. The EuRoC MAV Datasets \- ResearchGate, accessed September 13, 2025, [https://www.researchgate.net/profile/Michael\_Burri2/publication/291954561\_The\_EuRoC\_micro\_aerial\_vehicle\_datasets/links/56af0c6008ae19a38516937c.pdf](https://www.researchgate.net/profile/Michael_Burri2/publication/291954561_The_EuRoC_micro_aerial_vehicle_datasets/links/56af0c6008ae19a38516937c.pdf)  
9. kmavvisualinertialdatasets – ASL Datasets, accessed September 13, 2025, [https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)  
10. Active Cooling Fan Add-on Board, accessed September 13, 2025, [https://ubitap.com/reader?store-page=Active-Cooling-Fan-Add-on-Board-p145064209](https://ubitap.com/reader?store-page=Active-Cooling-Fan-Add-on-Board-p145064209)  
11. Buy a Raspberry Pi Active Cooler, accessed September 13, 2025, [https://www.raspberrypi.com/products/active-cooler/](https://www.raspberrypi.com/products/active-cooler/)  
12. Official Raspberry Pi 5 Active Cooler : ID 5815 \- Adafruit, accessed September 13, 2025, [https://www.adafruit.com/product/5815](https://www.adafruit.com/product/5815)  
13. Power Optimization with NVIDIA Jetson | NVIDIA Technical Blog, accessed September 13, 2025, [https://developer.nvidia.com/blog/power-optimization-with-nvidia-jetson/](https://developer.nvidia.com/blog/power-optimization-with-nvidia-jetson/)  
14. Accurate Calibration of Power Measurements from Internal Power Sensors on NVIDIA Jetson Devices \- Unipd, accessed September 13, 2025, [https://www.research.unipd.it/retrieve/d3298a28-1a3f-491c-ab1e-67ba14c7ec02/Accurate\_Calibration\_of\_Power\_Measurements\_from\_Internal\_Power\_Sensors\_on\_NVIDIA\_Jetson\_Devices.pdf](https://www.research.unipd.it/retrieve/d3298a28-1a3f-491c-ab1e-67ba14c7ec02/Accurate_Calibration_of_Power_Measurements_from_Internal_Power_Sensors_on_NVIDIA_Jetson_Devices.pdf)  
15. WT300E Digital Power Analyzer \- Yokogawa Test & Measurement, accessed September 13, 2025, [https://tmi.yokogawa.com/us/solutions/products/power-analyzers/digital-power-meter-wt300e/](https://tmi.yokogawa.com/us/solutions/products/power-analyzers/digital-power-meter-wt300e/)  
16. How to measure power consumption of a device | Rohde & Schwarz, accessed September 13, 2025, [https://www.rohde-schwarz.com/us/products/test-and-measurement/essentials-test-equipment/dc-power-supplies/how-to-measure-power-consumption-of-a-device\_258278.html](https://www.rohde-schwarz.com/us/products/test-and-measurement/essentials-test-equipment/dc-power-supplies/how-to-measure-power-consumption-of-a-device_258278.html)  
17. Intro and Specs | Nvidia Jetson Orin / Nano NX \- Turing Pi, accessed September 13, 2025, [https://docs.turingpi.com/docs/nvidia-jetson-orin-nx-intro-specs](https://docs.turingpi.com/docs/nvidia-jetson-orin-nx-intro-specs)  
18. TEK6100-ORIN-NX \- TechNexion, accessed September 13, 2025, [https://www.technexion.com/products/embedded-computing/aivision/tek6100-orin-nx/](https://www.technexion.com/products/embedded-computing/aivision/tek6100-orin-nx/)  
19. Qualcomm Robotics RB5 & RB6 Platform Support \- RidgeRun Developer Wiki, accessed September 13, 2025, [https://developer.ridgerun.com/wiki/index.php/Qualcomm\_Robotics\_RB5](https://developer.ridgerun.com/wiki/index.php/Qualcomm_Robotics_RB5)  
20. Qualcomm® Robotics RB5 Development Platform \- 96Boards, accessed September 13, 2025, [https://www.96boards.org/product/qualcomm-robotics-rb5/](https://www.96boards.org/product/qualcomm-robotics-rb5/)  
21. Qualcomm® Robotics RB5 Platform, accessed September 13, 2025, [https://www.macnica.co.jp/en/business/semiconductor/manufacturers/qualcomm/products/134616/](https://www.macnica.co.jp/en/business/semiconductor/manufacturers/qualcomm/products/134616/)  
22. Single Board Computers \- ameriDroid, accessed September 13, 2025, [https://ameridroid.com/collections/single-board-computer](https://ameridroid.com/collections/single-board-computer)  
23. Yokogawa WT300E \- Digital Power Analyzer \- Instru-Measure, accessed September 13, 2025, [https://instru-measure.com/yokogawa-wt300e-digital-power-analyzer/](https://instru-measure.com/yokogawa-wt300e-digital-power-analyzer/)  
24. thien94/orb\_slam3\_ros: A ROS implementation of ORB\_SLAM3 \- GitHub, accessed September 13, 2025, [https://github.com/thien94/orb\_slam3\_ros](https://github.com/thien94/orb_slam3_ros)  
25. thien94/orb\_slam3\_ros\_wrapper: A ROS wrapper for ORB-SLAM3. Focus on portability and flexibility. \- GitHub, accessed September 13, 2025, [https://github.com/thien94/orb\_slam3\_ros\_wrapper](https://github.com/thien94/orb_slam3_ros_wrapper)  
26. rajivbishwokarma/orb\_slam3\_ros\_xavier: Running the ROS implementation of ORB\_SLAM3 in NVIDIA Jetson Xavier NX \- GitHub, accessed September 13, 2025, [https://github.com/rajivbishwokarma/orb\_slam3\_ros\_xavier](https://github.com/rajivbishwokarma/orb_slam3_ros_xavier)  
27. stevenlovegrove/Pangolin: Pangolin is a lightweight ... \- GitHub, accessed September 13, 2025, [https://github.com/stevenlovegrove/Pangolin](https://github.com/stevenlovegrove/Pangolin)  
28. ORB-SLAM3 0n ubuntu 22 \- GitHub, accessed September 13, 2025, [https://github.com/bharath5673/ORB-SLAM3](https://github.com/bharath5673/ORB-SLAM3)  
29. OrbSLAM3 won't build? \- Jetson Xavier NX \- NVIDIA Developer Forums, accessed September 13, 2025, [https://forums.developer.nvidia.com/t/orbslam3-wont-build/176844](https://forums.developer.nvidia.com/t/orbslam3-wont-build/176844)  
30. How to Use Power Quality Analyzers \- YouTube, accessed September 13, 2025, [https://www.youtube.com/watch?v=C-Ik6PTSQ-o](https://www.youtube.com/watch?v=C-Ik6PTSQ-o)  
31. How to verify the correct installation of a power quality analyzer such as the Fluke 435 SII, accessed September 13, 2025, [https://www.youtube.com/watch?v=4UhToihoG0I](https://www.youtube.com/watch?v=4UhToihoG0I)  
32. Embedded Processor IP for AI SoCs: 7 Benchmarking Tips | Synopsys Blog, accessed September 13, 2025, [https://www.synopsys.com/blogs/chip-design/embedded-processor-ip-ai-socs-7-tips.html](https://www.synopsys.com/blogs/chip-design/embedded-processor-ip-ai-socs-7-tips.html)  
33. Measuring maximum power consumption \- Jetson AGX Orin \- NVIDIA Developer Forums, accessed September 13, 2025, [https://forums.developer.nvidia.com/t/measuring-maximum-power-consumption/319008](https://forums.developer.nvidia.com/t/measuring-maximum-power-consumption/319008)