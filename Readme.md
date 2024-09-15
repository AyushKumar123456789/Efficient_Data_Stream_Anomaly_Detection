# Anomaly Detection System

## Overview

The **Anomaly Detection System** is a comprehensive tool designed to identify anomalies in continuous streams of floating-point data. This system integrates multiple anomaly detection methods, including Exponential Moving Average (EMA), Isolation Forest, and seasonal-based techniques. It is capable of adapting to new data patterns and visualizing anomalies in real-time, making it useful for applications requiring timely anomaly detection and monitoring.

## Features

- **Multi-Method Anomaly Detection**: Utilizes EMA, Isolation Forest, and seasonal-based anomaly detection to identify anomalies from different perspectives.
- **Real-Time Data Stream Simulation**: Generates a continuous data stream with seasonal variations, noise, and occasional anomalies.
- **Adaptive Learning**: Online learning through periodic updates to the Isolation Forest model to handle concept drift.
- **Interactive Visualization**: Provides real-time visualizations of the data stream, anomaly scores, and extracted trend/seasonal components.

## How It Works

### Anomaly Detection Methods

1. **Exponential Moving Average (EMA)**:
   - **Purpose**: Smooths the data and detects anomalies based on deviation from the smoothed value.
   - **Mechanism**: Updates the moving average with each new data point. Anomalies are detected if the deviation from the EMA exceeds a dynamically calculated threshold.
2. **Isolation Forest**:

   - **Purpose**: Detects anomalies by isolating observations in a tree-based model.
   - **Mechanism**: Constructs multiple isolation trees and measures anomaly scores based on how easily a data point can be isolated. The model is periodically retrained to adapt to new data patterns and detect concept drift.

3. **Seasonal-Based Detection**:
   - **Purpose**: Identifies anomalies based on seasonal patterns in the data.
   - **Mechanism**: Computes seasonal averages and thresholds to detect anomalies when data points deviate significantly from expected seasonal behavior.

### Handling Concept Drift

- **Concept Drift Detection**: The system tracks the number of data points processed and performs periodic retraining of the Isolation Forest model. If concept drift is suspected (e.g., if the number of data points exceeds a predefined interval), the model is retrained using the latest data to adapt to changes in data distribution.

### Handling Seasonal Variation

- **Seasonal Adjustment**: The seasonal-based detection method computes averages over seasonal windows and adjusts thresholds dynamically to account for periodic patterns in the data.

## Visualizations

The system generates the following real-time visualizations:

1. **Data Stream Plot**:

   - **Displays**: The continuous data stream with real-time updates.
   - **Features**: Marks detected anomalies using different markers (e.g., `^` for EMA, `s` for Isolation Forest, `*` for Seasonal).

2. **Anomaly Scores Plot**:

   - **Displays**: Anomaly scores calculated by the detection methods.
   - **Features**: Includes a threshold line indicating the anomaly detection threshold.

3. **Seasonality and Trend Plot**:
   - **Displays**: Extracted trend and seasonal components from the data.
   - **Features**: Shows trends and seasonal variations over time to help visualize the underlying data patterns.

![Screenshot (141)](https://github.com/user-attachments/assets/9cf2841f-ef8d-459b-b45e-50a27c5a45a7)


## Algorithms and Rationale

1. **Exponential Moving Average (EMA)**:

   - **Rationale**: EMA is chosen for its simplicity and effectiveness in smoothing data and detecting deviations. It provides a straightforward method for anomaly detection based on recent data trends.

2. **Isolation Forest**:

   - **Rationale**: Isolation Forest is effective for high-dimensional data and can handle large data sets. It is chosen for its ability to isolate anomalies and adapt to changing data patterns through periodic retraining.

3. **Seasonal-Based Detection**:
   - **Rationale**: Seasonal-based detection captures periodic patterns in the data, making it suitable for data with regular seasonal variations. It complements other methods by focusing on periodic deviations.

## Installation

To run this project, you need Python 3.6 or later. Install the required libraries using:

```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file with the following content:

```
numpy
scikit-learn
matplotlib
```

## Usage

1. **Run the Main Script**: Execute `run.py` to start the anomaly detection and visualization process.

   ```bash
   python run.py
   ```

2. **Observe Visualizations**: The real-time visualizer will display the data stream, detected anomalies, anomaly scores, and trend/seasonal components.

## Customization

- **Adjust Parameters**: Modify parameters such as `window_size`, `alpha`, `contamination`, and `season_length` in `anomaly_detection.py` to fit your specific use case.
- **Modify Data Simulation**: Change the `simulate_data_stream` function in `data_stream.py` to simulate different data characteristics or anomalies.

## Acknowledgements

- The project uses [scikit-learn](https://scikit-learn.org/) for machine learning algorithms.
- Data visualization is powered by [Matplotlib](https://matplotlib.org/).
