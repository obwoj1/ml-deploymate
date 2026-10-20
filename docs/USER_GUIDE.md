# ML DeployMate User Guide

ML DeployMate helps users monitor a machine-learning model for data drift.
Data drift happens when incoming data becomes different from the data used
to train the model.

The application includes an Overview, Live Monitor, Alerts, Model Performance,
and Config Generator page.

## Getting Started

1. Open the hosted application or start the dashboard locally.
2. To start it locally, run this command from the project folder:

   streamlit run app/Home.py

3. Open the local address displayed in the terminal, usually
   http://localhost:8501.
4. Allow the application to finish loading. The first startup may take
   longer because the application downloads the dataset and trains the model.
5. Check the Home page for the application's status and any demo-data banner.
6. Use the sidebar to move between pages.

The monitoring demonstration uses replayed census data with simulated changes.
It is not connected to a real production data stream. If the backend cannot
initialize, the application can fall back to mock demonstration data.

## Overview

### Purpose

The Overview page provides a quick summary of model health and monitoring
activity.

### Steps

1. Select Overview from the sidebar.
2. Review the model status.
3. Review the displayed accuracy to understand the model's recorded performance.
4. Check the batch count to see how much monitoring activity is reported.
5. Check the alert count for detected data changes.
6. Open Live Monitor to run a monitoring scenario.
7. Open Alerts to investigate available alerts.

### Understanding the Results

- Model status summarizes the health information reported by the application.
- Accuracy describes how often the model's predictions matched the correct
  labels in the reported evaluation.
- A batch is a group of data records checked together.
- Alerts indicate detected changes in the input data.

Recorded test accuracy does not guarantee that the model will perform equally
well on future data. Drift alerts help identify changes that need investigation.

## Live Monitor

### Purpose

The Live Monitor page demonstrates how DeployMate checks incoming batches
against reference data from the training dataset.

### Steps

1. Select Live Monitor from the sidebar.
2. Choose a monitoring scenario from the available options.
3. Use the page's run control to start the scenario.
4. Watch the displayed batch results and charts.
5. Look for changes in the input data and any reported alerts.
6. Open Alerts to investigate the detected changes.

### Available Scenarios

| Scenario | What It Demonstrates |
| --- | --- |
| healthy | Unmodified data used to demonstrate normal traffic. |
| gradual_shift | Ages gradually increase during the run. |
| sudden_shift | Hours worked per week suddenly increase. |
| category_shift | The distribution of occupations changes. |
| missing_values | Missing employer, occupation, and country values increase. |
| noise | Random variation is added to age and working hours. |

### Suggested First Run

1. Run healthy to observe normal monitoring behavior.
2. Run sudden_shift to observe a clear change.
3. Compare the monitoring results.
4. Review the resulting alerts.

### Understanding the Results

DeployMate compares each batch with a reference sample representing normal data.
Its drift tests check whether numeric values or category distributions have
changed enough to cross the configured thresholds.

Model-behavior anomaly indicators provide additional information. They do not
independently trigger drift alerts in the documented implementation.

A drift alert means the data changed. It does not, by itself, prove that the
model's accuracy decreased.