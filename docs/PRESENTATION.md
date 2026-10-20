# ML DeployMate Presentation Plan

Course: COSC 472 — Fall 2026  
Team: Obaloluwa Wojuade, Marissa Savage, and Aubrenesia Tomlinson

## Slide Outline

### Slide 1 — ML DeployMate
- Introduce the team and project.
- Explain that DeployMate detects data drift and explains alerts
  in plain English.

### Slide 2 — The Problem
- Incoming data can change after a model is deployed.
- The model may continue producing predictions without recognizing
  the change.
- Correct outcomes may arrive too late to use accuracy as an early warning.
- Introduce data drift as a reason to monitor incoming data.

### Slide 3 — Our Solution
- Process incoming data in batches.
- Compare each of the 12 input features with training reference data.
- Raise an alert when a drift threshold is crossed.
- Generate an explanation describing the change, possible causes,
  and suggested action.
- Introduce the Streamlit dashboard and FastAPI interface.

### Slide 4 — Data and Model
- Introduce the UCI Adult census dataset with 48,842 records.
- Explain the 12 retained features: five numeric and seven categorical.
- Explain the task: predicting whether income exceeds $50,000.
- Describe the class imbalance: approximately 24% earn above $50,000.
- Explain model comparison using five-fold cross-validation.
- Present the selected model's held-out test results:
  - Accuracy: 87.3%
  - ROC-AUC: 0.928
  - F1: 0.707
  - Precision: 77.3%
  - Recall: 65.2%
- Explain that similar training and test scores suggest limited
  overfitting; they do not prove its absence.

### Slide 5 — Drift Detection
- Explain Population Stability Index for numeric features.
- State the project's numeric drift threshold: PSI >= 0.20.
- Explain chi-squared testing and Cramér's V for categorical features.
- State the categorical drift threshold: V >= 0.10.
- Explain why rare categories are grouped into Other.
- Explain why effect sizes are used for alert decisions.
- Clarify that Isolation Forest and z-scores provide context rather
  than independently raising alerts.

### Slide 6 — Live Demo
- Demonstrate the sudden_shift scenario.
- Explain the injected increase in hours worked per week.
- Show where the change begins and when the alert appears.
- Compare drift scores before and after the change.
- Explain that input drift may be visible even when accuracy changes little.
- Use the batch number and values from the actual run; the slide's
  example shows the change beginning at batch 16.

### Slide 7 — AI Explanations
- Explain the three output sections:
  - What happened.
  - Likely cause.
  - Suggested action.
- Explain that only summary statistics are sent to the AI provider.
- Clarify that possible causes are suggestions, not confirmed diagnoses.
- Explain the template fallback when AI requests are unavailable.

### Slide 8 — Results
- Present results from 2,400 simulated batches.
- Report 100% precision and approximately 93% recall.
- Explain that 1,006 of 1,080 modified batches were detected.
- Report zero false alarms in 1,320 healthy batches.
- Explain that all 50 modified runs were detected.
- Distinguish detecting every run from detecting every modified batch.
- Explain the slower detection of gradual drift.
- Describe the ablation result: allowing anomaly indicators to raise
  alerts added eight false alarms for one additional detected batch.

### Slide 9 — Limitations and Next Steps
- Explain that testing used simulated changes on one dataset.
- Explain that input drift tests cannot detect every form of concept drift.
- Describe the delay in detecting gradual changes.
- Acknowledge sensitive attributes and missing fairness monitoring.
- Discuss future work:
  - Delayed-label concept-drift checks.
  - Fairness monitoring.
  - Adaptive thresholds.
  - Persistent alert history.
  - Email or Slack notifications.

### Slide 10 — Conclusion and Questions
- Summarize the results within the evaluated simulation.
- Emphasize that every modified run was detected and no healthy
  batches triggered false alarms in that evaluation.
- Share the application and repository links.
- Invite questions.