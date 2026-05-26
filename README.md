# PD Weibull Modelling Framework

## Overview

This project implements a modular Python framework for the simulation, calibration, and visualisation of lifetime Probability of Default (PD) term structures using Weibull-based survival models.

The framework is designed to emulate a simplified banking risk analytics workflow, including:
- portfolio data ingestion,
- preprocessing pipelines,
- survival curve generation,
- stress scenario analysis,
- and automated visualisation outputs.

The project uses fully synthetic data for demonstration purposes only.

---

## Key Features

- Weibull-based lifetime PD modelling
- Modular risk analytics pipeline
- Synthetic portfolio data integration
- Portfolio segmentation
- Scenario and stress testing capabilities
- Automated chart generation
- Reproducible Python workflow
- Configurable architecture

---

## Repository Structure

```text
src/
├── ingestion/
├── preprocessing/
├── modelling/
├── visualization/

data/
├── raw/
├── processed/

outputs/
├── charts/
├── reports/



:contentReference[oaicite:1]{index=1}


```markdown id="esjlwm"
Where:
- \(k\) represents the shape parameter
- \(\lambda\) represents the scale parameter
- \(t\) represents the time horizon

The model can be extended to:
- stress testing applications,
- macroeconomic overlays,
- portfolio segmentation,
- and IFRS9-style lifetime PD estimation.

---

## Example Workflow

1. Load portfolio data
2. Apply preprocessing and validation
3. Generate Weibull survival curves
4. Apply stress scenarios
5. Export charts and reports

---

## Technologies Used

- Python
- Pandas
- NumPy
- SciPy
- Plotly
- Matplotlib

---

## Future Enhancements

- Macroeconomic scenario integration
- API-based data ingestion
- Interactive dashboards
- SQL integration
- Automated reporting pipelines

---

## Disclaimer

This repository is intended exclusively for educational and demonstration purposes.  
All data contained in the repository are synthetic and do not represent real banking portfolios or confidential information.
