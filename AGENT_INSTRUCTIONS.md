Agent Instructions — MonteSight: 1-Year Monte Carlo Stock Forecaster

These instructions define exactly how you (the Codex agent) must read and execute the architecture, PRD, data model, and pipeline spec to generate the full application codebase.

You must follow these instructions strictly.

1. Mission

Your mission is to generate all Python source files required for the project MonteSight, a Streamlit app that performs 1-year Monte Carlo stock price forecasting.

You will turn the Markdown specifications in this repo into a fully functional codebase.

2. Inputs You Must Use

You must read and comply with:

PRD.md

DATA_MODEL.md

PIPELINE_SPEC.md

ARCHITECTURE.md (if provided)

Any additional Markdown files in the project root

All code must strictly follow the data models, function signatures, and pipelines defined there.

3. Output Requirements

You must generate ALL of the following:

3.1 Python Source Files

As defined in the architecture:

/app.py
/config/settings.py
/core/data_loader.py
/core/simulation.py
/core/analytics.py
/core/utils.py
/ui/layout.py
/ui/components.py
/ui/plots.py
/tests/...

3.2 Requirements File

requirements.txt — containing exact imports used.

3.3 README

Human-readable quickstart for running the app.

3.4 Clean, Production-Ready Code

Idiomatic Python

Docstrings for every function

No unused imports

Stable code that runs without modification

4. Coding Rules & Standards

The following rules must be followed in all generated code.

4.1 Modular Design

Follow the module boundaries specified in ARCHITECTURE.md:

Data loading logic must stay in core/data_loader.py

Simulation logic must stay in core/simulation.py

Analytics pipeline must stay in core/analytics.py

UI rendering must stay in ui/*

No cross-dependencies except as defined in the pipeline spec

4.2 Function Requirements

Every function must:

Include a complete docstring

Accept and return exactly the parameters listed in the Data Model

Never implicitly mutate input objects

Return pure results (no side effects unless UI)

4.3 Error Handling

You MUST implement:

Ticker not found errors

Not enough historical data errors

Any failed numpy/pandas operations must raise helpful exceptions

UI must present user-friendly messages

4.4 Performance

All simulation code must run efficiently:

Use vectorized numpy operations only

No loops over simulations

No repeated downloads of data (cache where appropriate)

5. Implementation Instructions

Codex, follow this order:

5.1 Step 1 — Create Folder Structure

Automatically create all directories as defined in the architecture.

5.2 Step 2 — Implement Core Modules

Implement in this exact order:

core/utils.py

core/data_loader.py

core/simulation.py

core/analytics.py

Follow the function signatures from DATA_MODEL.md and the logic from PIPELINE_SPEC.md.

5.3 Step 3 — Implement UI Layer

Implement:

ui/layout.py

ui/components.py

ui/plots.py

UI must:

Use Streamlit

Have a sidebar control panel

Display plots using matplotlib or plotly

Show probability table (filtered ≥ 66%)

Show percentile band explanation in plain language

Ensure all UI functions call the correct modules in the pipeline.

5.4 Step 4 — Build app.py

app.py must:

Set Streamlit page config

Render header + sidebar

Execute pipeline

Render outputs in main area

Handle errors gracefully

Follow the pseudo-code from PIPELINE_SPEC.md.

5.5 Step 5 — Create Tests

Write simple tests validating:

Drift & volatility estimation works

Simulation outputs correct shapes

Probability table sums are correct

Price grid generation works

Place tests inside /tests.

5.6 Step 6 — Generate Requirements File

requirements.txt must include:

streamlit
yfinance
pandas
numpy
matplotlib
plotly

5.7 Step 7 — Create README

README must include:

Project overview

Setup instructions

How to run the app

How simulation works

Example screenshots (optional placeholder)

6. Rules for Code Generation

Do NOT hallucinate new architectures

Do NOT omit any module

Always reference the PRD, Data Model, and Pipeline Specs

Ensure consistent naming across all modules

All paths and imports must be valid

Use only documented dependencies

7. Completion Criteria

Your work is complete when:

The full folder structure is created

All Python modules are fully implemented

The Streamlit app runs end-to-end with no errors

Probability table, plots, and summaries all render correctly

README + requirements.txt are present

Tests run without failure

8. Additional Behavior

If a spec is ambiguous, follow this rule:

Choose the simplest, most maintainable implementation that fits the PRD and pipeline.

If two modules conflict:

The Pipeline Spec takes priority.
