# Churn Guardian — Customer Retention & Business Intelligence

A Streamlit analytics application for six supported industries. Upload Previous Year (PY) and Assessment Year (AY) datasets once on **PY vs AY Comparison**; all seven workspaces use the shared data.

## Supported industries
- Banking & Finance
- Healthcare
- Education
- Hospitality
- Telecommunications
- SaaS & Technology

Retail and E-commerce are intentionally excluded.

## Industry context
Churn Guardian uses schema signals to **auto-detect the likely industry** when uploaded data has clear domain-specific columns. You can override the detected industry from the sidebar. Ambiguous schemas do not force a confident guess.

## Workspaces
1. Executive Dashboard
2. PY vs AY Comparison
3. Advanced Visualizer
4. Data Quality & Ingestion
5. ML Model Evaluation
6. Stakeholder Report
7. Business Assistant

## Automatic analytical modes
Churn Guardian selects its analytical mode from dataset evidence—not industry name alone:

1. **Actual target / supervised ML** — A verified, interpretable binary outcome is detected. Decision Tree, Random Forest and XGBoost (when installed) are evaluated against the observed target.
2. **Derived inactivity proxy / rule-based ML** — When no usable target exists but an inactivity-days field or recognized last-activity date supports a threshold rule with both classes, the app creates a clearly labelled proxy. The source inactivity field/date is excluded from model inputs. Proxy scores measure agreement with the rule, not verified churn.
3. **Exploratory analytics** — When neither a usable target nor a defensible proxy exists, the app continues with descriptive analytics, distributions, segmentation, correlation and data exploration. It does not fabricate supervised metrics.

PY vs AY movement is a separate workflow. A shared identity key indicates presence in both extracts; absence from AY is not confirmed churn. Year-coded/generated IDs with zero overlap are flagged, and movement/retention categories are suppressed when matching is not comparable. Aggregate year-to-year measure comparisons may still be available.

## Models and metrics
- Decision Tree
- Random Forest
- XGBoost (shown when available)

For supervised modes, metrics include Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix, ROC/Precision–Recall curves, feature importance and held-out test-set predictions. Recall is emphasized when comparing models, but consider precision and outreach capacity too. Proxy metrics are explicitly caveated.

## Run locally
1. Install Python 3.11 or 3.12 (recommended for broad package compatibility).
2. Open the project folder in VS Code.
3. (Optional) create and activate a virtual environment.
4. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
5. Start the app:
   ```bash
   streamlit run app.py
   ```
6. Upload PY and AY files on **PY vs AY Comparison**. The industry is inferred when schema evidence is strong; you can change it in the sidebar.

## Deploy on Streamlit Community Cloud
1. Push the project files (`app.py`, `requirements.txt`, `logo.png`, `README.md`, and `.streamlit/config.toml`) to a GitHub repository.
2. In Streamlit Community Cloud, create a new app and select that repository, branch, and `app.py` as the main file.
3. Choose Python 3.11 or 3.12 if the cloud configuration offers it.
4. Deploy and wait for dependencies to finish installing.

Do not commit private customer datasets, credentials, API keys, or health information to a public repository. Use synthetic/anonymized demo data for public demos.

## Data notes
- Identity matching prefers common stable IDs, then shared email/phone; cross-named IDs are a last resort and should be reviewed.
- Transaction/order/purchase/invoice/row IDs are not treated as customer identity.
- Generic multi-class statuses are not automatically treated as churn targets; ambiguous labels require review.
- Column recognition uses common aliases; confirm detected fields and definitions with the source-data owner.
- Healthcare data requires appropriate authorization, privacy controls and data minimization.

## Dashboard themes
- Executive Dashboard — Electric Cyan / Ice Blue / Deep Navy
- PY vs AY Comparison — Indigo / Violet
- Advanced Visualizer — Teal / Aqua
- Data Quality & Ingestion — Emerald / Neon Lime
- ML Model Evaluation — Magenta / Orchid
- Stakeholder Report — Rose / Pink
- Business Assistant — Gold / Amber

## Important limitations
- The built-in Business Assistant is a context-aware, rule-based analytical helper; it is not an external generative-AI chatbot.
- Risk bands are illustrative and not calibrated for operational deployment.
- Predictions, proxy labels, descriptive associations and feature importance are not causal explanations. Validate findings before business or clinical decisions.
- A PY customer missing from AY may be inactive, migrated, omitted or represented under a changed identifier; do not interpret absence as confirmed churn without validation.
