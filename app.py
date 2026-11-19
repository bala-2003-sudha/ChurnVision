
import io
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

# =============================================================================
# CHURN GUARDIAN | SINGLE-FILE BUSINESS ANALYTICS SUITE
# =============================================================================

st.set_page_config(
    page_title="Churn Guardian | Enterprise Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------- THEME ---------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background: #08111f; color: #e5edf7; }
[data-testid="stSidebar"] { background: #0d192b; border-right: 1px solid #20324a; }
[data-testid="stSidebar"] * { color: #e5edf7; }
h1, h2, h3 { color: #f8fafc; }
div[data-testid="stMetric"] {
    background: linear-gradient(145deg,#12243a,#102033);
    border: 1px solid #25415d; padding: 16px; border-radius: 14px;
}
div[data-testid="stMetric"] label { color: #9fb3c8 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #eaf5ff; }
.block-container { padding-top: 1.5rem; }
div[data-testid="stDataFrame"] { border: 1px solid #263b54; border-radius: 10px; }
div.stButton > button, div.stDownloadButton > button {
    border-radius: 9px; border: 1px solid #2c7894; font-weight: 650;
}
.small-note { color: #9fb3c8; font-size: .88rem; }
.status-pill { padding: 7px 12px; border-radius: 20px; display: inline-block;
    background: #123c35; color: #86efac; border: 1px solid #23785f; font-size: 13px; }
.warning-pill { padding: 7px 12px; border-radius: 20px; display: inline-block;
    background: #49351a; color: #fcd34d; border: 1px solid #9a6b25; font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# ----------------------------- UTILITIES -------------------------------------

def read_uploaded_file(file):
    """Read CSV or Excel safely and return a dataframe."""
    if file is None:
        return None

    try:
        file.seek(0)

        if file.name.lower().endswith(".csv"):
            try:
                return pd.read_csv(file)
            except UnicodeDecodeError:
                file.seek(0)
                return pd.read_csv(file, encoding="latin-1")

        file.seek(0)
        return pd.read_excel(file)

    except Exception as exc:
        st.error(f"Could not read **{file.name}**: {exc}")
        return None


def clean_column_names(df):
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def find_column(df, candidates):
    """Case-insensitive exact or normalized column lookup."""
    lookup = {
        re.sub(r"[^a-z0-9]", "", str(c).lower()): c
        for c in df.columns
    }

    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())

        if key in lookup:
            return lookup[key]

    return None


def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df):
    return df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()


def money(value):
    if pd.isna(value):
        return "N/A"

    return f"{value:,.2f}"


def safe_percent(numerator, denominator):
    return (numerator / denominator * 100) if denominator else 0.0


def normalize_id(series):
    return series.astype(str).str.strip().str.replace(
        r"\\.0$", "", regex=True
    )


@st.cache_data(show_spinner=False)
def aggregate_hierarchy(data, path_columns, value_column=None, max_segments=150):
    """Aggregate records before building a Sunburst; never plot raw customer rows."""

    selected = data[list(path_columns)].copy()

    if value_column and value_column in data.columns:
        selected["_chart_value"] = pd.to_numeric(
            data[value_column], errors="coerce"
        ).fillna(0).values

        grouped = (
            selected.groupby(
                list(path_columns),
                dropna=False,
                observed=True
            )["_chart_value"]
            .sum()
            .reset_index(name="Chart Value")
        )

    else:
        grouped = (
            selected.groupby(
                list(path_columns),
                dropna=False,
                observed=True
            )
            .size()
            .reset_index(name="Chart Value")
        )

    return grouped.sort_values(
        "Chart Value",
        ascending=False
    ).head(max_segments)


def add_download_button(label, data, filename, mime):
    st.download_button(
        label,
        data=data,
        file_name=filename,
        mime=mime
    )


def make_pdf_report(report_data):
    """Create a stakeholder PDF in memory using ReportLab."""

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import (
            getSampleStyleSheet,
            ParagraphStyle
        )
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle
        )

    except ImportError:
        return None

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="Subtle",
            parent=styles["BodyText"],
            textColor=colors.HexColor("#475569"),
            leading=15
        )
    )

    story = [
        Paragraph("CHURN GUARDIAN", styles["Title"]),
        Paragraph(
            "Business Performance & Customer Retention Report",
            styles["Heading2"]
        ),
        Paragraph(
            f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            styles["Subtle"]
        ),
        Spacer(1, 8 * mm),
        Paragraph("Executive summary", styles["Heading2"]),
        Paragraph(report_data["summary"], styles["BodyText"]),
        Spacer(1, 4 * mm),
        Paragraph("Key indicators", styles["Heading2"])
    ]

    table_data = [
        ["Indicator", "Previous year", "Assessment year", "Change"]
    ]

    for row in report_data["table"]:
        table_data.append([str(v) for v in row])

    tbl = Table(
        table_data,
        colWidths=[55 * mm, 37 * mm, 37 * mm, 37 * mm],
        repeatRows=1
    )

    tbl.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12304a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#cbd5e1")),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#0f172a")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story += [
        tbl,
        Spacer(1, 6 * mm),
        Paragraph("What the data indicates", styles["Heading2"])
    ]

    for item in report_data["observations"]:
        story.append(
            Paragraph("• " + item, styles["BodyText"])
        )
        story.append(Spacer(1, 2 * mm))

    story += [
        Spacer(1, 3 * mm),
        Paragraph("Recommended actions", styles["Heading2"])
    ]

    for item in report_data["actions"]:
        story.append(
            Paragraph("• " + item, styles["BodyText"])
        )
        story.append(Spacer(1, 2 * mm))

    story += [
        Spacer(1, 5 * mm),
        Paragraph("Methodology and limitations", styles["Heading2"]),
        Paragraph(report_data["limitations"], styles["Subtle"]),
        Spacer(1, 4 * mm),
        Paragraph(
            "This report is generated from the files uploaded by the user. "
            "Validate findings with business owners before operational decisions.",
            styles["Subtle"]
        )
    ]

    doc.build(story)
    buffer.seek(0)

    return buffer.getvalue()


# ----------------------------- SIDEBAR ---------------------------------------

with st.sidebar:

    try:
        st.image("logo.png", use_container_width=True)

    except Exception:
        st.markdown("## 🛡️ CHURN GUARDIAN")
        st.caption("PREDICT. RETAIN. GROW.")

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Executive Dashboard",
            "PY vs AY Comparison",
            "Advanced Visualizer",
            "Data Quality & Ingestion",
            "ML Model Evaluation",
            "Stakeholder Report",
            "Business Assistant"
        ]
    )

    st.markdown("---")
    st.subheader("Analysis settings")

    inactivity_days = st.number_input(
        "Inactivity threshold (days)",
        min_value=30,
        max_value=365,
        value=90,
        step=15,
        help="Used only for an explicitly labelled inactivity-risk proxy when an inactivity-days column exists."
    )

    st.caption(
        "Upload PY and AY files once on the PY vs AY Comparison page."
    )


# ------------------------ CENTRAL DATA UPLOAD -------------------------------

# The ONLY upload controls in the application are on the PY vs AY Comparison page.

if "py_data" not in st.session_state:
    st.session_state.py_data = None

if "ay_data" not in st.session_state:
    st.session_state.ay_data = None

if "py_filename" not in st.session_state:
    st.session_state.py_filename = ""

if "ay_filename" not in st.session_state:
    st.session_state.ay_filename = ""


if page == "PY vs AY Comparison":

    st.title("📂 Upload Previous Year and Assessment Year")

    st.write(
        "Upload both files here once. Every other page will automatically use these datasets."
    )

    upload_col1, upload_col2 = st.columns(2)

    with upload_col1:
        uploaded_py = st.file_uploader(
            "Previous Year (PY) — e.g. 2024",
            type=["csv", "xlsx"],
            key="central_py_upload"
        )

    with upload_col2:
        uploaded_ay = st.file_uploader(
            "Assessment Year (AY) — e.g. 2025",
            type=["csv", "xlsx"],
            key="central_ay_upload"
        )

    load_col, clear_col = st.columns(2)

    with load_col:
        load_clicked = st.button(
            "Load & Analyse Both Files",
            type="primary",
            use_container_width=True
        )

    with clear_col:
        clear_clicked = st.button(
            "Clear Uploaded Datasets",
            use_container_width=True
        )

    if load_clicked:

        if uploaded_py is None or uploaded_ay is None:
            st.error(
                "Please select both the PY file and the AY file before loading."
            )

        else:
            new_py = read_uploaded_file(uploaded_py)
            new_ay = read_uploaded_file(uploaded_ay)

            if new_py is not None and new_ay is not None:

                st.session_state.py_data = (
                    clean_column_names(new_py)
                    .drop_duplicates()
                    .reset_index(drop=True)
                )

                st.session_state.ay_data = (
                    clean_column_names(new_ay)
                    .drop_duplicates()
                    .reset_index(drop=True)
                )

                st.session_state.py_filename = uploaded_py.name
                st.session_state.ay_filename = uploaded_ay.name

                st.success(
                    "Both datasets were loaded. All seven pages can now use the same data."
                )

                st.rerun()


    if clear_clicked:
        st.session_state.py_data = None
        st.session_state.ay_data = None
        st.session_state.py_filename = ""
        st.session_state.ay_filename = ""
        st.rerun()

    if st.session_state.py_data is not None and st.session_state.ay_data is not None:
        st.success(
            f"Loaded: PY — {st.session_state.py_filename} "
            f"({len(st.session_state.py_data):,} rows) | "
            f"AY — {st.session_state.ay_filename} "
            f"({len(st.session_state.ay_data):,} rows)"
        )


# ----------------------------- LOAD DATA -------------------------------------

py_df = st.session_state.py_data
ay_df = st.session_state.ay_data

if py_df is None and ay_df is None:

    st.title("🛡️ Churn Guardian")
    st.markdown("### Enterprise customer analytics and retention intelligence")

    st.info(
        "Open **PY vs AY Comparison** from the sidebar and upload "
        "your PY and AY Excel/CSV files to begin."
    )

    st.markdown("""
    **Your seven connected workspaces**

    - Executive Dashboard — management KPIs and business performance.
    - PY vs AY Comparison — the only place to upload the two yearly files.
    - Advanced Visualizer — interactive charts using the uploaded datasets.
    - Data Quality & Ingestion — missing values, duplicates and schema checks.
    - ML Model Evaluation — Decision Tree, Random Forest and XGBoost.
    - Stakeholder Report — evidence-based findings and downloadable reports.
    - Business Assistant — answers about the currently loaded data.
    """)

    st.stop()


if py_df is not None:
    py_df = clean_column_names(py_df)

if ay_df is not None:
    ay_df = clean_column_names(ay_df)


# AY is the main analysis dataset; PY is used for year-over-year comparison.

main_df = ay_df if ay_df is not None else py_df

main_label = (
    "Assessment Year (AY)"
    if ay_df is not None
    else "Previous Year (PY)"
)

main_df = (
    main_df.copy()
    .drop_duplicates()
    .reset_index(drop=True)
)


# Standardize known dataset column variants.

id_col = find_column(
    main_df,
    ["CustomerID", "Customer_ID", "ClientID", "AccountID", "UserID"]
)

py_id_col = (
    find_column(
        py_df,
        ["CustomerID", "Customer_ID", "ClientID", "AccountID", "UserID"]
    )
    if py_df is not None else None
)

ay_id_col = (
    find_column(
        ay_df,
        ["CustomerID", "Customer_ID", "ClientID", "AccountID", "UserID"]
    )
    if ay_df is not None else None
)


# Detect business measures by common aliases.

spend_col = find_column(
    main_df,
    ["TotalSpend", "Total_Spend", "Revenue", "Sales", "TotalRevenue", "Amount"]
)

orders_col = find_column(
    main_df,
    ["OrderCount", "Orders", "NumberOfOrders", "PurchaseCount"]
)

aov_col = find_column(
    main_df,
    ["AverageOrderValue", "AOV", "AvgOrderValue"]
)

satisfaction_col = find_column(
    main_df,
    ["SatisfactionScore", "CustomerSatisfaction", "CSAT"]
)

region_col = find_column(
    main_df,
    ["Region", "Location", "Market", "Territory"]
)

category_col = find_column(
    main_df,
    ["PreferredCategory", "Category", "ProductCategory"]
)

days_col = find_column(
    main_df,
    ["DaysSinceLastPurchase", "DaysInactive", "RecencyDays", "DaysSinceLastOrder"]
)

target_col = find_column(
    main_df,
    ["Churn", "Churned", "Attrition", "Exited", "Target", "Label", "Status"]
)


# --------------------------- PY / AY MATCHING --------------------------------

comparison = None

if py_df is not None and ay_df is not None:

    if py_id_col and ay_id_col:

        py_work = py_df.copy()
        ay_work = ay_df.copy()

        py_work["_match_id"] = normalize_id(py_work[py_id_col])
        ay_work["_match_id"] = normalize_id(ay_work[ay_id_col])

        py_ids = set(py_work["_match_id"].dropna())
        ay_ids = set(ay_work["_match_id"].dropna())

        common_ids = py_ids & ay_ids
        lost_ids = py_ids - ay_ids
        new_ids = ay_ids - py_ids

        comparison = {
            "py_count": len(py_ids),
            "ay_count": len(ay_ids),
            "retained": len(common_ids),
            "not_observed": len(lost_ids),
            "new": len(new_ids),
            "py_ids": py_ids,
            "ay_ids": ay_ids,
            "common_ids": common_ids,
            "lost_ids": lost_ids,
            "new_ids": new_ids,
            "py_work": py_work,
            "ay_work": ay_work
        }

        comparison["retention_rate"] = safe_percent(
            len(common_ids),
            len(py_ids)
        )

        comparison["not_observed_rate"] = safe_percent(
            len(lost_ids),
            len(py_ids)
        )

        comparison["new_customer_share"] = safe_percent(
            len(new_ids),
            len(ay_ids)
        )

    else:
        comparison = {
            "error": (
                "Both files must contain a customer identifier column "
                "(for example CustomerID) to calculate customer-level retention."
            )
        }


# --------------------------- MODEL PREPARATION -------------------------------

def prepare_supervised_data(df, threshold):
    """
    Prepare a defensible model dataset.

    Explicit target columns take priority. If absent, derive a documented
    proxy from inactivity days and exclude the inactivity field itself.
    """

    work = df.copy()

    detected_target = find_column(
        work,
        ["Churn", "Churned", "Attrition", "Exited", "Target", "Label"]
    )

    proxy_used = False
    label_description = None

    if detected_target:

        raw = work[detected_target]

        if raw.nunique(dropna=True) != 2:
            return (
                None, None, None, None,
                "Target column must contain exactly two classes."
            )

        if pd.api.types.is_numeric_dtype(raw):

            y = pd.to_numeric(raw, errors="coerce")

            vals = sorted(y.dropna().unique().tolist())

            if len(vals) != 2:
                return (
                    None, None, None, None,
                    "Target must have exactly two usable values."
                )

            y = y.map({
                vals[0]: 0,
                vals[1]: 1
            })

        else:

            text = raw.astype(str).str.strip().str.lower()

            positive = {
                "yes", "true", "1", "y",
                "churn", "churned", "exited",
                "inactive", "attrited"
            }

            negative = {
                "no", "false", "0", "n",
                "retained", "active", "stay", "stayed"
            }

            mapped = text.map(
                lambda v: 1 if v in positive
                else (0 if v in negative else np.nan)
            )

            if mapped.notna().sum() < len(work) * .8:

                le = LabelEncoder()

                encoded = pd.Series(
                    le.fit_transform(raw.astype(str)),
                    index=work.index
                )

                y = encoded

            else:
                y = mapped

        excluded = {detected_target}

        label_description = (
            f"Observed target column: {detected_target}"
        )

    else:

        inactivity_col = find_column(
            work,
            [
                "DaysSinceLastPurchase",
                "DaysInactive",
                "RecencyDays",
                "DaysSinceLastOrder"
            ]
        )

        if not inactivity_col:
            return (
                None, None, None, None,
                "No binary target or inactivity-days field was found. "
                "Model accuracy cannot be evaluated honestly."
            )

        inactivity = pd.to_numeric(
            work[inactivity_col],
            errors="coerce"
        )

        y = (inactivity >= threshold).astype(float)
        y[inactivity.isna()] = np.nan

        excluded = {inactivity_col}
        proxy_used = True

        label_description = (
            f"Proxy label: inactivity >= {threshold} days; "
            "the inactivity field is excluded from model inputs."
        )

    # Exclude IDs, free-text identifiers and target from features.

    for col in work.columns:

        normalized = re.sub(
            r"[^a-z0-9]",
            "",
            str(col).lower()
        )

        if normalized in {
            "customerid",
            "clientid",
            "accountid",
            "userid",
            "id"
        }:
            excluded.add(col)

    X = work.drop(
        columns=list(excluded),
        errors="ignore"
    ).copy()

    X = X.dropna(axis=1, how="all")

    for col in X.columns:

        if X[col].nunique(dropna=True) > max(
            1000,
            int(len(X) * .95)
        ):
            X = X.drop(columns=[col])

    valid = y.notna()

    X = X.loc[valid].copy()
    y = y.loc[valid].astype(int)

    if len(X) < 30 or y.nunique() < 2:
        return (
            None, None, None, None,
            "At least 30 usable rows and two target classes are required."
        )

    if y.value_counts().min() < 2:
        return (
            None, None, None, None,
            "Each target class needs at least two records for a train/test split."
        )

    return X, y, proxy_used, label_description, None


@st.cache_resource(show_spinner=False)
def train_models_cached(data_hash, X, y):

    """Train the three existing project algorithms with preprocessing."""

    numeric = X.select_dtypes(include=np.number).columns.tolist()

    categorical = [
        c for c in X.columns
        if c not in numeric
    ]

    transformers = []

    if numeric:

        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric
        ))

    if categorical:

        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=.25,
        random_state=42,
        stratify=y
    )

    model_specs = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=160,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    }

    if XGBOOST_AVAILABLE:

        model_specs["XGBoost"] = XGBClassifier(
            n_estimators=140,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=2
        )

    output = {}

    for name, estimator in model_specs.items():

        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", estimator)
        ])

        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)

        output[name] = {
            "pipeline": pipe,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "cm": confusion_matrix(y_test, pred, labels=[0, 1]),
            "y_test": y_test,
            "pred": pred,
            "X_test": X_test,
            "feature_names": X.columns.tolist()
        }

    return output


model_results = {}
model_note = ""

X_model, y_model, proxy_used, label_description, model_error = (
    prepare_supervised_data(main_df, inactivity_days)
)

if model_error is None:

    try:

        model_results = train_models_cached(
            str(pd.util.hash_pandas_object(
                X_model,
                index=True
            ).sum()),
            X_model,
            y_model
        )

        model_note = label_description

    except Exception as exc:

        model_error = (
            f"Model training could not be completed: {exc}"
        )


# ----------------------------- REPORT DATA -----------------------------------

report_summary = (
    "Upload both PY and AY files with a shared customer identifier "
    "to generate a year-over-year retention summary."
)

report_table = []
observations = []

actions = [
    "Segment customers by purchase recency, order frequency and spend; prioritize high-value customers whose activity is declining.",
    "Investigate low satisfaction scores and repeated support contacts before designing retention campaigns.",
    "Use region, preferred category and membership tier to tailor customer engagement rather than sending one generic offer to everyone.",
    "Track the same KPIs each month and validate campaign outcomes against a holdout group where practical."
]

limitations = (
    "A customer present in PY but absent in AY is classified as 'not observed in AY'. "
    "This does not prove permanent churn. Annual snapshot files cannot establish the exact date or reason a customer stopped purchasing. "
    "If a proxy inactivity target is used for ML, model metrics measure agreement with that business rule, not verified real-world churn."
)


if isinstance(comparison, dict) and "error" not in comparison:

    c = comparison

    report_summary = (
        f"The PY file contains {c['py_count']:,} distinct customers "
        f"and the AY file contains {c['ay_count']:,}. "
        f"{c['retained']:,} customers appear in both files, "
        f"{c['not_observed']:,} PY customers are not observed in AY, "
        f"and {c['new']:,} customers appear for the first time in AY. "
        f"The PY-to-AY observed retention rate is {c['retention_rate']:.1f}%."
    )

    report_table = [
        (
            "Distinct customers",
            f"{c['py_count']:,}",
            f"{c['ay_count']:,}",
            f"{c['ay_count']-c['py_count']:+,}"
        ),
        (
            "Returning customers",
            "—",
            f"{c['retained']:,}",
            "Matched by customer ID"
        ),
        (
            "Not observed in AY",
            "—",
            f"{c['not_observed']:,}",
            "PY IDs absent from AY"
        ),
        (
            "Newly observed in AY",
            "—",
            f"{c['new']:,}",
            "AY IDs absent from PY"
        ),
        (
            "Observed retention rate",
            "100.0% reference",
            f"{c['retention_rate']:.1f}%",
            "Matched PY customers / PY customers"
        )
    ]

    observations += [
        f"Observed retention from PY to AY is {c['retention_rate']:.1f}%.",

        f"{c['not_observed']:,} customers from PY do not appear in the AY file; validate whether they are inactive, migrated, or missing from the extract.",

        f"{c['new']:,} customer IDs are newly observed in AY."
    ]

    if spend_col and py_df is not None and ay_df is not None:

        py_spend_col = find_column(
            py_df,
            ["TotalSpend", "Total_Spend", "Revenue", "Sales", "TotalRevenue", "Amount"]
        )

        ay_spend_col = find_column(
            ay_df,
            ["TotalSpend", "Total_Spend", "Revenue", "Sales", "TotalRevenue", "Amount"]
        )

        if py_spend_col and ay_spend_col:

            psv = pd.to_numeric(
                py_df[py_spend_col],
                errors="coerce"
            ).sum()

            asv = pd.to_numeric(
                ay_df[ay_spend_col],
                errors="coerce"
            ).sum()

            report_table.append((
                "Total spend / revenue",
                money(psv),
                money(asv),
                money(asv - psv)
            ))

            observations.append(
                f"Total spend/revenue changed from {money(psv)} in PY to {money(asv)} in AY."
            )

else:

    observations.append(
        "A PY-versus-AY customer-level comparison is unavailable until both files with a common customer ID are uploaded."
    )


# =============================================================================
# PAGE 1 — EXECUTIVE DASHBOARD
# =============================================================================

if page == "Executive Dashboard":

    st.title("🛡️ Churn Guardian")
    st.subheader("Executive business intelligence dashboard")

    st.markdown(
        f'<span class="status-pill">Analysis dataset: {main_label}</span>',
        unsafe_allow_html=True
    )

    st.caption(
        "Metrics are calculated from the uploaded data. "
        "Fields not present in your file are not fabricated."
    )

    if isinstance(comparison, dict) and "error" not in comparison:

        c = comparison

        k1, k2, k3, k4, k5 = st.columns(5)

        k1.metric("PY customers", f"{c['py_count']:,}")

        k2.metric(
            "AY customers",
            f"{c['ay_count']:,}",
            f"{c['ay_count']-c['py_count']:+,}"
        )

        k3.metric("Returning", f"{c['retained']:,}")

        k4.metric("Not observed", f"{c['not_observed']:,}")

        k5.metric("New in AY", f"{c['new']:,}")

        st.metric(
            "Observed retention rate",
            f"{c['retention_rate']:.1f}%"
        )

        left, right = st.columns(2)

        with left:

            fig = px.pie(
                names=[
                    "Returning in AY",
                    "PY not observed in AY"
                ],
                values=[
                    c["retained"],
                    c["not_observed"]
                ],
                hole=.58,
                color_discrete_sequence=[
                    "#10b981",
                    "#f97373"
                ],
                title="PY customer movement"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right:

            movement = pd.DataFrame({
                "Customer group": [
                    "Returning",
                    "Not observed",
                    "New in AY"
                ],
                "Customers": [
                    c["retained"],
                    c["not_observed"],
                    c["new"]
                ]
            })

            fig = px.bar(
                movement,
                x="Customer group",
                y="Customers",
                color="Customer group",
                text_auto=True,
                color_discrete_sequence=[
                    "#0ea5e9",
                    "#f97373",
                    "#a78bfa"
                ]
            )

            fig.update_layout(showlegend=False)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.warning(
            comparison["error"]
            if isinstance(comparison, dict) and "error" in comparison
            else "Upload both PY and AY files with a customer ID to see customer movement KPIs."
        )

        st.metric(
            "Rows in current dataset",
            f"{len(main_df):,}"
        )

    st.markdown("---")

    st.subheader("Business performance indicators available in AY")

    cols = st.columns(4)

    metrics = [
        ("Total spend / revenue", spend_col, "sum"),
        ("Total orders", orders_col, "sum"),
        ("Average order value", aov_col, "mean"),
        ("Average satisfaction", satisfaction_col, "mean")
    ]

    for colbox, (label, col, operation) in zip(cols, metrics):

        with colbox:

            if col:

                vals = pd.to_numeric(
                    main_df[col],
                    errors="coerce"
                )

                val = (
                    vals.sum()
                    if operation == "sum"
                    else vals.mean()
                )

                st.metric(
                    label,
                    money(val)
                    if "spend" in label.lower() or "value" in label.lower()
                    else (
                        f"{val:,.1f}"
                        if pd.notna(val)
                        else "N/A"
                    )
                )

            else:

                st.metric(label, "Not available")

    if region_col and spend_col:

        st.subheader("Revenue / spend by region")

        tmp = main_df.copy()

        tmp[spend_col] = pd.to_numeric(
            tmp[spend_col],
            errors="coerce"
        )

        region_data = (
            tmp.groupby(region_col, dropna=False)[spend_col]
            .sum()
            .reset_index()
            .sort_values(spend_col, ascending=False)
        )

        fig = px.bar(
            region_data,
            x=region_col,
            y=spend_col,
            color=spend_col,
            color_continuous_scale="Teal"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if model_results:

        st.markdown(
            f'<span class="warning-pill">ML label basis: {model_note}</span>',
            unsafe_allow_html=True
        )

        st.caption(
            "These are test-set classification metrics. "
            "If a proxy label is used, they are not verified churn accuracy."
        )

        best_name = max(
            model_results,
            key=lambda n: model_results[n]["f1"]
        )

        st.success(
            f"Highest test-set F1 in this run: "
            f"{best_name} ({model_results[best_name]['f1']:.3f})."
        )

    else:

        st.info(
            "ML evaluation is not available for this dataset yet. "
            "See the ML Model Evaluation page for the required fields."
        )


# =============================================================================
# PAGE 2 — PY VS AY COMPARISON
# =============================================================================

elif page == "PY vs AY Comparison":

    st.title("🔄 Previous Year vs Assessment Year")

    st.write(
        "Customer-level comparison is based on matching customer identifiers—not on independently calculated churn percentages."
    )

    if not (
        isinstance(comparison, dict)
        and "error" not in comparison
    ):

        st.warning(
            comparison["error"]
            if isinstance(comparison, dict) and "error" in comparison
            else "Upload both PY and AY files to activate this page."
        )

        st.stop()

    c = comparison

    a, b, d, e = st.columns(4)

    a.metric("PY customer base", f"{c['py_count']:,}")

    b.metric("Returning customers", f"{c['retained']:,}")

    d.metric("Not observed in AY", f"{c['not_observed']:,}")

    e.metric("New in AY", f"{c['new']:,}")

    st.metric(
        "Observed retention rate",
        f"{c['retention_rate']:.2f}%"
    )

    st.caption(
        "Not observed in AY is a data-presence classification, "
        "not proof that the customer permanently churned."
    )

    chart1, chart2 = st.columns(2)

    with chart1:

        fig = go.Figure(data=[
            go.Bar(
                name="PY",
                x=["Customer base"],
                y=[c["py_count"]],
                marker_color="#64748b"
            ),
            go.Bar(
                name="AY",
                x=["Customer base"],
                y=[c["ay_count"]],
                marker_color="#0d9488"
            )
        ])

        fig.update_layout(
            barmode="group",
            title="Customer base size"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with chart2:

        fig = px.funnel(
            pd.DataFrame({
                "Stage": [
                    "PY customers",
                    "Returning in AY"
                ],
                "Customers": [
                    c["py_count"],
                    c["retained"]
                ]
            }),
            x="Customers",
            y="Stage",
            title="PY customer retention funnel"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Movement summary")

    movement_df = pd.DataFrame([
        [
            "Returning customers",
            c["retained"],
            "Customer ID exists in both years"
        ],
        [
            "Not observed in AY",
            c["not_observed"],
            "Customer ID exists in PY but not AY"
        ],
        [
            "Newly observed in AY",
            c["new"],
            "Customer ID exists in AY but not PY"
        ]
    ], columns=[
        "Segment",
        "Customer count",
        "Definition"
    ])

    st.dataframe(
        movement_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Optional measure comparison")

    common_measures = []

    for candidates, label in [
        (
            ["TotalSpend", "Total_Spend", "Revenue", "Sales", "TotalRevenue", "Amount"],
            "Spend / Revenue"
        ),
        (
            ["OrderCount", "Orders", "NumberOfOrders", "PurchaseCount"],
            "Order count"
        ),
        (
            ["AverageOrderValue", "AOV", "AvgOrderValue"],
            "Average order value"
        ),
        (
            ["SatisfactionScore", "CustomerSatisfaction", "CSAT"],
            "Satisfaction"
        )
    ]:

        pcol = find_column(py_df, candidates)
        acol = find_column(ay_df, candidates)

        if pcol and acol:

            pv = (
                pd.to_numeric(py_df[pcol], errors="coerce").mean()
                if label != "Spend / Revenue"
                else pd.to_numeric(py_df[pcol], errors="coerce").sum()
            )

            av = (
                pd.to_numeric(ay_df[acol], errors="coerce").mean()
                if label != "Spend / Revenue"
                else pd.to_numeric(ay_df[acol], errors="coerce").sum()
            )

            common_measures.append([
                label,
                pv,
                av,
                av - pv
            ])

    if common_measures:

        measures_df = pd.DataFrame(
            common_measures,
            columns=["Measure", "PY", "AY", "Change"]
        )

        st.dataframe(
            measures_df.style.format({
                "PY": "{:,.2f}",
                "AY": "{:,.2f}",
                "Change": "{:+,.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            measures_df,
            x="Measure",
            y=["PY", "AY"],
            barmode="group",
            title="PY vs AY business measures"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "No matching numeric business measures were detected in both files."
        )

    with st.expander("View customer ID lists"):

        tab1, tab2, tab3 = st.tabs([
            "Returning",
            "Not observed",
            "New in AY"
        ])

        with tab1:
            st.dataframe(
                pd.DataFrame({
                    "CustomerID": sorted(c["common_ids"])
                }),
                use_container_width=True
            )

        with tab2:
            st.dataframe(
                pd.DataFrame({
                    "CustomerID": sorted(c["lost_ids"])
                }),
                use_container_width=True
            )

        with tab3:
            st.dataframe(
                pd.DataFrame({
                    "CustomerID": sorted(c["new_ids"])
                }),
                use_container_width=True
            )


# =============================================================================
# PAGE 3 — ADVANCED VISUALIZER
# =============================================================================

elif page == "Advanced Visualizer":

    st.title("📈 Advanced Business Visualizer")

    st.caption(
        "Choose columns from the uploaded dataset to create interactive charts."
    )

    # Exclude identifier-like and very high-cardinality columns.
    max_category_values = min(
        100,
        max(20, int(len(main_df) * 0.10))
    )

    cats = [
        col for col in categorical_columns(main_df)
        if col != id_col
        and main_df[col].nunique(dropna=True) <= max_category_values
    ]

    nums = numeric_columns(main_df)

    if not nums:

        st.warning(
            "No numeric columns are available for numeric visualizations."
        )

    else:

        c1, c2 = st.columns(2)

        with c1:

            st.subheader("Distribution")

            metric = st.selectbox(
                "Numeric metric",
                nums,
                key="dist_metric"
            )

            fig = px.histogram(
                main_df,
                x=metric,
                nbins=30,
                marginal="box",
                color_discrete_sequence=["#0ea5e9"]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with c2:

            st.subheader("Relationship")

            x_metric = st.selectbox(
                "X-axis",
                nums,
                index=0,
                key="scatter_x"
            )

            y_metric = st.selectbox(
                "Y-axis",
                nums,
                index=min(1, len(nums) - 1),
                key="scatter_y"
            )

            color_by = st.selectbox(
                "Color by (optional)",
                ["None"] + cats,
                key="scatter_color"
            )

            scatter_df = main_df

            if len(scatter_df) > 3000:

                scatter_df = scatter_df.sample(
                    3000,
                    random_state=42
                )

                st.caption(
                    "Scatter plot uses a reproducible 3,000-row sample for browser performance."
                )

            fig = px.scatter(
                scatter_df,
                x=x_metric,
                y=y_metric,
                color=None if color_by == "None" else color_by,
                opacity=0.7,
                hover_data=[id_col]
                if id_col and id_col in scatter_df.columns
                else None
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    if cats and nums:

        st.markdown("---")
        st.subheader("Category performance")

        cat = st.selectbox(
            "Group category",
            cats,
            key="group_cat"
        )

        val = st.selectbox(
            "Measure",
            nums,
            key="group_num"
        )

        agg = st.selectbox(
            "Aggregation",
            ["Sum", "Mean", "Median", "Count"],
            key="group_agg"
        )

        tmp = main_df.copy()

        tmp[val] = pd.to_numeric(
            tmp[val],
            errors="coerce"
        )

        grouped = tmp.groupby(
            cat,
            dropna=False
        )[val]

        result = (
            grouped.sum()
            if agg == "Sum"
            else grouped.mean()
            if agg == "Mean"
            else grouped.median()
            if agg == "Median"
            else grouped.count()
        )

        plot_df = result.reset_index(name=agg)

        fig = px.bar(
            plot_df,
            x=cat,
            y=agg,
            color=agg,
            color_continuous_scale="Tealgrn",
            text_auto=".2s"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            plot_df.sort_values(agg, ascending=False),
            use_container_width=True,
            hide_index=True
        )

    # ----------------------- OPTIMIZED SUNBURST ------------------------------

    if len(cats) >= 2:

        st.markdown("---")
        st.subheader("Customer segment hierarchy")

        st.caption(
            "Performance-safe Sunburst: records are aggregated before plotting. "
            "Customer IDs and very high-cardinality fields are excluded."
        )

        default_path = cats[:min(3, len(cats))]

        path = st.multiselect(
            "Choose 2 or 3 category levels",
            cats,
            default=default_path,
            max_selections=3,
            key="sunburst_path"
        )

        sun_value_options = ["Customer count"] + nums

        default_value_index = (
            sun_value_options.index(spend_col)
            if spend_col in nums
            else 0
        )

        sun_value = st.selectbox(
            "Sunburst measure",
            sun_value_options,
            index=default_value_index,
            key="sunburst_measure"
        )

        if len(path) >= 2:

            value_column = (
                None
                if sun_value == "Customer count"
                else sun_value
            )

            with st.spinner(
                "Aggregating categories for the Sunburst chart..."
            ):

                sun_df = aggregate_hierarchy(
                    main_df,
                    path,
                    value_column=value_column,
                    max_segments=150
                )

            if sun_df.empty:

                st.info(
                    "There are no valid category combinations to display."
                )

            else:

                fig = px.sunburst(
                    sun_df,
                    path=path,
                    values="Chart Value",
                    title=f"Customer composition by {' / '.join(path)}"
                )

                fig.update_layout(
                    height=620,
                    margin=dict(
                        t=55,
                        l=10,
                        r=10,
                        b=10
                    )
                )

                fig.update_traces(
                    insidetextorientation="auto",
                    sort=True,
                    maxdepth=3
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="aggregated_sunburst"
                )

                st.caption(
                    f"Showing up to {len(sun_df):,} aggregated category combinations "
                    "(not individual customer records)."
                )

        else:

            st.info(
                "Select at least two category levels to display the Sunburst chart."
            )

    # ----------------------- OUTLIER ANALYSIS --------------------------------

    if cats and nums:

        st.subheader("Outlier and spread analysis")

        box_cat = st.selectbox(
            "Segment",
            cats,
            key="box_cat"
        )

        box_num = st.selectbox(
            "Measure",
            nums,
            key="box_num"
        )

        box_df = main_df.copy()

        top_groups = (
            box_df[box_cat]
            .value_counts(dropna=False)
            .head(20)
            .index
        )

        box_df = box_df[
            box_df[box_cat].isin(top_groups)
        ]

        if len(box_df) > 5000:

            box_df = box_df.sample(
                5000,
                random_state=42
            )

        fig = px.box(
            box_df,
            x=box_cat,
            y=box_num,
            points="outliers"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =============================================================================
# PAGE 4 — DATA QUALITY & INGESTION
# =============================================================================

elif page == "Data Quality & Ingestion":

    st.title("📥 Data Quality & Smart Ingestion")

    st.write(
        "Review the uploaded data, schema, missing values, duplicates and numeric summaries."
    )

    st.subheader("Dataset preview")

    st.dataframe(
        main_df.head(30),
        use_container_width=True
    )

    st.subheader("Schema diagnostics")

    quality = pd.DataFrame({
        "Column": main_df.columns,

        "Data type": [
            str(main_df[c].dtype)
            for c in main_df.columns
        ],

        "Missing values": [
            int(main_df[c].isna().sum())
            for c in main_df.columns
        ],

        "Missing %": [
            safe_percent(
                main_df[c].isna().sum(),
                len(main_df)
            )
            for c in main_df.columns
        ],

        "Unique values": [
            int(main_df[c].nunique(dropna=True))
            for c in main_df.columns
        ]
    })

    st.dataframe(
        quality,
        use_container_width=True,
        hide_index=True
    )

    q1, q2, q3 = st.columns(3)

    q1.metric(
        "Rows",
        f"{len(main_df):,}"
    )

    q2.metric(
        "Columns",
        f"{len(main_df.columns):,}"
    )

    q3.metric(
        "Duplicate rows",
        f"{int(main_df.duplicated().sum()):,}"
    )

    st.subheader("Numeric summary")

    if numeric_columns(main_df):

        st.dataframe(
            main_df[numeric_columns(main_df)].describe().T,
            use_container_width=True
        )

    st.subheader("Download cleaned copy")

    cleaned = main_df.drop_duplicates().copy()

    csv_bytes = cleaned.to_csv(
        index=False
    ).encode("utf-8")

    add_download_button(
        "⬇️ Download cleaned CSV",
        csv_bytes,
        "churn_guardian_cleaned_data.csv",
        "text/csv"
    )


# =============================================================================
# PAGE 5 — ML MODEL EVALUATION
# =============================================================================

elif page == "ML Model Evaluation":

    st.title("🤖 Machine Learning Model Evaluation")

    st.write(
        "Compare Decision Tree, Random Forest and XGBoost using a held-out test set."
    )

    if not XGBOOST_AVAILABLE:

        st.warning(
            "XGBoost is not available in this Python environment. "
            "Decision Tree and Random Forest can still run. "
            "Install the compatible xgboost package to enable the third model."
        )

    if model_results:

        st.markdown(
            f'<span class="warning-pill">{model_note}</span>',
            unsafe_allow_html=True
        )

        if proxy_used:

            st.warning(
                "This dataset has no verified churn label. "
                "The app created a rule-based proxy using the inactivity threshold "
                "and excluded the inactivity-days column from model inputs. "
                "Metrics show how well models reproduce this proxy "
                "on a test split—not confirmed real-world churn prediction accuracy."
            )

        else:

            st.success(
                "A binary target column was detected. "
                "Metrics below are evaluated against that supplied target."
            )

        chosen = st.selectbox(
            "Choose model",
            list(model_results.keys())
        )

        res = model_results[chosen]

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Accuracy",
            f"{res['accuracy'] * 100:.1f}%"
        )

        m2.metric(
            "Precision",
            f"{res['precision'] * 100:.1f}%"
        )

        m3.metric(
            "Recall",
            f"{res['recall'] * 100:.1f}%"
        )

        m4.metric(
            "F1 score",
            f"{res['f1'] * 100:.1f}%"
        )

        left, right = st.columns(2)

        with left:

            fig = px.imshow(
                res["cm"],
                text_auto=True,
                x=["Predicted 0", "Predicted 1"],
                y=["Actual 0", "Actual 1"],
                color_continuous_scale="Purples",
                title="Confusion matrix"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right:

            report = classification_report(
                res["y_test"],
                res["pred"],
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(report).T

            st.subheader("Classification report")

            st.dataframe(
                report_df.round(3),
                use_container_width=True
            )

        st.subheader("Feature contribution")

        pipe = res["pipeline"]
        estimator = pipe.named_steps["model"]

        try:

            importances = estimator.feature_importances_

            transformed_names = (
                pipe.named_steps["prep"].get_feature_names_out()
            )

            fi = pd.DataFrame({
                "Feature": transformed_names,
                "Importance": importances
            })

            fi = (
                fi.sort_values("Importance", ascending=False)
                .head(15)
            )

            fig = px.bar(
                fi.sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Teal"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        except Exception:

            st.info(
                "Feature importance is not available for this model configuration."
            )

    else:

        st.warning(
            model_error or "Model evaluation is unavailable."
        )

        st.markdown("""
        **To enable supervised evaluation, upload a dataset with:**

        - A binary target column such as `Churn`, `Churned`, `Attrition` or `Exited`; or
        - A numeric inactivity field such as `DaysSinceLastPurchase`, which enables an explicitly labelled proxy evaluation.

        For a dataset with no target and no inactivity measure, the app will not invent accuracy, precision, recall or F1 scores.
        """)


# =============================================================================
# PAGE 6 — STAKEHOLDER REPORT
# =============================================================================

elif page == "Stakeholder Report":

    st.title("📑 Stakeholder Business Report")

    st.write(
        "A concise, evidence-based report generated from the uploaded datasets."
    )

    st.subheader("Executive summary")

    st.write(report_summary)

    st.subheader("Data observations")

    for item in observations:
        st.markdown(f"- {item}")

    st.subheader("Recommended actions")

    for item in actions:
        st.markdown(f"- {item}")

    st.subheader("Methodology and limitations")

    st.info(limitations)

    if report_table:

        st.subheader("Key indicator table")

        report_df = pd.DataFrame(
            report_table,
            columns=[
                "Indicator",
                "PY",
                "AY",
                "Change / explanation"
            ]
        )

        st.dataframe(
            report_df,
            use_container_width=True,
            hide_index=True
        )

    report_payload = {
        "summary": report_summary,

        "table": report_table or [
            (
                "Rows in uploaded dataset",
                "N/A",
                f"{len(main_df):,}",
                "Single dataset view"
            )
        ],

        "observations": observations,
        "actions": actions,
        "limitations": limitations
    }

    pdf = make_pdf_report(report_payload)

    if pdf:

        add_download_button(
            "⬇️ Download stakeholder report (PDF)",
            pdf,
            "Churn_Guardian_Stakeholder_Report.pdf",
            "application/pdf"
        )

    else:

        st.error(
            "PDF generation requires ReportLab. "
            "Add `reportlab` to requirements.txt and redeploy."
        )

    csv_report = (
        pd.DataFrame(
            report_table,
            columns=[
                "Indicator",
                "PY",
                "AY",
                "Change / explanation"
            ]
        )
        if report_table
        else pd.DataFrame({
            "Summary": [report_summary]
        })
    )

    add_download_button(
        "⬇️ Download KPI summary (CSV)",
        csv_report.to_csv(index=False).encode("utf-8"),
        "Churn_Guardian_KPI_Summary.csv",
        "text/csv"
    )


# =============================================================================
# PAGE 7 — DATA-AWARE STAKEHOLDER ASSISTANT
# =============================================================================

elif page == "Business Assistant":

    st.title("💬 Churn Guardian Business Assistant")

    st.write(
        "Ask questions about the uploaded data and the calculations in this dashboard."
    )

    st.caption(
        "This built-in assistant uses the current dataset context and predefined explanations. "
        "It does not call an external generative-AI service."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, message in st.session_state.chat_history:

        with st.chat_message(role):
            st.markdown(message)

    question = st.chat_input(
        "Ask about retention, customer movement, revenue, model results or recommendations..."
    )

    if question:

        st.session_state.chat_history.append(
            ("user", question)
        )

        q = question.lower()

        answer_parts = []

        if any(
            word in q
            for word in ["retention", "retained", "returning"]
        ):

            if isinstance(comparison, dict) and "error" not in comparison:

                answer_parts.append(
                    f"The observed retention rate is **{comparison['retention_rate']:.1f}%**. "
                    f"{comparison['retained']:,} customer IDs appear in both PY and AY, "
                    f"out of {comparison['py_count']:,} PY customers."
                )

            else:

                answer_parts.append(
                    "Upload both PY and AY files with a common CustomerID column to calculate observed retention."
                )

        if any(
            word in q
            for word in ["new customer", "new customers", "acquisition", "newly"]
        ):

            if isinstance(comparison, dict) and "error" not in comparison:

                answer_parts.append(
                    f"There are **{comparison['new']:,} newly observed customers** in AY "
                    "(IDs present in AY but not PY)."
                )

        if any(
            word in q
            for word in ["lost", "missing", "not observed", "inactive"]
        ):

            if isinstance(comparison, dict) and "error" not in comparison:

                answer_parts.append(
                    f"**{comparison['not_observed']:,} PY customers are not observed in AY.** "
                    "This does not prove permanent churn; check source coverage, "
                    "account migration and transaction activity."
                )

        if any(
            word in q
            for word in ["revenue", "sales", "spend"]
        ):

            if spend_col:

                value = pd.to_numeric(
                    main_df[spend_col],
                    errors="coerce"
                ).sum()

                answer_parts.append(
                    f"Total {spend_col} in the current AY dataset is **{money(value)}**."
                )

            else:

                answer_parts.append(
                    "I cannot find a revenue/spend column in the current dataset."
                )

        if any(
            word in q
            for word in ["order", "purchase", "frequency"]
        ):

            if orders_col:

                value = pd.to_numeric(
                    main_df[orders_col],
                    errors="coerce"
                ).sum()

                answer_parts.append(
                    f"Total recorded {orders_col} in the current dataset is **{value:,.0f}**."
                )

            else:

                answer_parts.append(
                    "No order-count field was detected in the current dataset."
                )

        if any(
            word in q
            for word in ["satisfaction", "csat", "customer experience"]
        ):

            if satisfaction_col:

                value = pd.to_numeric(
                    main_df[satisfaction_col],
                    errors="coerce"
                ).mean()

                answer_parts.append(
                    f"Average {satisfaction_col} is **{value:.2f}**."
                )

            else:

                answer_parts.append(
                    "No satisfaction score field was detected."
                )

        if any(
            word in q
            for word in [
                "model",
                "accuracy",
                "xgboost",
                "random forest",
                "decision tree",
                "machine learning"
            ]
        ):

            if model_results:

                best = max(
                    model_results,
                    key=lambda n: model_results[n]["f1"]
                )

                answer_parts.append(
                    f"The highest F1 score in the current test split is **{best}** "
                    f"({model_results[best]['f1']:.3f}). "
                    f"Label basis: {model_note}. "
                    "If the label is a proxy, this is not verified churn accuracy."
                )

            else:

                answer_parts.append(
                    "Model metrics are unavailable because the dataset has no valid binary target or supported inactivity proxy."
                )

        if any(
            word in q
            for word in ["recommend", "improve", "action", "what should", "how to"]
        ):

            answer_parts.extend(actions[:3])

        if any(
            word in q
            for word in ["what can you", "help", "features", "what do you do"]
        ):

            answer_parts.append(
                "I can explain customer movement, observed retention, new customers, "
                "available business measures, data quality, ML evaluation and the report. "
                "I only answer from fields available in the uploaded data."
            )

        if not answer_parts:

            answer_parts.append(
                "I can answer questions about observed retention, new customers, "
                "customers not observed in AY, revenue/spend, orders, satisfaction, "
                "model evaluation and retention recommendations. "
                "Try asking one of those questions."
            )

        answer = "\n\n".join(answer_parts)

        st.session_state.chat_history.append(
            ("assistant", answer)
        )

        st.rerun()

    if st.button("Clear conversation"):

        st.session_state.chat_history = []

        st.rerun()


# ------------------------------- FOOTER --------------------------------------

st.markdown("---")

st.caption(
    "🛡️ Churn Guardian | Business analytics support tool | "
    "Validate business decisions with source-system owners."
)
