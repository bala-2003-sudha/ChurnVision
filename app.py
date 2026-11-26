
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
    confusion_matrix, classification_report, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
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
/* Refined dashboard surfaces and stronger hierarchy */
.cg-workspace-hero { position:relative; overflow:hidden; display:flex; align-items:center; gap:1.05rem; padding:1.15rem 1.35rem; margin:.15rem 0 1.15rem; border:1px solid color-mix(in srgb, var(--cg-accent) 28%, #24364e); border-radius:20px; background:linear-gradient(112deg, color-mix(in srgb, var(--cg-page-glow) 76%, #101b2b), rgba(11,20,34,.92)); box-shadow:0 8px 30px rgba(0,0,0,.18), inset 0 1px rgba(255,255,255,.035); }
.cg-workspace-hero:before { content:''; position:absolute; inset:0; background:linear-gradient(115deg, color-mix(in srgb, var(--cg-accent) 10%, transparent), transparent 56%); pointer-events:none; }
.cg-hero-icon { position:relative; display:grid; place-items:center; width:54px; height:54px; flex:0 0 54px; border-radius:16px; font-size:1.65rem; background:color-mix(in srgb, var(--cg-accent) 13%, #101c2c); border:1px solid color-mix(in srgb, var(--cg-accent) 35%, transparent); box-shadow:0 0 22px color-mix(in srgb, var(--cg-accent) 12%, transparent); }
.cg-hero-copy { position:relative; z-index:1; }
.cg-hero-eyebrow { font-size:.66rem; letter-spacing:.15em; font-weight:750; color:var(--cg-accent); margin-bottom:.25rem; }
.cg-hero-title { color:#f7fbff; font-size:clamp(1.25rem,2.2vw,1.8rem); line-height:1.18; font-weight:800; letter-spacing:-.035em; }
.cg-hero-subtitle { color:#a9bdd2; font-size:.88rem; margin-top:.25rem; }
.cg-hero-orb { position:absolute; right:-35px; top:-72px; width:190px; height:190px; border-radius:50%; background:color-mix(in srgb, var(--cg-accent) 13%, transparent); filter:blur(20px); }
.block-container { max-width: 1500px; padding-left: 2.2rem; padding-right: 2.2rem; padding-bottom: 3rem; }
[data-testid="stHorizontalBlock"] { gap: 1rem; }
[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(13,25,43,.48); border: 1px solid rgba(117,151,190,.14); border-radius: 16px; }
[data-testid="stMetric"] { min-height: 108px; display: flex; flex-direction: column; justify-content: center; }
[data-testid="stMetricLabel"] { font-size: .82rem !important; letter-spacing: .025em; }
[data-testid="stMetricValue"] { font-size: clamp(1.35rem, 2.1vw, 2rem) !important; font-weight: 750 !important; }
[data-testid="stPlotlyChart"] { padding: .35rem .25rem; }
[data-testid="stExpander"] { border: 1px solid rgba(117,151,190,.16); border-radius: 13px; }
hr { border-color: rgba(117,151,190,.20) !important; }
.stMarkdown p { line-height: 1.55; }
@media (max-width: 900px) { .block-container { padding-left: 1rem; padding-right: 1rem; } }
div.stButton > button, div.stDownloadButton > button {
    border-radius: 9px; border: 1px solid #2c7894; font-weight: 650;
}
.small-note { color: #9fb3c8; font-size: .88rem; }
.status-pill { padding: 7px 12px; border-radius: 20px; display: inline-block;
    background: #123c35; color: #86efac; border: 1px solid #23785f; font-size: 13px; }
 .warning-pill { padding: 7px 12px; border-radius: 20px; display: inline-block;
    background: #49351a; color: #fcd34d; border: 1px solid #9a6b25; font-size: 13px; }
/* Premium ambient glow: subtle, accessible, and theme-preserving */
div[data-testid="stMetric"], div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"],
div[data-testid="stVerticalBlockBorderWrapper"] { transition: box-shadow .30s ease, transform .25s ease, border-color .30s ease; }
div[data-testid="stMetric"] { box-shadow: 0 0 10px rgba(255,255,255,.035), 0 4px 18px rgba(0,0,0,.12); }
div[data-testid="stMetric"]:hover { transform: translateY(-2px); box-shadow: 0 0 15px rgba(255,255,255,.08), 0 0 22px rgba(100,190,255,.10), 0 8px 24px rgba(0,0,0,.20); }
div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"], div[data-testid="stVerticalBlockBorderWrapper"] { box-shadow: 0 0 10px rgba(255,255,255,.025), 0 5px 20px rgba(0,0,0,.12); border-radius: 14px; }
div[data-testid="stPlotlyChart"]:hover, div[data-testid="stVerticalBlockBorderWrapper"]:hover { box-shadow: 0 0 15px rgba(255,255,255,.055), 0 0 20px rgba(110,170,255,.07), 0 8px 24px rgba(0,0,0,.17); }
.stButton > button, .stDownloadButton > button { transition: box-shadow .25s ease, transform .20s ease, filter .25s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); box-shadow: 0 0 12px rgba(255,255,255,.10), 0 0 18px rgba(100,190,255,.12); filter: brightness(1.04); }
@media (prefers-reduced-motion: reduce) { div[data-testid="stMetric"], div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"], div[data-testid="stVerticalBlockBorderWrapper"], .stButton > button, .stDownloadButton > button { transition:none !important; transform:none !important; } }
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
        r"\.0$", "", regex=True
    )


INDUSTRIES = {
    "Banking & Finance": {
        "customer_terms": ["Customer", "Account", "Client"],
        "activity_terms": ["Transaction", "Balance", "Product", "Account", "Tenure"],
        "retention_language": "account retention",
        "examples": "account closure, reduced product usage, declining balances, service issues",
        "actions": [
            "Prioritize high-value accounts showing declining engagement or product usage.",
            "Review service complaints, fees, and friction in digital banking journeys.",
            "Offer relevant financial products based on verified customer needs and eligibility.",
        ],
    },
    "Healthcare": {
        "customer_terms": ["Patient", "Member"],
        "activity_terms": ["Visit", "Appointment", "Treatment", "FollowUp", "Care", "Provider"],
        "retention_language": "patient engagement / continuity of care",
        "examples": "missed follow-ups, care disengagement, appointment gaps, service experience",
        "actions": [
            "Identify patients with missed or overdue follow-ups and use appropriate outreach.",
            "Review appointment access, care coordination, and patient experience barriers.",
            "Use privacy-conscious, clinically appropriate engagement workflows.",
        ],
    },
    "Education": {
        "customer_terms": ["Student", "Learner", "Enrollment"],
        "activity_terms": ["Course", "Attendance", "Grade", "Engagement", "Semester", "Program"],
        "retention_language": "student persistence / continuation",
        "examples": "attendance decline, low engagement, course progress, discontinuation",
        "actions": [
            "Flag learners with declining attendance or engagement for supportive check-ins.",
            "Review academic, scheduling, financial, and access barriers before outreach.",
            "Connect at-risk students with tutoring, advising, and relevant support resources.",
        ],
    },
    "Hospitality": {
        "customer_terms": ["Guest", "Customer", "Member"],
        "activity_terms": ["Booking", "Reservation", "Stay", "Visit", "Room", "Rating"],
        "retention_language": "repeat-guest retention",
        "examples": "fewer repeat stays, poor ratings, booking gaps, service dissatisfaction",
        "actions": [
            "Personalize guest follow-up using preference and stay history where consent allows.",
            "Investigate low ratings and recurring service issues promptly.",
            "Design loyalty and return-visit offers around guest segments and seasonality.",
        ],
    },
    "Telecommunications": {
        "customer_terms": ["Customer", "Subscriber", "Account"],
        "activity_terms": ["Plan", "Usage", "Call", "Data", "Network", "Recharge", "Tenure"],
        "retention_language": "subscriber retention / cancellation",
        "examples": "plan downgrades, low usage, network issues, billing complaints, cancellation",
        "actions": [
            "Investigate network quality and billing/service complaints for high-risk subscribers.",
            "Offer plan-fit reviews for customers with usage changes or repeated overages.",
            "Prioritize retention outreach by risk, tenure, and customer value.",
        ],
    },
    "SaaS & Technology": {
        "customer_terms": ["Customer", "Account", "Organization", "User"],
        "activity_terms": ["Subscription", "Renewal", "Login", "Usage", "Feature", "Seat", "Plan"],
        "retention_language": "subscription retention / renewal",
        "examples": "declining product usage, renewal risk, low adoption, support friction",
        "actions": [
            "Identify accounts with declining product usage or low adoption of key features.",
            "Provide targeted onboarding, training, and customer-success support.",
            "Review renewal timing, unresolved support issues, and plan-value alignment.",
        ],
    },
}

# Separate, memorable page identities. Chart series still use semantic colors for outcomes.
PAGE_PALETTES = {
    "Executive Dashboard": {"accent": "#22D3EE", "secondary": "#60A5FA", "tertiary": "#A78BFA", "gradient": "#0B2440", "chart": ["#22D3EE", "#60A5FA", "#A78BFA", "#34D399", "#FBBF24", "#FB7185"]},
    "PY vs AY Comparison": {"accent": "#A78BFA", "secondary": "#818CF8", "tertiary": "#C084FC", "gradient": "#25184D", "chart": ["#818CF8", "#A78BFA", "#C084FC", "#38BDF8", "#F472B6", "#FBBF24"]},
    "Advanced Visualizer": {"accent": "#2DD4BF", "secondary": "#5EEAD4", "tertiary": "#38BDF8", "gradient": "#073B3A", "chart": ["#2DD4BF", "#38BDF8", "#A3E635", "#F59E0B", "#F472B6", "#818CF8"]},
    "Data Quality & Ingestion": {"accent": "#34D399", "secondary": "#A3E635", "tertiary": "#86EFAC", "gradient": "#123B2C", "chart": ["#34D399", "#A3E635", "#22D3EE", "#FBBF24", "#FB7185", "#818CF8"]},
    "ML Model Evaluation": {"accent": "#E879F9", "secondary": "#C084FC", "tertiary": "#F472B6", "gradient": "#42144D", "chart": ["#E879F9", "#C084FC", "#818CF8", "#22D3EE", "#FBBF24", "#34D399"]},
    "Stakeholder Report": {"accent": "#FB7185", "secondary": "#F472B6", "tertiary": "#FDA4AF", "gradient": "#4A182D", "chart": ["#FB7185", "#F472B6", "#FDA4AF", "#A78BFA", "#22D3EE", "#FBBF24"]},
    "Business Assistant": {"accent": "#FBBF24", "secondary": "#F59E0B", "tertiary": "#FDE68A", "gradient": "#493015", "chart": ["#FBBF24", "#F59E0B", "#FDE68A", "#34D399", "#22D3EE", "#F472B6"]},
}


def render_workspace_header(title, subtitle, icon="🛡️"):
    """Render a compact premium page header using the active workspace accent."""
    st.markdown(
        f"""<div class='cg-workspace-hero'><div class='cg-hero-icon'>{icon}</div>
        <div class='cg-hero-copy'><div class='cg-hero-eyebrow'>CHURN GUARDIAN · ANALYTICS WORKSPACE</div>
        <div class='cg-hero-title'>{title}</div><div class='cg-hero-subtitle'>{subtitle}</div></div>
        <div class='cg-hero-orb'></div></div>""",
        unsafe_allow_html=True
    )


INDUSTRY_COLUMN_ALIASES = {
    "Banking & Finance": {
        "id": ["CustomerID", "ClientID", "AccountID", "CustomerNumber", "AccountNumber", "MemberID"],
        "activity_date": ["LastTransactionDate", "LastActivityDate", "TransactionDate", "LastContactDate"],
        "target": ["Churn", "Attrition", "Exited", "AccountClosed", "ClosedAccount", "CustomerExit"],
        "activity": ["DaysInactive", "DaysSinceLastTransaction", "DaysSinceLastActivity", "RecencyDays", "LastActivityDays"],
        "value": ["Balance", "AccountBalance", "Revenue", "MonthlyCharges", "TotalSpend", "Amount"],
    },
    "Healthcare": {
        "id": ["PatientID", "MemberID", "ClientID", "CustomerID", "PersonID", "MRN"],
        "activity_date": ["LastVisitDate", "LastAppointmentDate", "LastFollowUpDate", "LastActivityDate"],
        "target": ["Attrition", "Churn", "Disengaged", "LostToFollowUp", "Inactive"],
        "activity": ["DaysSinceLastVisit", "DaysSinceLastAppointment", "DaysInactive", "DaysSinceLastActivity", "RecencyDays"],
        "value": ["Visits", "VisitCount", "Appointments", "Claims", "Cost", "Charges"],
    },
    "Education": {
        "id": ["StudentID", "LearnerID", "EnrollmentID", "MemberID", "CustomerID"],
        "activity_date": ["LastLoginDate", "LastClassDate", "LastAttendanceDate", "LastActivityDate"],
        "target": ["Dropout", "DroppedOut", "Attrition", "Withdrawn", "Discontinued", "Churn"],
        "activity": ["DaysSinceLastLogin", "DaysSinceLastActivity", "DaysInactive", "DaysSinceLastClass", "RecencyDays"],
        "value": ["Attendance", "AttendanceRate", "Engagement", "Credits", "Grade", "Progress", "Fees", "Tuition", "TotalFees", "PaymentAmount"],
    },
    "Hospitality": {
        "id": ["GuestID", "CustomerID", "MemberID", "ClientID"],
        "activity_date": ["LastStayDate", "LastBookingDate", "LastVisitDate", "LastActivityDate"],
        "target": ["Churn", "Attrition", "LostGuest", "NotReturning"],
        "activity": ["DaysSinceLastStay", "DaysSinceLastVisit", "DaysInactive", "RecencyDays"],
        "value": ["Bookings", "BookingCount", "StayCount", "Revenue", "Spend", "Rating"],
    },
    "Telecommunications": {
        "id": ["SubscriberID", "CustomerID", "AccountID", "PhoneNumber", "MobileNumber"],
        "activity_date": ["LastUsageDate", "LastRechargeDate", "LastCallDate", "LastActivityDate"],
        "target": ["Churn", "Churned", "Cancelled", "Canceled", "Disconnected", "Attrition"],
        "activity": ["DaysInactive", "DaysSinceLastUsage", "DaysSinceLastRecharge", "DaysSinceLastActivity", "RecencyDays"],
        "value": ["MonthlyCharges", "TotalCharges", "DataUsage", "Usage", "Revenue", "RechargeAmount"],
    },
    "SaaS & Technology": {
        "id": ["AccountID", "OrganizationID", "CustomerID", "UserID", "SubscriptionID"],
        "activity_date": ["LastLoginDate", "LastUsageDate", "LastFeatureUseDate", "LastActivityDate"],
        "target": ["Churn", "Cancelled", "Canceled", "NonRenewal", "RenewalStatus", "Attrition"],
        "activity": ["DaysSinceLastLogin", "DaysSinceLastUsage", "DaysInactive", "LastActiveDays", "RecencyDays"],
        "value": ["MRR", "ARR", "MonthlyRecurringRevenue", "SubscriptionValue", "Usage", "Seats", "FeatureUsage"],
    },
}




def detect_industry_from_dataset(py_frame=None, ay_frame=None):
    """Infer a supported industry from strong schema signals; return None if ambiguous."""
    frames = [f for f in (py_frame, ay_frame) if isinstance(f, pd.DataFrame) and not f.empty]
    if not frames:
        return None, 0, []
    normalized_cols = set()
    raw_cols = []
    for frame in frames:
        for col in frame.columns:
            raw = str(col).strip()
            raw_cols.append(raw)
            normalized_cols.add(re.sub(r"[^a-z0-9]", "", raw.lower()))
    # Strong domain-specific signals are weighted above generic words like customer/status/plan.
    signatures = {
        "Banking & Finance": {
            "strong": ["creditscore", "creditcard", "loanamount", "mortgage", "accountbalance", "bankbalance", "estimatedsalary", "numofproducts", "creditlimit", "overdraft"],
            "medium": ["transactiondate", "accounttype", "debit", "credit", "interest", "branch", "investment", "savingsaccount"],
        },
        "Healthcare": {
            "strong": ["patientid", "mrn", "diagnosis", "treatment", "medicalrecord", "providerid", "lastvisitdate", "appointmentdate", "prescription", "clinical"],
            "medium": ["visitcount", "appointment", "healthplan", "insurance", "careplan", "claim", "procedure", "patient"],
        },
        "Education": {
            "strong": ["studentid", "enrollmentid", "courseid", "attendance", "grade", "semester", "creditscompleted", "gpa", "dropout", "schoolid"],
            "medium": ["learner", "class", "faculty", "tuition", "academic", "program", "assignment", "exam", "student"],
        },
        "Hospitality": {
            "strong": ["guestid", "bookingid", "reservationid", "checkindate", "checkoutdate", "roomtype", "staycount", "hotelid", "laststaydate"],
            "medium": ["booking", "reservation", "guest", "room", "stay", "hotel", "hospitality", "rating", "review", "nights"],
        },
        "Telecommunications": {
            "strong": ["subscriberid", "monthlydatausagegb", "monthlycallminutes", "internetservice", "internationalroaming", "databundle", "voiceplan", "lastrechargedate", "networktype", "telecom"],
            "medium": ["datausage", "callminutes", "subscriptionplan", "contracttype", "recharge", "mobile", "phone", "sim", "roaming", "telecommunication", "voice", "broadband"],
        },
        "SaaS & Technology": {
            "strong": ["mrr", "arr", "monthlyrecurringrevenue", "featureusage", "lastlogindate", "seatcount", "organizationid", "subscriptionid", "productusage", "renewaldate"],
            "medium": ["saas", "subscription", "login", "feature", "seat", "software", "workspace", "api", "license", "productadoption", "renewal", "appusage"],
        },
    }
    scores = {}
    evidence = {}
    for industry, spec in signatures.items():
        strong_hits = [x for x in spec["strong"] if x in normalized_cols]
        medium_hits = [x for x in spec["medium"] if any(x in c for c in normalized_cols)]
        scores[industry] = len(strong_hits) * 4 + len(medium_hits)
        evidence[industry] = strong_hits + medium_hits
    # Extra evidence from customer identifier values (e.g. TEL2025... IDs).
    for frame in frames:
        for col in frame.columns:
            norm = re.sub(r"[^a-z0-9]", "", str(col).lower())
            if norm in {"customerid", "subscriberid", "accountid"}:
                sample = frame[col].dropna().astype(str).head(30).str.upper()
                if sample.str.match(r"^(TEL|MOB|SUB)[0-9]{6,}$").mean() >= .6:
                    scores["Telecommunications"] += 8
                    if "telecom-style customer IDs" not in evidence["Telecommunications"]:
                        evidence["Telecommunications"].append("telecom-style customer IDs")
                    break
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_industry, top_score = ranked[0]
    runner_score = ranked[1][1]
    # Require a clear margin and more than a single generic signal.
    if top_score < 5 or top_score - runner_score < 2:
        return None, top_score, evidence.get(top_industry, [])
    signal_strength = "Strong" if top_score >= 12 and (top_score - runner_score) >= 5 else "Moderate"
    return top_industry, signal_strength, evidence[top_industry]


def industry_dataset_signature(py_frame=None, ay_frame=None, py_name="", ay_name=""):
    pieces = [py_name, ay_name]
    for frame in (py_frame, ay_frame):
        if isinstance(frame, pd.DataFrame):
            pieces.append("|".join(map(str, frame.columns)))
            pieces.append(str(len(frame)))
    return "::".join(pieces)


def mark_industry_manual():
    st.session_state["_industry_manually_selected"] = True


def normalize_identifier(series, kind="generic"):
    values = series.fillna("").astype(str).str.strip()
    if kind == "email":
        return values.str.lower()
    if kind == "phone":
        return values.str.replace(r"\D", "", regex=True)
    return values.str.lower().str.replace(r"\.0$", "", regex=True)


def detect_identity_column(df, industry):
    aliases = INDUSTRY_COLUMN_ALIASES.get(industry, {}).get("id", []) + [
        "CustomerID", "Customer_ID", "ClientID", "AccountID", "UserID", "MemberID", "StudentID", "PatientID", "GuestID", "SubscriberID", "OrganizationID"
    ]
    found = find_column(df, aliases)
    if found:
        normalized = re.sub(r"[^a-z0-9]", "", found.lower())
        if normalized in {"phonenumber", "mobilenumber", "contactnumber"}:
            return found, "phone"
        if normalized not in {"transactionid", "purchaseid", "orderid", "recordid", "rowid", "invoiceid"}:
            return found, "id"
    email = find_column(df, ["Email", "EmailAddress", "ContactEmail", "Mail"])
    if email:
        return email, "email"
    phone = find_column(df, ["Phone", "PhoneNumber", "Mobile", "MobileNumber", "ContactNumber"])
    if phone:
        return phone, "phone"
    return None, None


def map_churn_binary(series, industry):
    """Map common attrition values to 0=retained/active and 1=attrited.
    If labels are unfamiliar, return None so callers can use a clearly documented fallback.
    """
    positive = {
        "yes", "true", "1", "1.0", "y", "churn", "churned", "attrited", "exited",
        "inactive", "closed", "closed account", "cancelled", "canceled",
        "disconnected", "dropout", "dropped out", "withdrawn", "discontinued",
        "lost", "lost to follow up", "disengaged", "nonrenewal", "non-renewal",
        "not renewed", "terminated", "left", "no longer active"
    }
    negative = {
        "no", "false", "0", "0.0", "n", "retained", "active", "stay", "stayed",
        "open", "current", "enrolled", "continuing", "continued", "renewed",
        "not churned", "not attrited", "not dropped", "not cancelled", "not canceled",
        "engaged", "returning", "ongoing"
    }
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(lambda v: 1 if v in positive else (0 if v in negative else np.nan))
    if mapped.notna().sum() >= max(2, int(series.notna().sum() * 0.8)) and mapped.dropna().nunique() == 2:
        return mapped
    return None


def detect_identity_pair(py_frame, ay_frame, industry):
    """Choose a reliable identity field available in BOTH annual datasets.
    Prefer same-named stable IDs, then shared email/phone, and only then
    cross-named ID fields (which may be different identifier systems).
    """
    industry_ids = INDUSTRY_COLUMN_ALIASES.get(industry, {}).get("id", [])
    generic_ids = ["CustomerID", "Customer_ID", "ClientID", "AccountID", "UserID", "MemberID", "StudentID", "PatientID", "GuestID", "SubscriberID", "OrganizationID", "CustomerNumber", "AccountNumber"]
    id_candidates = industry_ids + generic_ids
    py_lookup = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in py_frame.columns}
    ay_lookup = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in ay_frame.columns}
    for candidate in id_candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in py_lookup and key in ay_lookup:
            return py_lookup[key], ay_lookup[key], "id"
    for candidates, kind in [
        (["Email", "EmailAddress", "ContactEmail", "Mail"], "email"),
        (["Phone", "PhoneNumber", "Mobile", "MobileNumber", "ContactNumber"], "phone"),
    ]:
        py_col = find_column(py_frame, candidates)
        ay_col = find_column(ay_frame, candidates)
        if py_col and ay_col:
            return py_col, ay_col, kind
    # Last resort: cross-named IDs only. Validate the resulting overlap and duplicates.
    py_col = find_column(py_frame, id_candidates)
    ay_col = find_column(ay_frame, id_candidates)
    if py_col and ay_col:
        return py_col, ay_col, "id"
    return None, None, None


def map_binary_target_for_display(series, industry):
    """Map a verified binary target for KPI/segment display; numeric direction is explicit."""
    mapped = map_churn_binary(series, industry)
    if mapped is not None and mapped.dropna().nunique() == 2:
        return mapped
    if pd.api.types.is_numeric_dtype(series) and series.dropna().nunique() == 2:
        values = sorted(pd.to_numeric(series, errors="coerce").dropna().unique().tolist())
        return pd.to_numeric(series, errors="coerce").map({values[0]: 0, values[1]: 1})
    return None


def identify_churn_target(df, industry):
    """Find explicitly named churn/attrition outcomes; generic Status is validated separately."""
    candidates = INDUSTRY_COLUMN_ALIASES.get(industry, {}).get("target", []) + [
        "Churn", "Churned", "ChurnFlag", "IsChurned", "Attrition", "AttritionFlag",
        "Exited", "ExitFlag", "Dropout", "DroppedOut", "Cancelled", "Canceled",
        "Cancellation", "Disconnected", "Disconnection", "NonRenewal", "LostToFollowUp",
        "AccountClosed", "ClosedAccount", "CustomerExit", "LostGuest", "NotReturning",
        "Disengaged", "Withdrawn", "Discontinued"
    ]
    found = find_column(df, candidates)
    if found:
        return found
    # Generic outcome names are considered only after checking their actual values.
    generic = find_column(df, ["Target", "Label", "Outcome", "ChurnTarget", "RetentionTarget"])
    if generic is not None:
        raw = df[generic]
        if raw.dropna().nunique() == 2 and (pd.api.types.is_numeric_dtype(raw) or map_churn_binary(raw, industry) is not None):
            return generic
    # Status is deliberately last and must be binary with interpretable values.
    status = find_column(df, ["CustomerStatus", "AccountStatus", "SubscriptionStatus", "EnrollmentStatus", "PatientStatus", "GuestStatus", "SubscriberStatus", "Status"])
    if status is not None:
        raw = df[status]
        mapped = map_churn_binary(raw, industry)
        if raw.dropna().nunique() == 2 and mapped is not None and mapped.dropna().nunique() == 2:
            return status
    return None


def classify_analysis_mode(df, industry, threshold):
    """Select one of the three modes using data evidence, never industry alone."""
    target = identify_churn_target(df, industry)
    if target is not None:
        raw = df[target]
        mapped = map_churn_binary(raw, industry)
        if mapped is not None and mapped.dropna().nunique() == 2:
            return "actual_target", target, "A usable binary target with interpretable class labels was detected."
        if pd.api.types.is_numeric_dtype(raw) and raw.dropna().nunique() == 2:
            return "actual_target", target, "A two-class numeric target was detected; numeric class direction requires review."
        # Do not elevate ambiguous categories to actual churn labels.

    inactivity_col = detect_inactivity_column(df, industry)
    if inactivity_col:
        inactivity = pd.to_numeric(df[inactivity_col], errors="coerce")
        proxy = (inactivity >= threshold).where(inactivity.notna())
        if proxy.dropna().nunique() == 2 and proxy.notna().sum() >= 30:
            return "proxy_target", inactivity_col, f"A usable inactivity field supports a transparent {threshold}-day rule with both classes present."

    date_col = detect_activity_date_column(df, industry)
    if date_col:
        inactivity, ref_date, detected_date = derive_inactivity_days(df, industry)
        if inactivity is not None:
            proxy = (inactivity >= threshold).where(inactivity.notna())
            if proxy.dropna().nunique() == 2 and proxy.notna().sum() >= 30:
                return "proxy_target", date_col, f"A last-activity date supports a {threshold}-day inactivity proxy relative to {ref_date.date()}."

    return "exploratory", None, "No verified binary target or sufficiently supported inactivity proxy was found."


def detect_inactivity_column(df, industry):
    candidates = INDUSTRY_COLUMN_ALIASES.get(industry, {}).get("activity", []) + [
        "DaysSinceLastPurchase", "DaysInactive", "RecencyDays", "DaysSinceLastOrder", "DaysSinceLastActivity", "DaysSinceLastLogin"
    ]
    return find_column(df, candidates)


def detect_activity_date_column(df, industry):
    candidates = INDUSTRY_COLUMN_ALIASES.get(industry, {}).get("activity_date", []) + [
        "LastActivityDate", "LastActiveDate", "LastLoginDate", "LastVisitDate", "LastTransactionDate", "LastUsageDate", "LastContactDate"
    ]
    return find_column(df, candidates)


def derive_inactivity_days(df, industry):
    date_col = detect_activity_date_column(df, industry)
    if not date_col:
        return None, None, None
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    valid = parsed.dropna()
    if valid.empty:
        return None, None, date_col
    # Use the latest date present in this dataset as the snapshot reference date.
    reference_date = valid.max()
    days = (reference_date - parsed).dt.days.astype(float)
    days[parsed.isna()] = np.nan
    return days, reference_date, date_col


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

# Central upload state is initialized before the sidebar so industry inference can
# update the selectbox value safely on reruns (including after a new upload).
for _key, _default in {
    "py_data": None, "ay_data": None, "py_filename": "", "ay_filename": "",
    "_industry_manually_selected": False, "_industry_dataset_signature": "",
}.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default

_auto_industry, _auto_confidence, _auto_evidence = detect_industry_from_dataset(
    st.session_state.py_data, st.session_state.ay_data
)
_current_signature = industry_dataset_signature(
    st.session_state.py_data, st.session_state.ay_data,
    st.session_state.py_filename, st.session_state.ay_filename
)
if st.session_state.py_data is not None or st.session_state.ay_data is not None:
    if _current_signature != st.session_state.get("_industry_dataset_signature", ""):
        st.session_state["_industry_dataset_signature"] = _current_signature
        st.session_state["_industry_manually_selected"] = False
        if _auto_industry:
            st.session_state["selected_industry"] = _auto_industry
    elif not st.session_state.get("_industry_manually_selected", False) and _auto_industry:
        st.session_state["selected_industry"] = _auto_industry

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
    selected_palette = PAGE_PALETTES.get(page, PAGE_PALETTES["Executive Dashboard"])
    swatches = "".join(
        f'<span style="display:inline-block;width:18px;height:18px;border-radius:5px;margin-right:5px;background:{c};box-shadow:0 0 10px {c}55"></span>'
        for c in selected_palette["chart"][:5]
    )
    st.markdown(f"<div style='margin:0 0 8px 0'>{swatches}</div><small style='color:#9fb3c8'>Active page palette</small>", unsafe_allow_html=True)
    st.subheader("Industry intelligence")

    selected_industry = st.selectbox(
        "Select industry",
        list(INDUSTRIES.keys()),
        key="selected_industry",
        on_change=mark_industry_manual,
        help="Industry is automatically inferred from dataset fields when confidence is strong. You can override it at any time."
    )

    if _auto_industry and (st.session_state.py_data is not None or st.session_state.ay_data is not None):
        if st.session_state.get("_industry_manually_selected", False):
            st.caption(f"🖐️ Manual industry selection · Dataset signal: {_auto_industry} ({_auto_confidence.lower()} match)")
        else:
            st.caption(f"🔎 Auto-detected: {_auto_industry} · {_auto_confidence.lower()} schema match")
    elif st.session_state.py_data is None and st.session_state.ay_data is None:
        st.caption("Industry will be auto-detected after you upload your dataset.")
    else:
        st.caption("No strong industry signal found in the uploaded columns. Please select the industry manually.")

    st.caption(f"Industry lens: {INDUSTRIES[selected_industry]['retention_language']}")

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


# ----------------------- ACTIVE PAGE THEME ----------------------------------

active_palette = PAGE_PALETTES.get(page, PAGE_PALETTES["Executive Dashboard"])
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = active_palette["chart"]
st.markdown(f"""
<style>
:root {{ --cg-accent: {active_palette['accent']}; --cg-secondary: {active_palette['secondary']}; --cg-tertiary: {active_palette['tertiary']}; --cg-page-glow: {active_palette['gradient']}; }}
.stApp {{ background: radial-gradient(ellipse at 12% 0%, var(--cg-page-glow) 0%, #08111f 58%, #070d18 100%) !important; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, var(--cg-page-glow), #0d192b 68%) !important; }}
[data-testid="stSidebar"] [aria-selected="true"] {{ border-left: 3px solid var(--cg-accent) !important; background: color-mix(in srgb, var(--cg-accent) 10%, transparent) !important; }}
[data-testid="stMetric"] {{ border-color: color-mix(in srgb, var(--cg-accent) 46%, #20324a) !important; box-shadow: 0 0 18px color-mix(in srgb, var(--cg-accent) 9%, transparent); }}
[data-testid="stMetric"] [data-testid="stMetricValue"] {{ text-shadow: 0 0 18px color-mix(in srgb, var(--cg-accent) 35%, transparent); }}
h1, h2, h3 {{ text-shadow: 0 0 14px color-mix(in srgb, var(--cg-accent) 20%, transparent); }}
div.stButton > button, div.stDownloadButton > button {{ border-color: var(--cg-accent) !important; box-shadow: 0 0 9px color-mix(in srgb, var(--cg-accent) 10%, transparent); }}
</style>
""", unsafe_allow_html=True)

# ------------------------ CENTRAL DATA UPLOAD -------------------------------

# The ONLY upload controls in the application are on the PY vs AY Comparison page.

if page == "PY vs AY Comparison":

    render_workspace_header("PY vs AY · Dataset Upload", "Upload both year snapshots once to activate all connected workspaces.", "📂")

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
                st.session_state["_industry_manually_selected"] = False
                st.session_state["_industry_dataset_signature"] = ""

                st.success(
                    "Both datasets were loaded. All seven pages can now use the same data."
                )

                st.rerun()


    if clear_clicked:
        st.session_state.py_data = None
        st.session_state.ay_data = None
        st.session_state.py_filename = ""
        st.session_state.ay_filename = ""
        st.session_state["_industry_manually_selected"] = False
        st.session_state["_industry_dataset_signature"] = ""
        st.session_state["selected_industry"] = "Banking & Finance"
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

    render_workspace_header("Enterprise Customer Intelligence", "Upload annual datasets to unlock connected analytics, comparisons and insights.", "🛡️")
    st.info(f"Selected industry: **{selected_industry}** · {INDUSTRIES[selected_industry]['retention_language']}")

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

id_col, id_kind = detect_identity_column(main_df, selected_industry)

py_id_col, py_id_kind = (
    detect_identity_column(py_df, selected_industry)
    if py_df is not None else (None, None)
)
ay_id_col, ay_id_kind = (
    detect_identity_column(ay_df, selected_industry)
    if ay_df is not None else (None, None)
)


# Detect business measures by common aliases.

industry_value_candidates = INDUSTRY_COLUMN_ALIASES[selected_industry]["value"]
industry_metric_col = find_column(main_df, industry_value_candidates)
financial_candidates_by_industry = {
    "Banking & Finance": ["Balance", "AccountBalance", "Revenue", "Income", "TotalSpend", "Amount"],
    "Healthcare": ["Cost", "Charges", "TotalCost", "Revenue", "BillingAmount", "Amount"],
    "Education": ["Fees", "Tuition", "Revenue", "TotalFees", "PaymentAmount", "Amount"],
    "Hospitality": ["Revenue", "Spend", "TotalSpend", "Amount"],
    "Telecommunications": ["MonthlyCharges", "TotalCharges", "Revenue", "RechargeAmount", "Amount"],
    "SaaS & Technology": ["MRR", "ARR", "MonthlyRecurringRevenue", "SubscriptionValue", "Revenue", "Amount"],
}
spend_col = find_column(
    main_df,
    financial_candidates_by_industry[selected_industry] + ["TotalSpend", "Total_Spend", "Revenue", "Sales", "TotalRevenue", "Amount"]
)

industry_metric_label = {
    "Banking & Finance": "Account balance / financial value",
    "Healthcare": "Healthcare activity / cost measure",
    "Education": "Student progress / fee measure",
    "Hospitality": "Guest / booking measure",
    "Telecommunications": "Subscriber usage / charge measure",
    "SaaS & Technology": "Subscription / product-usage measure",
}[selected_industry]

orders_col = find_column(
    main_df,
    ["OrderCount", "Orders", "NumberOfOrders", "PurchaseCount", "VisitCount", "BookingCount", "StayCount", "AppointmentCount", "TransactionCount", "UsageCount"]
)

aov_col = find_column(
    main_df,
    ["AverageOrderValue", "AOV", "AvgOrderValue", "AverageRevenuePerUser", "ARPU", "AverageBalance", "AverageSpend"]
)

satisfaction_col = find_column(
    main_df,
    ["SatisfactionScore", "CustomerSatisfaction", "CSAT", "PatientSatisfaction", "GuestRating", "Rating", "NPS", "StudentSatisfaction"]
)

region_col = find_column(
    main_df,
    ["Region", "Location", "Market", "Territory"]
)

category_col = find_column(
    main_df,
    ["PreferredCategory", "Category", "ProductCategory", "Segment", "CustomerSegment", "Plan", "PlanType", "Course", "Program", "ServiceType", "Department", "MembershipTier"]
)

days_col = detect_inactivity_column(main_df, selected_industry)
activity_date_col = detect_activity_date_column(main_df, selected_industry)

target_col = identify_churn_target(main_df, selected_industry)
analysis_mode, analysis_source, analysis_mode_reason = classify_analysis_mode(
    main_df, selected_industry, inactivity_days
)

MODE_LABELS = {
    "actual_target": "Mode 1 · Actual target / supervised ML",
    "proxy_target": "Mode 2 · Derived proxy / rule-based ML",
    "exploratory": "Mode 3 · Exploratory analytics",
}


# --------------------------- PY / AY MATCHING --------------------------------

comparison = None

if py_df is not None and ay_df is not None:

    py_identity, ay_identity, match_kind = detect_identity_pair(py_df, ay_df, selected_industry)

    if py_identity and ay_identity and match_kind:
        py_work = py_df.copy()
        ay_work = ay_df.copy()
        py_work["_match_id"] = normalize_identifier(py_work[py_identity], match_kind)
        ay_work["_match_id"] = normalize_identifier(ay_work[ay_identity], match_kind)
        # Exclude empty / invalid normalized identifiers.
        py_work = py_work[py_work["_match_id"].ne("")]
        ay_work = ay_work[ay_work["_match_id"].ne("")]
        py_ids = set(py_work["_match_id"])
        ay_ids = set(ay_work["_match_id"])
        common_ids = py_ids & ay_ids
        lost_ids = py_ids - ay_ids
        new_ids = ay_ids - py_ids
        # Detect identifiers that appear to be regenerated/year-coded rather than stable across snapshots.
        py_raw_ids = py_work["_match_id"].astype(str)
        ay_raw_ids = ay_work["_match_id"].astype(str)
        year_code_pattern = re.compile(r"^([a-z]+)(20\d{2})(\d+)$", re.I)
        py_codes = [year_code_pattern.match(v) for v in py_raw_ids]
        ay_codes = [year_code_pattern.match(v) for v in ay_raw_ids]
        py_years = {m.group(2) for m in py_codes if m}
        ay_years = {m.group(2) for m in ay_codes if m}
        year_coded_mismatch = bool(
            len(py_years) == 1 and len(ay_years) == 1 and py_years != ay_years
            and len(py_codes) >= max(20, int(len(py_raw_ids) * .8))
            and len(ay_codes) >= max(20, int(len(ay_raw_ids) * .8))
            and len(common_ids) == 0
        )
        comparison = {
            "py_count": len(py_ids), "ay_count": len(ay_ids),
            "retained": len(common_ids), "not_observed": len(lost_ids), "new": len(new_ids),
            "py_ids": py_ids, "ay_ids": ay_ids, "common_ids": common_ids,
            "lost_ids": lost_ids, "new_ids": new_ids, "py_work": py_work, "ay_work": ay_work,
            "match_field": f"{py_identity} / {ay_identity}", "match_kind": match_kind,
            "py_duplicates": int(py_work["_match_id"].duplicated().sum()),
            "ay_duplicates": int(ay_work["_match_id"].duplicated().sum()),
            "retention_comparable": not year_coded_mismatch,
            "identity_warning": (f"IDs appear year-coded ({', '.join(sorted(py_years))} vs {', '.join(sorted(ay_years))}) and have zero overlap. These may be regenerated identifiers; customer retention cannot be reliably calculated from these fields." if year_coded_mismatch else None)
        }
        comparison["retention_rate"] = safe_percent(len(common_ids), len(py_ids))
        comparison["not_observed_rate"] = safe_percent(len(lost_ids), len(py_ids))
        comparison["new_customer_share"] = safe_percent(len(new_ids), len(ay_ids))
    else:
        comparison = {
            "error": (
                "A comparable identity field was not detected in both datasets. "
                "Provide a stable Customer/Account/Patient/Student/Guest/Subscriber ID, "
                "or a shared email or phone field. Transaction/order IDs are not used as customer identities."
            )
        }


# --------------------------- MODEL PREPARATION -------------------------------

def prepare_supervised_data(df, threshold):
    """Prepare a supervised dataset only when a defensible label exists."""
    work = df.copy()
    mode, detected_target, mode_reason = classify_analysis_mode(work, selected_industry, threshold)
    if mode == "exploratory":
        return None, None, False, mode_reason, mode_reason

    proxy_used = mode == "proxy_target"
    label_description = mode_reason
    excluded = set()

    if not proxy_used:
        raw = work[detected_target]
        if pd.api.types.is_numeric_dtype(raw):
            values = sorted(pd.to_numeric(raw, errors="coerce").dropna().unique().tolist())
            if len(values) != 2:
                return None, None, False, label_description, "Actual target must have exactly two numeric classes."
            y = pd.to_numeric(raw, errors="coerce").map({values[0]: 0, values[1]: 1})
            label_description += f" Numeric classes mapped {values[0]}→0 and {values[1]}→1; verify that the higher class represents attrition."
        else:
            y = map_churn_binary(raw, selected_industry)
            if y is None:
                return None, None, False, label_description, "Target labels are not confidently mapped to retained vs attrited."
        excluded.add(detected_target)
        label_description = f"Observed target: {detected_target}. " + label_description
    else:
        inactivity_col = detect_inactivity_column(work, selected_industry)
        if inactivity_col:
            inactivity = pd.to_numeric(work[inactivity_col], errors="coerce")
            excluded.add(inactivity_col)
            proxy_source = inactivity_col
        else:
            inactivity, reference_date, date_col = derive_inactivity_days(work, selected_industry)
            if inactivity is None:
                return None, None, True, label_description, "A date field was detected but could not produce usable inactivity values."
            excluded.add(date_col)
            proxy_source = f"{date_col} (reference {reference_date.date()})"
        y = (inactivity >= threshold).astype(float)
        y[inactivity.isna()] = np.nan
        label_description = f"Proxy target: inactivity ≥ {threshold} days from {proxy_source}; the source field is excluded from model features. Metrics evaluate agreement with this rule—not confirmed churn."

    # Remove identifiers and high-cardinality text identifiers from features.
    id_names = {"customerid", "customeridentifier", "clientid", "accountid", "accountnumber", "userid", "useridentifier", "memberid", "studentid", "learnerid", "patientid", "personid", "mrn", "guestid", "subscriberid", "organizationid", "orgid", "subscriptionid", "enrollmentid", "customernumber", "transactionid", "purchaseid", "orderid", "recordid", "rowid", "invoiceid", "id", "email", "emailaddress", "contactemail", "phone", "phonenumber", "mobile", "mobilenumber", "contactnumber"}
    for col in work.columns:
        normalized = re.sub(r"[^a-z0-9]", "", str(col).lower())
        if normalized in id_names:
            excluded.add(col)

    X = work.drop(columns=list(excluded), errors="ignore").copy().dropna(axis=1, how="all")
    for col in list(X.columns):
        if X[col].nunique(dropna=True) > max(1000, int(len(X) * .95)):
            X = X.drop(columns=[col])
    valid = y.notna()
    X, y = X.loc[valid].copy(), y.loc[valid].astype(int)
    if len(X) < 30 or y.nunique() < 2:
        return None, None, proxy_used, label_description, "At least 30 usable rows and two classes are required for model training."
    if y.value_counts().min() < 2:
        return None, None, proxy_used, label_description, "Each class needs at least two records for stratified splitting."
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
        try:
            prob = pipe.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, prob)
            pr_auc = average_precision_score(y_test, prob)
            fpr, tpr, _ = roc_curve(y_test, prob)
            pr_precision, pr_recall, _ = precision_recall_curve(y_test, prob)
        except Exception:
            prob = None
            roc_auc = np.nan
            pr_auc = np.nan
            fpr, tpr, pr_precision, pr_recall = None, None, None, None

        output[name] = {
            "pipeline": pipe,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "prob": prob,
            "roc_curve": (fpr, tpr),
            "pr_curve": (pr_recall, pr_precision),
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
            str(pd.util.hash_pandas_object(X_model, index=True).sum()) + "_" + str(pd.util.hash_pandas_object(y_model, index=True).sum()),
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
    f"Industry selected: {selected_industry}. Analysis mode: {MODE_LABELS[analysis_mode]}. "
    f"{analysis_mode_reason} Upload both PY and AY files with a shared customer identifier "
    "to generate a year-over-year customer movement summary."
)
report_table = []
observations = [f"Analysis mode: {MODE_LABELS[analysis_mode]}. {analysis_mode_reason}"]

actions = INDUSTRIES[selected_industry]["actions"] + [
    "Monitor retention and engagement indicators consistently; evaluate interventions against measured outcomes where practical."
]

limitations = (
    "A customer present in PY but absent in AY is classified as 'not observed in AY'. "
    "This does not prove permanent churn. Annual snapshot files cannot establish the exact date or reason a customer stopped purchasing. "
    "If a proxy inactivity target is used for ML, model metrics measure agreement with that business rule, not verified real-world churn."
)


if isinstance(comparison, dict) and "error" not in comparison:

    c = comparison

    report_summary = (
        f"Industry: {selected_industry}. The PY file contains {c['py_count']:,} distinct identity keys "
        f"and the AY file contains {c['ay_count']:,}. {c['retained']:,} keys appear in both files, "
        f"{c['not_observed']:,} PY keys are not observed in AY, and {c['new']:,} keys appear in AY. "
        + (f"Customer retention is not reliably calculable: {c.get('identity_warning')}" if not c.get('retention_comparable', True) else f"The observed identity overlap is {c['retention_rate']:.1f}%; this is a data-presence measure, not confirmed churn.")
    )
    if c.get("identity_warning"):
        observations.append(c["identity_warning"])

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
            "Observed retention / identity overlap",
            "100.0% reference",
            ("Not reliable" if not c.get("retention_comparable", True) else f"{c['retention_rate']:.1f}%"),
            ("Year-coded IDs; overlap cannot establish retention" if not c.get("retention_comparable", True) else "Matched PY customers / PY customers")
        )
    ]

    observations += [
        (f"Identity overlap from PY to AY is {c['retention_rate']:.1f}%; this is not reliable retention because the identifiers appear year-coded." if not c.get("retention_comparable", True) else f"Observed retention from PY to AY is {c['retention_rate']:.1f}% based on shared identity keys; this is presence-based and not confirmed churn."),
        f"{c['not_observed']:,} PY identity keys do not appear in the AY file; validate whether records are inactive, migrated, or missing from the extract.",
        f"{c['new']:,} identity keys are newly observed in AY."
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
                f"Total {spend_col}",
                money(psv),
                money(asv),
                money(asv - psv)
            ))

            observations.append(
                f"{py_spend_col} total changed from {money(psv)} in PY to {money(asv)} in AY."
            )

else:

    observations.append(
        "A PY-versus-AY customer-level comparison is unavailable until both files with a common customer ID are uploaded."
    )


# =============================================================================
# PAGE 1 — EXECUTIVE DASHBOARD
# =============================================================================

if page == "Executive Dashboard":

    render_workspace_header("Executive Dashboard", f"Business intelligence · {selected_industry} · {main_label}", "🛡️")

    st.markdown(
        f'<span class="status-pill">Analysis dataset: {main_label}</span>',
        unsafe_allow_html=True
    )

    st.caption(
        "Metrics are calculated from the uploaded data. "
        "Fields not present in your file are not fabricated."
    )

    if analysis_mode == "exploratory":
        st.info("**Exploratory mode is active.** No verified churn target or defensible inactivity proxy is available. The app will focus on profiles, segments, engagement patterns, descriptive risk indicators, and anomalies—not fabricated churn-model scores.")
    elif analysis_mode == "proxy_target":
        st.warning("**Proxy mode is active.** ML scores reflect agreement with the configured inactivity rule, not verified real-world churn.")
    else:
        st.success("**Actual-target mode is active.** Supervised model evaluation uses the detected target; check the label mapping before interpreting scores.")

    st.markdown(f"**Automatic analysis mode:** `{MODE_LABELS[analysis_mode]}`")
    st.caption(analysis_mode_reason)
    if isinstance(comparison, dict) and "error" not in comparison:
        st.markdown("**Additional workflow:** PY vs AY customer movement (identity matching; absence is not confirmed churn).")

    total_cells = int(main_df.shape[0] * main_df.shape[1])
    missing_cells = int(main_df.isna().sum().sum())
    completeness = 100.0 if total_cells == 0 else (total_cells - missing_cells) / total_cells * 100
    health_cols = st.columns(4)
    health_cols[0].metric("Current dataset rows", f"{len(main_df):,}")
    health_cols[1].metric("Data fields", f"{main_df.shape[1]:,}")
    health_cols[2].metric("Data completeness", f"{completeness:.1f}%", f"{missing_cells:,} missing cells")
    health_cols[3].metric("Numeric fields", f"{len(numeric_columns(main_df)):,}")

    if isinstance(comparison, dict) and "error" not in comparison:

        c = comparison

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("PY records / IDs", f"{c['py_count']:,}")
        k2.metric(
            "AY records / IDs", f"{c['ay_count']:,}",
            f"{c['ay_count']-c['py_count']:+,} records" if c.get("retention_comparable", True) else None
        )
        k3.metric("Identity overlap", f"{c['retained']:,}" if c.get("retention_comparable", True) else "Not comparable")
        k4.metric("Observed retention", f"{c['retention_rate']:.1f}%" if c.get("retention_comparable", True) else "Not calculable")
        if c.get("identity_warning"):
            st.warning(c["identity_warning"])

        if c.get("retention_comparable", True):
            left, right = st.columns(2)
            with left:
                fig = px.pie(
                    names=["Returning in AY", "PY not observed in AY"],
                    values=[c["retained"], c["not_observed"]],
                    hole=.58, color_discrete_sequence=["#34D399", "#FB7185"],
                    title="PY customer movement"
                )
                st.plotly_chart(fig, use_container_width=True)
            with right:
                movement = pd.DataFrame({
                    "Customer group": ["Returning", "Not observed", "New in AY"],
                    "Customers": [c["retained"], c["not_observed"], c["new"]]
                })
                fig = px.bar(movement, x="Customer group", y="Customers", color="Customer group", text_auto=True, color_discrete_sequence=["#22D3EE", "#FB7185", "#A78BFA"])
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Customer movement charts are suppressed because the identity keys appear regenerated or year-coded. The PY and AY record counts are still valid snapshots, but zero ID overlap cannot be interpreted as customer loss, acquisition, or retention.")

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

    financial_metric_title = (
        f"Total {spend_col}" if spend_col else "Financial value not detected"
    )
    metrics = [
        (financial_metric_title, spend_col, "sum"),
        (f"Total recorded volume ({orders_col})" if orders_col else "Total volume not detected", orders_col, "sum"),
        ("Average value / fee" if aov_col else "Average value not detected", aov_col, "mean"),
        (f"Average {satisfaction_col}" if satisfaction_col else "Average satisfaction not detected", satisfaction_col, "mean")
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

    # Industry-aware observed target KPI and segment insight when a target exists.
    if industry_metric_col and industry_metric_col not in {spend_col, orders_col, aov_col, satisfaction_col}:
        metric_values = pd.to_numeric(main_df[industry_metric_col], errors="coerce")
        metric_normalized = re.sub(r"[^a-z0-9]", "", str(industry_metric_col).lower())
        count_like = any(term in metric_normalized for term in ["count", "visits", "bookings", "stays", "appointments", "transactions", "credits"])
        average_like = any(term in metric_normalized for term in ["rate", "score", "rating", "grade", "attendance", "engagement", "usage", "progress", "datausage", "seats"])
        industry_value = metric_values.mean() if average_like and not count_like else metric_values.sum()
        metric_aggregation = "Average" if average_like and not count_like else "Total"
        st.metric(f"{metric_aggregation} · {industry_metric_col}", f"{industry_value:,.2f}" if pd.notna(industry_value) else "N/A")
        st.caption(f"Industry-aware measure: {industry_metric_label}. This is a descriptive numeric summary, not a universal KPI.")

    if target_col:
        churn_binary = map_binary_target_for_display(main_df[target_col], selected_industry)
        if churn_binary is not None and churn_binary.dropna().nunique() == 2:
            churn_rate = safe_percent(churn_binary.eq(1).sum(), churn_binary.notna().sum())
            st.subheader(f"Observed {INDUSTRIES[selected_industry]['retention_language']} indicator")
            churn_kpi, retained_kpi = st.columns(2)
            churn_kpi.metric("Observed attrition / churn", f"{churn_rate:.1f}%", help=f"Mapped from {target_col}; this reflects the dataset's target label.")
            retained_kpi.metric("Observed retained / active", f"{100-churn_rate:.1f}%")
            segment_candidates = [c for c in main_df.columns if c != target_col and main_df[c].nunique(dropna=True) <= 30 and main_df[c].nunique(dropna=True) >= 2 and (main_df[c].dtype == 'object' or str(main_df[c].dtype) == 'category')]
            if segment_candidates:
                segment_col = next((c for c in segment_candidates if re.sub(r'[^a-z0-9]','',str(c).lower()) in {'segment','customersegment','plan','plantype','tenuregroup','region','department','course','program','membershiptier','servicetype'}), segment_candidates[0])
                seg_frame = pd.DataFrame({'Segment': main_df[segment_col].fillna('Missing').astype(str), 'Attrition': churn_binary})
                seg_frame = seg_frame.dropna(subset=['Attrition'])
                segment_rates = seg_frame.groupby('Segment').agg(Customers=('Attrition','size'), AttritionRate=('Attrition','mean')).reset_index()
                segment_rates['AttritionRate'] *= 100
                segment_rates = segment_rates.sort_values('AttritionRate', ascending=False).head(15)
                st.caption(f"Segment analysis using **{segment_col}**. Select a different segment field in Advanced Visualizer for further exploration.")
                fig = px.bar(segment_rates, x='Segment', y='AttritionRate', color='AttritionRate', text=segment_rates['AttritionRate'].map(lambda v:f'{v:.1f}%'), color_continuous_scale=['#34D399','#FBBF24','#FB7185'], title='Highest observed attrition-rate segments')
                fig.update_layout(yaxis_title='Attrition rate (%)')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption(f"Detected target field **{target_col}**, but its values could not be mapped confidently to retained vs attrited. Review labels in Data Quality before interpreting a churn rate.")

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
            color_continuous_scale=[[0, "#111827"], [0.5, active_palette["secondary"]], [1, active_palette["accent"]]]
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
            key=lambda n: (model_results[n]["recall"], model_results[n]["f1"])
        )

        st.success(
            f"Recall-prioritized model in this run: {best_name} "
            f"(Recall {model_results[best_name]['recall']:.3f}; F1 {model_results[best_name]['f1']:.3f}). "
            "Recall is emphasized because missed at-risk customers may be costly; review precision and business capacity too."
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

    render_workspace_header("PY vs AY Comparison", f"Customer movement and year-over-year measures · {selected_industry}", "🔄")

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
    b.metric("AY customer base", f"{c['ay_count']:,}")
    d.metric("Returning customers", f"{c['retained']:,}" if c.get("retention_comparable", True) else "Not comparable")
    e.metric("Observed retention", f"{c['retention_rate']:.2f}%" if c.get("retention_comparable", True) else "Not calculable")
    if c.get("retention_comparable", True):
        st.metric("Not observed in AY", f"{c['not_observed']:,}")
        st.metric("New in AY", f"{c['new']:,}")
    else:
        st.info("The customer movement categories are hidden because the detected identity keys appear year-generated. Only aggregate PY/AY record counts and optional aggregate measure comparisons are meaningful here until a stable cross-year ID is provided.")
    if c.get("identity_warning"):
        st.warning(c["identity_warning"])

    st.caption(
        f"Matched using {c.get('match_field', 'detected identity field')} ({c.get('match_kind', 'identity')}). "
        "Not observed in AY is a data-presence classification, not proof that the customer permanently churned."
    )
    if c.get("match_kind") == "id" and py_id_col != ay_id_col:
        st.warning("PY and AY use differently named identity fields. Confirm that both fields represent the same stable customer/entity identity before interpreting the match results.")
    if c.get("py_duplicates", 0) or c.get("ay_duplicates", 0):
        st.warning(
            f"Duplicate identity keys detected: PY {c.get('py_duplicates', 0):,}, "
            f"AY {c.get('ay_duplicates', 0):,}. Counts represent distinct normalized keys; "
            "review duplicates before interpreting customer movement."
        )

    chart1, chart2 = st.columns(2)

    with chart1:

        fig = go.Figure(data=[
            go.Bar(
                name="PY",
                x=["Customer base"],
                y=[c["py_count"]],
                marker_color=active_palette["secondary"]
            ),
            go.Bar(
                name="AY",
                x=["Customer base"],
                y=[c["ay_count"]],
                marker_color=active_palette["accent"]
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
        if c.get("retention_comparable", True):
            fig = px.funnel(
                pd.DataFrame({"Stage": ["PY identity keys", "Returning in AY"], "Customers": [c["py_count"], c["retained"]]}),
                x="Customers", y="Stage", title="PY customer retention funnel"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Retention funnel unavailable: year-coded IDs have no verified cross-year match.")

    if c.get("retention_comparable", True):
        st.subheader("Movement summary")
        movement_df = pd.DataFrame([
            ["Returning customers", c["retained"], "Customer ID exists in both years"],
            ["Not observed in AY", c["not_observed"], "Customer ID exists in PY but not AY"],
            ["Newly observed in AY", c["new"], "Customer ID exists in AY but not PY"]
        ], columns=["Segment", "Customer count", "Definition"])
        st.dataframe(movement_df, use_container_width=True, hide_index=True)

    st.subheader("Optional measure comparison")

    common_measures = []

    for candidates, label in [
        (
            financial_candidates_by_industry[selected_industry] + ["TotalSpend", "Total_Spend", "Revenue", "Sales", "TotalRevenue", "Amount"],
            "Financial value"
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

    render_workspace_header("Advanced Visualizer", f"Explore distributions, relationships and segments · {selected_industry}", "📈")

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
                color_discrete_sequence=active_palette["chart"]
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
            color_continuous_scale=[[0, "#111827"], [0.5, active_palette["secondary"]], [1, active_palette["accent"]]],
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

    render_workspace_header("Data Quality & Ingestion", f"Profile, validate and prepare your dataset · {selected_industry}", "📥")

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

    render_workspace_header("ML Model Evaluation", f"Model readiness, evaluation and explainability · {selected_industry}", "🤖")

    st.write("The workspace adapts automatically to the dataset's analytical mode.")
    st.markdown(f"**Current mode:** `{MODE_LABELS[analysis_mode]}`")
    st.caption(analysis_mode_reason)

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

        model_comparison = pd.DataFrame([
            {"Model": name, "Accuracy": result["accuracy"], "Precision": result["precision"], "Recall": result["recall"], "F1": result["f1"], "ROC-AUC": result.get("roc_auc", np.nan), "PR-AUC": result.get("pr_auc", np.nan)}
            for name, result in model_results.items()
        ]).sort_values(["Recall", "F1"], ascending=False)
        st.subheader("Model comparison · recall prioritized")
        st.dataframe(model_comparison.style.format({c:"{:.3f}" for c in model_comparison.columns if c != "Model"}), use_container_width=True, hide_index=True)
        st.caption("Recall is emphasized for churn-risk screening. Choose thresholds and outreach capacity with business stakeholders; a high recall model can also produce more false positives.")

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

        a_auc, a_pr = st.columns(2)
        a_auc.metric("ROC-AUC", f"{res['roc_auc']:.3f}" if pd.notna(res.get("roc_auc")) else "N/A")
        a_pr.metric("PR-AUC", f"{res['pr_auc']:.3f}" if pd.notna(res.get("pr_auc")) else "N/A")

        left, right = st.columns(2)

        with left:

            fig = px.imshow(
                res["cm"],
                text_auto=True,
                x=["Predicted 0", "Predicted 1"],
                y=["Actual 0", "Actual 1"],
                color_continuous_scale=[[0, "#111827"], [0.45, active_palette["secondary"]], [1, active_palette["accent"]],],
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

        if res.get("prob") is not None:
            curve_left, curve_right = st.columns(2)
            with curve_left:
                roc_fpr, roc_tpr = res["roc_curve"]
                roc_fig = go.Figure()
                roc_fig.add_trace(go.Scatter(x=roc_fpr, y=roc_tpr, mode="lines", name=f"ROC · AUC {res['roc_auc']:.3f}", line={"color": active_palette["accent"], "width": 3}))
                roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline", line={"color": "#64748B", "dash": "dash"}))
                roc_fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate", template="plotly_dark")
                st.plotly_chart(roc_fig, use_container_width=True)
            with curve_right:
                pr_recall_values, pr_precision_values = res["pr_curve"]
                pr_fig = go.Figure()
                pr_fig.add_trace(go.Scatter(x=pr_recall_values, y=pr_precision_values, mode="lines", name=f"PR · AP {res['pr_auc']:.3f}", line={"color": active_palette["secondary"], "width": 3}))
                pr_fig.update_layout(title="Precision–Recall curve", xaxis_title="Recall", yaxis_title="Precision", template="plotly_dark")
                st.plotly_chart(pr_fig, use_container_width=True)

        st.subheader("Test-set churn risk predictions")
        prediction_frame = pd.DataFrame({
            "Actual label": res["y_test"].values,
            "Predicted label": res["pred"],
            "Predicted attrition probability": res["prob"] if res.get("prob") is not None else np.nan,
        }, index=res["X_test"].index)
        if id_col and id_col in main_df.columns:
            prediction_frame.insert(0, "Customer / entity ID", main_df.loc[prediction_frame.index, id_col].astype(str).values)
        prediction_frame["Risk band"] = prediction_frame["Predicted attrition probability"].apply(
            lambda p: "High" if pd.notna(p) and p >= .70 else ("Medium" if pd.notna(p) and p >= .40 else ("Low" if pd.notna(p) else "Unavailable"))
        )
        high_risk_predictions = prediction_frame.sort_values("Predicted attrition probability", ascending=False).head(25)
        st.dataframe(high_risk_predictions, use_container_width=True, hide_index=True)
        add_download_button(
            "⬇️ Download test-set risk predictions (CSV)",
            prediction_frame.sort_values("Predicted attrition probability", ascending=False).to_csv(index=False).encode("utf-8"),
            "Churn_Guardian_Test_Risk_Predictions.csv",
            "text/csv"
        )
        st.caption("Probabilities are generated from the held-out test split. Risk bands are illustrative thresholds (40% and 70%) and should be calibrated before operational use.")

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
                color_continuous_scale=[[0, "#111827"], [0.45, active_palette["secondary"]], [1, active_palette["accent"]]]
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
        st.warning(model_error or "Model evaluation is unavailable.")
        if analysis_mode == "exploratory":
            st.subheader("Exploratory analytics instead of fabricated ML metrics")
            st.info("This dataset does not contain a defensible supervised target. Explore it with descriptive analytics below.")
            numeric = main_df.select_dtypes(include=np.number).columns.tolist()
            categorical = [c for c in main_df.columns if c not in numeric and main_df[c].nunique(dropna=True) <= 30 and main_df[c].nunique(dropna=True) >= 2]
            ex1, ex2, ex3 = st.columns(3)
            ex1.metric("Records", f"{len(main_df):,}")
            ex2.metric("Numeric fields", f"{len(numeric):,}")
            ex3.metric("Categorical fields", f"{len(categorical):,}")
            if numeric:
                selected_numeric = st.selectbox("Explore a numeric field", numeric, key="explore_numeric_mode")
                values = pd.to_numeric(main_df[selected_numeric], errors="coerce").dropna()
                if not values.empty:
                    exa, exb, exc = st.columns(3)
                    exa.metric("Mean", f"{values.mean():,.2f}")
                    exb.metric("Median", f"{values.median():,.2f}")
                    exc.metric("Std. deviation", f"{values.std():,.2f}")
                    hist = px.histogram(main_df, x=selected_numeric, nbins=30, color_discrete_sequence=[active_palette["accent"]], title=f"Distribution · {selected_numeric}")
                    st.plotly_chart(hist, use_container_width=True)
            if categorical:
                selected_category = st.selectbox("Explore a categorical field", categorical, key="explore_category_mode")
                counts = main_df[selected_category].fillna("Missing").astype(str).value_counts().head(20).reset_index()
                counts.columns = [selected_category, "Records"]
                cat_fig = px.bar(counts, x=selected_category, y="Records", color="Records", color_continuous_scale=[[0,"#172033"],[0.5,active_palette["secondary"]],[1,active_palette["accent"]]], title=f"Top categories · {selected_category}")
                st.plotly_chart(cat_fig, use_container_width=True)
            if len(numeric) >= 2:
                st.subheader("Numeric correlation overview")
                corr = main_df[numeric].corr(numeric_only=True)
                corr_fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1, title="Correlation matrix")
                st.plotly_chart(corr_fig, use_container_width=True)
        else:
            st.markdown("**To enable supervised evaluation, verify the detected target or supported inactivity field/date.** The app will not invent Accuracy, Precision, Recall, or F1 scores.")


# =============================================================================
# PAGE 6 — STAKEHOLDER REPORT
# =============================================================================

elif page == "Stakeholder Report":

    render_workspace_header("Stakeholder Report", f"Evidence-based summary and recommendations · {selected_industry}", "📑")

    st.write(
        f"A concise, evidence-based report generated from the uploaded datasets for **{selected_industry}**."
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

    render_workspace_header("Business Assistant", f"Ask questions about the uploaded dataset · {selected_industry}", "💬")

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

                if comparison.get("retention_comparable", True):
                    answer_parts.append(f"The observed identity overlap is **{comparison['retention_rate']:.1f}%**. {comparison['retained']:,} IDs appear in both PY and AY, out of {comparison['py_count']:,} PY IDs. This is a presence-based metric, not confirmed churn.")
                else:
                    answer_parts.append(f"Retention cannot be reliably calculated from the current identifiers. {comparison.get('identity_warning', '')}")

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
                    f"Total recorded **{spend_col}** in the current dataset is **{money(value)}**."
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
                    f"Total recorded **{orders_col}** in the current dataset is **{value:,.0f}**."
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

        if any(word in q for word in ["mode", "analysis mode", "exploratory", "proxy"]):
            answer_parts.append(f"The current automatic analysis mode is **{MODE_LABELS[analysis_mode]}**. {analysis_mode_reason}")

        if any(word in q for word in ["industry", "healthcare", "banking", "education", "hospitality", "telecom", "saas"]):
            answer_parts.append(
                f"The selected industry is **{selected_industry}**. In this context, the app interprets retention as **{INDUSTRIES[selected_industry]['retention_language']}**. "
                f"Industry-specific risk examples include {INDUSTRIES[selected_industry]['examples']}."
            )

        if any(word in q for word in ["threshold", "inactivity days", "inactive for", "days inactive"]):
            answer_parts.append(
                f"The current inactivity threshold is **{inactivity_days} days**. "
                "This threshold is used only when an inactivity-days field is available and the dataset does not provide a usable observed target. "
                "The resulting label is a rule-based proxy, not verified churn."
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
                    key=lambda n: (model_results[n]["recall"], model_results[n]["f1"])
                )

                answer_parts.append(
                    f"The recall-prioritized model in the current test split is **{best}** "
                    f"(Recall {model_results[best]['recall']:.3f}; F1 {model_results[best]['f1']:.3f}). "
                    f"Label basis: {model_note}. "
                    "If the label is a proxy, this is not verified churn accuracy."
                )

            else:

                answer_parts.append(
                    "Model metrics are unavailable because the dataset has no valid binary target or supported inactivity proxy."
                )

        if any(word in q for word in ["highest churn", "highest attrition", "highest dropout", "most at risk", "risk segment", "segment with highest"]):
            if target_col:
                mapped_target = map_binary_target_for_display(main_df[target_col], selected_industry)
                candidate_segments = [col for col in main_df.columns if col != target_col and main_df[col].nunique(dropna=True) <= 30 and main_df[col].nunique(dropna=True) >= 2 and (main_df[col].dtype == 'object' or str(main_df[col].dtype) == 'category')]
                if mapped_target is not None and candidate_segments:
                    segment_col = next((col for col in candidate_segments if re.sub(r'[^a-z0-9]','',str(col).lower()) in {'segment','customersegment','plan','plantype','tenuregroup','region','department','course','program','membershiptier','servicetype'}), candidate_segments[0])
                    seg = pd.DataFrame({'Segment':main_df[segment_col].fillna('Missing').astype(str),'Target':mapped_target}).dropna(subset=['Target'])
                    rates=seg.groupby('Segment').agg(Customers=('Target','size'), Rate=('Target','mean')).reset_index()
                    rates['Rate']*=100
                    rates=rates.sort_values('Rate',ascending=False)
                    if len(rates):
                        top=rates.iloc[0]
                        answer_parts.append(f"Using **{segment_col}** as the segment field, **{top['Segment']}** has the highest observed attrition/churn rate at **{top['Rate']:.1f}%** ({int(top['Customers']):,} records). This is calculated from target column **{target_col}** and is descriptive of the uploaded data, not causal.")
                else:
                    answer_parts.append("A highest-churn segment cannot be calculated reliably: a binary attrition target and a suitable categorical segment field are required.")
            else:
                answer_parts.append("No recognized churn/attrition target is available. I cannot claim a segment has the highest observed churn rate without a target; I can still describe customer movement or a supported inactivity proxy.")

        if any(word in q for word in ["reason", "reasons", "factor", "factors", "why churn", "contributing"]):
            if model_results:
                best_factor_model = max(model_results, key=lambda n: (model_results[n]["recall"], model_results[n]["f1"]))
                estimator = model_results[best_factor_model]["pipeline"].named_steps["model"]
                try:
                    importances = estimator.feature_importances_
                    names = model_results[best_factor_model]["pipeline"].named_steps["prep"].get_feature_names_out()
                    top_features = pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False).head(5)
                    factor_text = ", ".join(f"**{str(row.feature).replace('num__','').replace('cat__','')}** ({row.importance:.3f})" for row in top_features.itertuples())
                    answer_parts.append(f"Top model-associated features from **{best_factor_model}**: {factor_text}. These are predictive associations in this dataset—not proof of causes. Validate them with domain experts before acting.")
                except Exception:
                    answer_parts.append("Feature importance is unavailable for the current model configuration; I cannot substantiate contributing factors from this run.")
            else:
                answer_parts.append("No trained model is available to identify model-associated factors. I can only describe available fields and descriptive patterns.")

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
                f"I can explain customer movement, observed retention, new customers, business measures, data quality, ML evaluation and the report for **{selected_industry}**. "
                "I only answer from fields available in the uploaded data and clearly identify proxies or unavailable metrics."
            )

        if not answer_parts:

            answer_parts.append(
                f"I can answer questions about {INDUSTRIES[selected_industry]['retention_language']}, customer/entity movement, available measures, data quality, model evaluation, feature associations and industry recommendations. "
                "Try asking a specific question about the uploaded data."
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
    f"🛡️ Churn Guardian | {selected_industry} analytics | Business analytics support tool | "
    "Validate business decisions with source-system owners."
)
