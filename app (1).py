import io
import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st

from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Conservation Behavior Analytics", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #f4f7fb; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1300px; }
    h1, h2, h3, h4 { color: #0f3352; font-family: 'Segoe UI', sans-serif; letter-spacing: 0.2px; }
    div[data-testid="stTabs"] button[role="tab"] {
        background-color: #ffffff;
        border: 1px solid #c9d6e3;
        border-radius: 6px;
        padding: 9px 18px;
        margin-right: 6px;
        color: #0f3352;
        font-weight: 600;
        font-size: 0.86rem;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background-color: #0f3352;
        color: #ffffff;
        border-color: #0f3352;
        box-shadow: 0 2px 8px rgba(15,51,82,0.25);
    }
    div[data-testid="stTabs"] div[role="tablist"] { border-bottom: none; gap: 4px; }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none; }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dbe4ee;
        border-radius: 10px;
        padding: 12px 14px;
        box-shadow: 0 1px 3px rgba(15,51,82,0.06);
    }
    div[data-testid="stMetricValue"] { color: #0f3352; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #5b7387; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.6px; }
    .chip {
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        background: #e8eef5; color: #0f3352; font-size: 0.72rem;
        font-weight: 600; margin-right: 6px; margin-bottom: 4px;
    }
    .open-card {
        background: #ffffff; border-left: 4px solid #2e86c1;
        border-radius: 6px; padding: 11px 14px; margin-bottom: 8px;
        color: #24435f; font-size: 0.82rem; line-height: 1.5;
        box-shadow: 0 1px 2px rgba(15,51,82,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

AWARENESS_COLS = ["Q6", "Q7", "Q8", "Q9"]
ATTITUDE_COLS = ["Q10", "Q11", "Q12", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18"]
BEHAVIOR_COLS = ["Q19", "Q20", "Q21", "Q22", "Q23", "Q24", "Q25"]
PARTICIPATION_COLS = ["Q26", "Q27", "Q28", "Q29", "Q30"]
REVERSE_COLS = ["Q16", "Q17", "Q18"]
TARGET_COL = "Q31"
SECTIONS = ["AWARENESS_SCORE", "ATTITUDE_SCORE", "BEHAVIOR_SCORE", "PARTICIPATION_SCORE"]
SECTION_LABELS = {
    "AWARENESS_SCORE": "Awareness",
    "ATTITUDE_SCORE": "Attitude",
    "BEHAVIOR_SCORE": "Behavior",
    "PARTICIPATION_SCORE": "Participation",
}

RAW_COL_MAP = {
    "1. Edad / Age": "AGE",
    "2. Sekso / Gender": "GENDER",
    "3. Lebel sa Edukasyon / Educational Attainment": "EDUCATION",
    "4. Trabaho / Occupation": "OCCUPATION",
    "5. Distansya gikan sa Agusan Marsh": "DISTANCE",
}

Q_LABEL_MAP = {
    "6.I am aware that Agusan Marsh is a protected wildlife sanctuary.": "Q6",
    "7. Nasabtan nako ang ekologikal nga kamahinungdanon sa Agusan Marsh.": "Q7",
    "8. Nahibalo ko sa mga programa sa konserbasyon nga gihimo sa lugar.": "Q8",
    "9. Pamilyar ako sa mga balaod sa kalikopan nga nagpanalipod sa mga wetland.": "Q9",
    "10. Nakadawat ako og impormasyon bahin sa konserbasyon gikan sa lain-laing opisyal nga mga tinubdan.": "Q10",
    "11. Mahinungdanon ang pagpanalipod sa Agusan Marsh alang sa mga umaabot nga henerasyon.": "Q11",
    "12. Ang mga miyembro sa komunidad kinahanglan nga aktibong suportahan ang mga paningkamot sa konserbasyon.": "Q12",
    "13. Environmental protection should be a shared responsibility.": "Q13",
    "14. Ako mismo ang responsable sa pagpanalipod sa natural nga mga kahinguhaan.": "Q14",
    "15. Conservation activities benefit the local community.": "Q15",
    "16. Conservation efforts do not affect my daily life.": "Q16",
    "17. Protecting Agusan Marsh is a waste of time and money.": "Q17",
    "18. Restrictions are unreasonable and harmful to livelihoods.": "Q18",
    "19. Moapil ko sa mga aktibidad sa konserbasyon sa kalikopan.": "Q19",
    "20. Motambong ko sa environmental seminars o awareness programs.": "Q20",
    "21. Mosumbong ko sa environmental violations o problema sa among lugar.": "Q21",
    "22. Nagabuhat ko ug eco-friendly habits (proper waste disposal, avoiding pollution).": "Q22",
    "23. Mogamit ko ug eco-friendly products kung posible.": "Q23",
    "24. Nagashare ko ug impormasyon bahin sa konserbasyon sa akong pamilya o silingan.": "Q24",
    "25. Nagalikay ko sa mga binuhatan nga makadaot sa wildlife ug kinaiyahan.": "Q25",
    "26. Andam ako nga mag-volunteer sa mga aktibidad sa konserbasyon.": "Q26",
    "27. Andam ako nga moapil sa pagtanom og kahoy.": "Q27",
    "28. Suportahan nako ang community initiatives.": "Q28",
    "29. Motambong ko og workshops.": "Q29",
    "30. Awhagon nako ang uban nga moapil.": "Q30",
    "31. Posibilidad sa pag-apil sa umaabot nga mga programa sa pagpanalipod sa kinaiyahan.": "Q31",
    "34. Ideas or suggestions to protect Agusan Marsh?": "Q34",
    "35. Problems observed in Agusan Marsh?": "Q35",
    "36. Why do others not participate? What encourages participation?": "Q36",
}

LIKERT_MAP = {
    "strongly agree": 5, "agree": 4, "neutral": 3, "disagree": 2, "strongly disagree": 1,
    "always": 5, "often": 4, "sometimes": 3, "rarely": 2, "never": 1,
}

TARGET_SINGLE_MAP = {
    "very likely": "High", "likely": "High",
    "neutral": "Low/Neutral", "unlikely": "Low/Neutral", "very unlikely": "Low/Neutral",
}

VALID_DISTANCES = {"within marsh area", "1-5 km", "6-10 km", "more than 10 km", "less than 1 km"}

RISK_BANDS = [
    ("Not at Risk", 3.81, 5.00, "#1e8449"),
    ("Low Risk", 3.01, 3.80, "#2874a6"),
    ("Moderate Risk", 2.21, 3.00, "#b9770e"),
    ("High Risk", 1.00, 2.20, "#a93226"),
]

CLUSTER_NAME_BY_RANK = [
    "Active Conservationists", "Emerging Participants",
    "Passive Bystanders", "Non-Participants / High Risk",
]

DISTANCE_ORDER = ["Within marsh area", "Less than 1 km", "1-5 km", "6-10 km", "More than 10 km"]
PALETTE = ["#0f3352", "#2e86c1", "#48c9b0", "#f39c12", "#a93226", "#7d3c98", "#1e8449"]

def clean_text(value):
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).encode("ascii", "ignore").decode()
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    return re.sub(r"\s+", " ", text).strip()

def normalize_label(value):
    return clean_text(value).lower()

def likert_to_num(value):
    key = normalize_label(value)
    if not key:
        return np.nan
    return LIKERT_MAP.get(key, np.nan)

def risk_band(composite):
    for name, low, high, color in RISK_BANDS:
        if low <= composite <= high:
            return name, color
    return "Unclassified", "#7f8c8d"

def parse_upload(uploaded):
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded, sheet_name=0)

def preprocess(raw):
    df = raw.copy()
    df.columns = [clean_text(c) for c in df.columns]
    renaming = {}
    for col in df.columns:
        if col in RAW_COL_MAP:
            renaming[col] = RAW_COL_MAP[col]
            continue
        for fragment, target in Q_LABEL_MAP.items():
            if fragment in col:
                renaming[col] = target
                break
    df = df.rename(columns=renaming)
    df = df.loc[:, ~df.columns.duplicated()]

    for col in ["AGE", "GENDER", "EDUCATION", "OCCUPATION", "DISTANCE"]:
        if col not in df.columns:
            df[col] = np.nan
    for col in AWARENESS_COLS + ATTITUDE_COLS + BEHAVIOR_COLS + PARTICIPATION_COLS + [TARGET_COL]:
        if col not in df.columns:
            df[col] = np.nan
    for col in ["Q34", "Q35", "Q36"]:
        if col not in df.columns:
            df[col] = ""

    df = df.dropna(subset=["AGE"], how="all")
    if "Timestamp" in df.columns:
        df = df[df["Timestamp"].apply(clean_text).str.lower() != "d nnnnn"]

    likert_cols = AWARENESS_COLS + ATTITUDE_COLS + BEHAVIOR_COLS + PARTICIPATION_COLS
    for col in likert_cols:
        df[col] = df[col].apply(likert_to_num)

    missing_share = df[likert_cols].isna().mean(axis=1)
    df = df[missing_share <= 0.20].copy()
    df = df.drop_duplicates()

    for col in REVERSE_COLS:
        df[col] = df[col].apply(lambda v: 6 - v if pd.notna(v) else np.nan)

    df["AWARENESS_SCORE"] = df[AWARENESS_COLS].mean(axis=1)
    df["ATTITUDE_SCORE"] = df[ATTITUDE_COLS].mean(axis=1)
    df["BEHAVIOR_SCORE"] = df[BEHAVIOR_COLS].mean(axis=1)
    df["PARTICIPATION_SCORE"] = df[PARTICIPATION_COLS].mean(axis=1)

    df["DISTANCE_NORM"] = df["DISTANCE"].apply(normalize_label)
    df = df[df["DISTANCE_NORM"].isin(VALID_DISTANCES)].copy()

    for col in ["GENDER", "EDUCATION", "OCCUPATION", "AGE", "DISTANCE"]:
        df[col] = df[col].apply(clean_text)

    df["DISTANCE_ORDERED"] = df["DISTANCE_NORM"].map({
        "within marsh area": "Within marsh area",
        "less than 1 km": "Less than 1 km",
        "1-5 km": "1-5 km",
        "6-10 km": "6-10 km",
        "more than 10 km": "More than 10 km",
    })

    df["TARGET_RAW"] = df[TARGET_COL].apply(clean_text)
    df["TARGET_LABEL"] = df["TARGET_RAW"].apply(normalize_label).map(TARGET_SINGLE_MAP)
    return df

def build_cluster_matrix(df):
    features = df[SECTIONS].copy()
    demo = pd.get_dummies(df[["AGE", "GENDER", "EDUCATION", "OCCUPATION", "DISTANCE_NORM"]], drop_first=False)
    matrix = pd.concat([features, demo], axis=1)
    matrix = matrix.fillna(matrix.mean(numeric_only=True)).fillna(0)
    scaler = StandardScaler()
    return scaler.fit_transform(matrix)

def k_search(scaled, k_min=2, k_max=6):
    rows, inertias = [], {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(scaled)
        inertias[k] = km.inertia_
        sil = silhouette_score(scaled, labels) if len(set(labels)) > 1 else np.nan
        rows.append({
            "k": k,
            "Inertia": round(km.inertia_, 3),
            "Silhouette": round(sil, 4) if pd.notna(sil) else None,
        })
    return pd.DataFrame(rows), inertias

def run_kmeans(scaled, k):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    return km.fit_predict(scaled)

def profile_clusters(df, labels):
    frame = df.copy()
    frame["CLUSTER"] = labels
    profile = frame.groupby("CLUSTER")[SECTIONS].mean()
    profile["COMPOSITE"] = profile.mean(axis=1)
    profile["N"] = frame.groupby("CLUSTER").size()
    profile["SHARE"] = (profile["N"] / profile["N"].sum() * 100).round(2)
    profile = profile.sort_values("COMPOSITE", ascending=False).reset_index(drop=True)
    profile["RISK"] = profile["COMPOSITE"].apply(lambda v: risk_band(v)[0])
    profile["COLOR"] = profile["COMPOSITE"].apply(lambda v: risk_band(v)[1])
    profile["SEGMENT"] = [
        CLUSTER_NAME_BY_RANK[i] if i < len(CLUSTER_NAME_BY_RANK) else f"Segment {i+1}"
        for i in range(len(profile))
    ]
    return frame, profile

def classify(frame):
    usable = frame[frame["TARGET_LABEL"].notna()].copy()
    demo = pd.get_dummies(usable[["AGE", "GENDER", "EDUCATION", "OCCUPATION", "DISTANCE_NORM"]])
    X = pd.concat([usable[SECTIONS].reset_index(drop=True), demo.reset_index(drop=True)], axis=1).fillna(0)
    y = usable["TARGET_LABEL"].map({"High": 1, "Low/Neutral": 0}).reset_index(drop=True)

    if y.nunique() < 2 or len(y) < 20:
        return None, None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    models = {
        "Decision Tree": DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=6, min_samples_leaf=3),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    rows, roc_data, cm_data, preds = [], {}, {}, {}
    rf_model = None
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, pred), 4),
            "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, pred, zero_division=0), 4),
            "F1": round(f1_score(y_test, pred, zero_division=0), 4),
            "AUC": round(auc, 4),
        })
        cm_data[name] = confusion_matrix(y_test, pred)
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_data[name] = (fpr, tpr, auc)
        preds[name] = (y_test.reset_index(drop=True), pred)
        if name == "Random Forest":
            rf_model = model

    metrics = pd.DataFrame(rows)
    fi = None
    if rf_model is not None:
        fi = pd.DataFrame({
            "Feature": X.columns,
            "Importance": np.round(rf_model.feature_importances_, 5),
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        fi["Cumulative"] = fi["Importance"].cumsum().round(4)
    return metrics, fi, roc_data, cm_data, preds

def demographic_tests(frame):
    usable = frame.dropna(subset=["PARTICIPATION_SCORE"]).copy()
    results = []
    for group_col, label in [("AGE", "Age"), ("EDUCATION", "Education"), ("OCCUPATION", "Occupation")]:
        groups = [g["PARTICIPATION_SCORE"].dropna().values for _, g in usable.groupby(group_col) if len(g) > 1]
        if len(groups) > 1:
            f_stat, p_val = stats.f_oneway(*groups)
            results.append({"Test": f"ANOVA: {label} vs Participation", "Statistic": round(f_stat, 4), "p-value": round(p_val, 4)})
    
    gender_norm = usable["GENDER"].apply(normalize_label)
    female = gender_norm.str.startswith("babaye")
    male = gender_norm.str.startswith("lalaki")
    if female.any() and male.any():
        mask = female | male
        binary = female[mask].astype(int)
        scores = usable.loc[mask, "PARTICIPATION_SCORE"]
        if len(binary) > 2:
            r, p_val = stats.pointbiserialr(binary, scores)
            results.append({"Test": "Point-biserial: Gender vs Participation", "Statistic": round(r, 4), "p-value": round(p_val, 4)})
            
    distance_codes = usable["DISTANCE_NORM"].map({
        "within marsh area": 1, "less than 1 km": 2, "1-5 km": 3,
        "6-10 km": 4, "more than 10 km": 5,
    })
    mask = distance_codes.notna() & usable["PARTICIPATION_SCORE"].notna()
    if mask.sum() > 3:
        rho, p_val = stats.spearmanr(distance_codes[mask], usable.loc[mask, "PARTICIPATION_SCORE"])
        results.append({"Test": "Spearman: Distance vs Participation", "Statistic": round(rho, 4), "p-value": round(p_val, 4)})
        
    out = pd.DataFrame(results)
    if not out.empty:
        out["Significance"] = out["p-value"].apply(lambda p: "p < 0.01" if p < 0.01 else ("p < 0.05" if p < 0.05 else "n.s."))
    return out

st.sidebar.header("Data Source")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded is None:
    st.info("Upload a dataset to view results.")
    st.stop()

try:
    raw = parse_upload(uploaded)
except Exception:
    st.info("The uploaded file could not be read. Check the file format.")
    st.stop()

df = preprocess(raw)
if df.empty:
    st.info("No valid records remain after applying the study's data rules.")
    st.stop()

st.title("Community Conservation Behavior Analytics")
st.markdown(
    f"<span class='chip'>{len(df)} records</span>"
    f"<span class='chip'>{df['AGE'].replace('', np.nan).nunique()} age groups</span>"
    f"<span class='chip'>{df['DISTANCE'].replace('', np.nan).nunique()} distance strata</span>"
    f"<span class='chip'>{int(df['TARGET_LABEL'].notna().sum())} labeled outcomes</span>",
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Respondents", len(df))
k2.metric("Awareness", round(df["AWARENESS_SCORE"].mean(), 2), round(df["AWARENESS_SCORE"].std(), 2))
k3.metric("Attitude", round(df["ATTITUDE_SCORE"].mean(), 2), round(df["ATTITUDE_SCORE"].std(), 2))
k4.metric("Behavior", round(df["BEHAVIOR_SCORE"].mean(), 2), round(df["BEHAVIOR_SCORE"].std(), 2))
k5.metric("Participation", round(df["PARTICIPATION_SCORE"].mean(), 2), round(df["PARTICIPATION_SCORE"].std(), 2))

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Respondent Profile", "Conservation Measures", "Behavioral Segments",
    "Predictive Performance", "Predictors", "Open Responses",
])

with tab1:
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        age_counts = df["AGE"].replace("", np.nan).dropna().value_counts().reset_index()
        age_counts.columns = ["Age Group", "Count"]
        if not age_counts.empty:
            age_counts["Share"] = (age_counts["Count"] / age_counts["Count"].sum() * 100).round(1)
            fig = px.bar(age_counts, x="Age Group", y="Count", color="Share", color_continuous_scale="Blues", text="Count", title="Age Distribution")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        gender_counts = df["GENDER"].replace("", np.nan).dropna().value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]
        if not gender_counts.empty:
            fig = px.pie(gender_counts, names="Gender", values="Count", hole=0.55, color_discrete_sequence=["#0f3352", "#2e86c1", "#5dade2", "#a9cce3"], title="Gender Distribution")
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
    with c3:
        dist_counts = df["DISTANCE_ORDERED"].replace("", np.nan).dropna().value_counts().reset_index()
        dist_counts.columns = ["Distance", "Count"]
        if not dist_counts.empty:
            dist_counts["Order"] = dist_counts["Distance"].apply(lambda v: DISTANCE_ORDER.index(v) if v in DISTANCE_ORDER else 99)
            dist_counts = dist_counts.sort_values("Order")
            fig = go.Figure(go.Bar(x=dist_counts["Count"], y=dist_counts["Distance"], orientation="h", marker_color="#2e86c1", text=dist_counts["Count"]))
            fig.update_layout(title="Distance from Marsh", yaxis_title="", xaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

    c4, c5 = st.columns([1.6, 1])
    with c4:
        edu_counts = df["EDUCATION"].replace("", np.nan).dropna().value_counts().reset_index().head(8)
        edu_counts.columns = ["Education", "Count"]
        if not edu_counts.empty:
            edu_counts["Share"] = (edu_counts["Count"] / df["EDUCATION"].replace("", np.nan).notna().sum() * 100).round(1)
            fig = px.bar(edu_counts, x="Count", y="Education", orientation="h", color="Share", color_continuous_scale="Teal", text="Count", title="Educational Attainment")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
    with c5:
        occ_counts = df["OCCUPATION"].replace("", np.nan).dropna().value_counts().reset_index().head(8)
        occ_counts.columns = ["Occupation", "Count"]
        if not occ_counts.empty:
            fig = px.treemap(occ_counts, path=["Occupation"], values="Count", color="Count", color_continuous_scale="Blues", title="Occupation Composition")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='panel-title'>Demographic Cross-Tabulation</div>", unsafe_allow_html=True)
    cross = pd.crosstab(df["DISTANCE_ORDERED"].replace("", np.nan), df["GENDER"].replace("", np.nan))
    if not cross.empty:
        st.dataframe(cross, use_container_width=True)

    st.markdown("<div class='panel-title'>Target Distribution by Distance</div>", unsafe_allow_html=True)
    target_dist = pd.crosstab(df["DISTANCE_ORDERED"].replace("", np.nan), df["TARGET_LABEL"].replace("", np.nan), normalize="index").mul(100).round(1)
    if not target_dist.empty:
        fig = px.imshow(target_dist, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Likelihood", y="Distance", color="Percent"))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        summary = pd.DataFrame({
            "Construct": [SECTION_LABELS[c] for c in SECTIONS],
            "Mean": [round(df[c].mean(), 3) for c in SECTIONS],
            "SD": [round(df[c].std(), 3) for c in SECTIONS],
            "Min": [round(df[c].min(), 2) for c in SECTIONS],
            "Max": [round(df[c].max(), 2) for c in SECTIONS],
        })
        st.dataframe(summary, use_container_width=True)
    with c2:
        corr = df[SECTIONS].corr().round(3)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Inter-Construct Correlation", x=[SECTION_LABELS[c] for c in SECTIONS], y=[SECTION_LABELS[c] for c in SECTIONS])
        st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    for i, col in enumerate(SECTIONS):
        fig.add_trace(go.Violin(y=df[col].dropna(), name=SECTION_LABELS[col], box_visible=True, meanline_visible=True, fillcolor=PALETTE[i], line_color=PALETTE[i], opacity=0.7))
    fig.update_layout(title="Score Distributions", yaxis_title="Score", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        target_counts = df["TARGET_LABEL"].replace("", np.nan).dropna().value_counts().reset_index()
        target_counts.columns = ["Likelihood", "Count"]
        if not target_counts.empty:
            target_counts["Share"] = (target_counts["Count"] / target_counts["Count"].sum() * 100).round(1)
            fig = px.pie(target_counts, names="Likelihood", values="Count", hole=0.5, color="Likelihood", color_discrete_map={"High": "#1e8449", "Low/Neutral": "#b9770e"}, title="Binarized Future Participation")
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.box(df, x="DISTANCE_ORDERED", y="PARTICIPATION_SCORE", color="DISTANCE_ORDERED", color_discrete_sequence=PALETTE, title="Participation by Distance")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Participation Score")
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Section scores aggregate the Likert items. Reverse-coded items are inverted before averaging.")

with tab3:
    scaled = build_cluster_matrix(df)
    k_table, inertias = k_search(scaled)

    best_k = 4 

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        fig = go.Figure(go.Scatter(x=list(inertias.keys()), y=list(inertias.values()), mode="lines+markers", line=dict(color="#0f3352", width=2.5), marker=dict(size=9)))
        fig.update_layout(title="Elbow Curve", xaxis_title="k", yaxis_title="Inertia", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(k_table, x="k", y="Silhouette", markers=True, color_discrete_sequence=["#2e86c1"], title="Silhouette by k")
        fig.update_traces(line=dict(width=2.5), marker=dict(size=9))
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        valid_k = k_table.dropna(subset=["Silhouette"])
        st.metric("Selected k", best_k)
        st.metric("Best Silhouette", round(valid_k["Silhouette"].max(), 4) if not valid_k.empty else "n/a")
        st.metric("Evaluated Range", f"k = {int(k_table['k'].min())} to {int(k_table['k'].max())}")

    st.dataframe(k_table, use_container_width=True)

    labels = run_kmeans(scaled, best_k)
    frame, profile = profile_clusters(df, labels)

    c4, c5 = st.columns([1.3, 1])
    with c4:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(scaled)
        risk_map = dict(zip(profile.index, profile["RISK"]))
        scatter = pd.DataFrame({
            "PC1": coords[:, 0], 
            "PC2": coords[:, 1],
            "Segment": frame["CLUSTER"].astype(str),
            "Risk": frame["CLUSTER"].map(risk_map)
        })
        fig = px.scatter(scatter, x="PC1", y="PC2", color="Segment", hover_data=["Risk"], color_discrete_sequence=PALETTE, title="PCA Cluster Separation")
        st.plotly_chart(fig, use_container_width=True)
    with c5:
        fig = px.bar(profile, x="SEGMENT", y="COMPOSITE", color="SEGMENT", color_discrete_map={row["SEGMENT"]: row["COLOR"] for _, row in profile.iterrows()}, text=profile["COMPOSITE"].round(2), title="Composite Score by Segment")
        fig.update_layout(showlegend=False, yaxis_range=[0, 5], xaxis_title="", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    c6, c7 = st.columns(2)
    with c6:
        radar_cols = ["AWARENESS_SCORE", "ATTITUDE_SCORE", "BEHAVIOR_SCORE", "PARTICIPATION_SCORE"]
        fig = go.Figure()
        for _, row in profile.iterrows():
            values = [row[c] for c in radar_cols] + [row[radar_cols[0]]]
            theta = ["Awareness", "Attitude", "Behavior", "Participation", "Awareness"]
            fig.add_trace(go.Scatterpolar(r=values, theta=theta, fill="toself", name=row["SEGMENT"]))
        fig.update_layout(title="Segment Profiles", polar=dict(radialaxis=dict(range=[0, 5])))
        st.plotly_chart(fig, use_container_width=True)
    with c7:
        st.markdown("### Cluster Summary Table")
        st.dataframe(profile[["SEGMENT", "N", "SHARE", "COMPOSITE", "RISK"]].round(2), use_container_width=True)

with tab4:
    metrics, fi, roc_data, cm_data, preds = classify(df)
    if metrics is None:
        st.info("The binarized target does not contain enough usable records to train classifiers.")
    else:
        st.dataframe(metrics, use_container_width=True)

        fig = go.Figure()
        for _, row in metrics.iterrows():
            fig.add_trace(go.Bar(
                name=row["Model"],
                x=["Accuracy", "Precision", "Recall", "F1"],
                y=[row["Accuracy"], row["Precision"], row["Recall"], row["F1"]],
            ))
        fig.update_layout(barmode="group", title="Classifier Comparison", yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        if roc_data:
            fig = go.Figure()
            for name, (fpr, tpr, auc) in roc_data.items():
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Reference"))
            fig.update_layout(title="ROC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig, use_container_width=True)

        if cm_data:
            c1, c2, c3 = st.columns(3)
            for col, (name, matrix) in zip([c1, c2, c3], cm_data.items()):
                with col:
                    fig = px.imshow(
                        matrix, text_auto=True,
                        labels=dict(x="Predicted", y="Actual"),
                        x=["Low/Neutral", "High"], y=["Low/Neutral", "High"],
                        color_continuous_scale="Blues", title=name,
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.caption("Metrics computed on the stratified 20 percent holdout. Rows with ambiguous compound answers were excluded.")

with tab5:
    metrics, fi, roc_data, cm_data, preds = classify(df)
    if fi is not None:
        fig = px.bar(fi.sort_values("Importance"), x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Teal", title="Random Forest Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(fi, use_container_width=True)
    else:
        st.info("Feature importance requires a trained Random Forest model.")

    tests = demographic_tests(df)
    if not tests.empty:
        st.dataframe(tests, use_container_width=True)
        fig = px.bar(tests, x="Test", y="Statistic", color="p-value", color_continuous_scale="RdYlGn_r", title="Demographic Association Statistics")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No demographic association test had sufficient grouped data.")

    st.caption("Feature importance uses mean decrease in impurity. Association tests use ANOVA for grouped demographics, point-biserial for gender, and Spearman for distance.")

with tab6:
    for col, label in [("Q34", "Suggestions"), ("Q35", "Observed Problems"), ("Q36", "Barriers and Motivations")]:
        values = df[col].apply(clean_text)
        values = values[values.str.len() > 0].dropna().head(15)
        st.subheader(label)
        if values.empty:
            st.write("No responses recorded.")
        else:
            for value in values:
                st.markdown(f"<div class='open-card'>{value}</div>", unsafe_allow_html=True)