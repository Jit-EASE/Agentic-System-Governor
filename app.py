
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    SYSTEM_NAME,
    SYSTEM_VERSION,
    STRESS_MODEL_CARD_FILE,
    INFERENCE_LOG_FILE,
    GOVERNANCE_LOG_FILE,
    PROTECTED_ATTRIBUTES,
)
from governance.registry import (
    AlgorithmRegistry,
    init_registry_with_stress_model,
    init_stress_risk_model_card,
    verify_dataset_licensing,
)
from governance.consent import UserProfile, evaluate_consent
from governance.fairness import compute_group_metrics
from governance.energy import compute_energy_summary
from governance.aia import generate_aia_report
from models.risk_model import StressRiskModel


st.set_page_config(
    page_title="Governed Agri AI Prototype",
    layout="wide",
)

st.title("🇮🇪 Governed Agri-AI Prototype – Stress Risk Advisor")

init_registry_with_stress_model()
init_stress_risk_model_card()

registry = AlgorithmRegistry()


@st.cache_resource
def get_model():
    return StressRiskModel.train_on_synthetic()


model = get_model()

# Sidebar
st.sidebar.header("User Profile & Policy Mode")

policy_mode = st.sidebar.selectbox(
    "Operational Mode",
    ["Research", "Public Service"],
)

if policy_mode == "Public Service":
    st.sidebar.warning(
        "PUBLIC SERVICE COMPLIANCE MODE ACTIVE ⚠️\n"
        "Consent, child protections and licensing enforcement are strict."
    )
else:
    st.sidebar.info("Research Sandbox Mode – flexible prototyping with logging.")

user_id = st.sidebar.text_input("Pseudonymous User ID", value="farmer_demo_001")
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=35, step=1)
country = st.sidebar.text_input("Country", value="IE")

consent = st.sidebar.checkbox(
    "I consent to use of my data for this advisory prototype", value=False
)

user = UserProfile(user_id=user_id, age=int(age), country=country)
allowed, is_child = evaluate_consent(user, consent_granted=consent)

if is_child:
    st.sidebar.warning(
        "Child profile detected (<16). Additional legal constraints apply."
    )

st.sidebar.markdown("---")
st.sidebar.write(f"**System:** {SYSTEM_NAME} v{SYSTEM_VERSION}")
st.sidebar.write("This prototype is for **research/demo only** – not real advice.")

(
    tab_predict,
    tab_governance,
    tab_registry,
    tab_aia,
) = st.tabs(
    [
        "🌾 Stress Risk Prediction",
        "⚖️ Governance & Ethical Risk",
        "📚 Registry & Licensing",
        "🧩 AIA & Carbon Footprint",
    ]
)

# Prediction tab
with tab_predict:
    st.subheader("Farm-Level Stress Risk Assessment")

    col1, col2, col3 = st.columns(3)

    with col1:
        milk_yield = st.number_input("Milk yield (L/cow/day)", 5.0, 50.0, 22.0, 0.5)
        soil_moisture = st.slider("Soil moisture (0-1)", 0.0, 1.0, 0.55, 0.01)
        grass_growth_index = st.slider(
            "Grass growth index", 0.0, 2.0, 1.0, 0.05
        )

    with col2:
        financial_stress_index = st.slider(
            "Financial stress index (0-1)", 0.0, 1.0, 0.4, 0.01
        )
        herd_size = st.number_input("Herd size", 10, 1000, 120, 5)
        age_farmer = st.number_input("Age of farmer", 18, 90, int(age), 1)

    with col3:
        gender = st.selectbox("Gender (for fairness testing)", ["M", "F"])
        farm_size_class = st.selectbox(
            "Farm size class", ["small", "medium", "large"]
        )

    if st.button("Run governed stress risk assessment"):
        licence_status = verify_dataset_licensing(registry)

        if policy_mode == "Public Service":
            if not consent:
                st.error(
                    "Consent REQUIRED for Public Service mode — inference BLOCKED."
                )
                st.stop()

            if is_child:
                from governance.audit import log_governance_event

                st.error(
                    "Under-16 profile detected — prediction BLOCKED in Public Service mode."
                )
                log_governance_event(
                    "CHILD_INFERENCE_BLOCKED",
                    {"user_id": user.user_id, "age": user.age},
                )
                st.stop()

            if not licence_status["valid"]:
                st.error(
                    "Dataset licensing issues present — Public Service inference LOCKED."
                )
                st.json(licence_status["issues"])
                st.stop()
        else:
            if not consent:
                st.warning(
                    "Consent missing — allowed only because you are in Research mode. "
                    "In production this would be blocked."
                )

        df = pd.DataFrame(
            [
                {
                    "milk_yield": milk_yield,
                    "soil_moisture": soil_moisture,
                    "grass_growth_index": grass_growth_index,
                    "financial_stress_index": financial_stress_index,
                    "herd_size": herd_size,
                    "age_of_farmer": age_farmer,
                    "gender": gender,
                    "farm_size_class": farm_size_class,
                }
            ]
        )

        df_enc = pd.get_dummies(
            df, columns=["gender", "farm_size_class"], drop_first=True
        )
        template = model.get_feature_template()
        aligned = template.copy()
        for col in df_enc.columns:
            aligned[col] = df_enc.iloc[0][col]

        pred, proba, expl = model.predict_with_governance(
            user_id=user.user_id,
            feature_row=aligned,
            is_child=is_child,
            consent_granted=consent,
        )

        label = "High stress risk" if pred == 1 else "Low/moderate stress risk"
        st.success(
            f"**Prediction:** {label} "
            f"(probability high stress = {proba:.2%})"
        )

        st.markdown("#### Local Explanation (feature contributions)")
        top_expl = sorted(
            expl.items(), key=lambda kv: abs(kv[1]), reverse=True
        )[:8]
        expl_df = pd.DataFrame(
            top_expl, columns=["Feature", "Contribution (coef × value)"]
        )
        st.dataframe(expl_df, use_container_width=True)

# Governance tab
with tab_governance:
    st.subheader("Governance, Logs & Ethical Risk Dashboard")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### Recent Inference Logs")
        log_path = Path(INFERENCE_LOG_FILE)
        if log_path.exists():
            lines = log_path.read_text().strip().splitlines()[-100:]
            records = [json.loads(l) for l in lines]
            df_log = pd.json_normalize(records)
            st.dataframe(df_log, use_container_width=True, height=320)
        else:
            st.info("No inference logs yet.")

    with colB:
        st.markdown("#### Governance Events (consent, energy, child flags)")
        g_path = Path(GOVERNANCE_LOG_FILE)
        if g_path.exists():
            lines = g_path.read_text().strip().splitlines()[-100:]
            records = [json.loads(l) for l in lines]
            df_g = pd.json_normalize(records)
            st.dataframe(df_g, use_container_width=True, height=320)
        else:
            st.info("No governance events logged yet.")

    st.markdown("---")
    st.markdown("### Ethical Risk – Fairness Heatmaps")

    df_syn = StressRiskModel.generate_synthetic_data(800)
    df_enc = pd.get_dummies(
        df_syn, columns=["gender", "farm_size_class"], drop_first=True
    )
    X = df_enc.drop(columns=["stress_high"])
    y = df_enc["stress_high"]
    y_pred = model.model.predict(X)

    fairness_reports = {}
    for attr in PROTECTED_ATTRIBUTES:
        series_attr = df_syn[attr]
        fairness_reports[attr] = compute_group_metrics(
            y, y_pred, series_attr
        )

    for attr, report in fairness_reports.items():
        st.markdown(f"#### Protected attribute: `{attr}`")
        df_attr = pd.DataFrame(report).T

        st.write("Group metrics:")
        st.dataframe(df_attr, use_container_width=True)

        metrics_to_show = [
            c for c in df_attr.columns if c in ["positive_rate", "accuracy"]
        ]
        if metrics_to_show:
            fig, ax = plt.subplots()
            im = ax.imshow(
                df_attr[metrics_to_show].values, aspect="auto"
            )
            ax.set_xticks(range(len(metrics_to_show)))
            ax.set_xticklabels(metrics_to_show)
            ax.set_yticks(range(len(df_attr.index)))
            ax.set_yticklabels(df_attr.index)
            ax.set_title(f"Fairness heatmap for {attr}")
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)

# Registry tab
with tab_registry:
    st.subheader("Algorithm Registry & Licensing")

    st.markdown("#### Algorithm Registry")

    algos = registry.list_algorithms()
    if algos:
        st.dataframe(
            pd.json_normalize(algos), use_container_width=True
        )
    else:
        st.info("No algorithms registered yet.")

    st.markdown("---")
    st.markdown("#### Model Card")

    if STRESS_MODEL_CARD_FILE.exists():
        with open(STRESS_MODEL_CARD_FILE, "r") as f:
            card = json.load(f)
        st.json(card)
    else:
        st.info("Model card file not found.")

    st.markdown("---")
    st.markdown("#### Copyright & Dataset Licensing")

    licence_check = verify_dataset_licensing(registry)
    if licence_check["valid"]:
        st.success("All datasets have valid licences registered.")
    else:
        st.error(
            "Dataset licensing issues detected — Public Service deployment must be blocked."
        )
        st.json(licence_check["issues"])

# AIA tab
with tab_aia:
    st.subheader("Automated Algorithmic Impact Assessment (AIA)")

    aia_report = generate_aia_report(policy_mode=policy_mode)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### System Overview")
        st.json(aia_report["system_overview"])

        st.markdown("#### Usage Summary")
        st.json(aia_report["usage_summary"])

        st.markdown("#### Consent & Children")
        st.json(aia_report["consent_and_children"])

    with col2:
        st.markdown("#### Data Governance")
        st.json(aia_report["data_governance"])

        st.markdown("#### Qualitative Risk Assessment")
        st.json(aia_report["qualitative_risk_assessment"])

    st.markdown("#### Recommended Mitigations")
    for m in aia_report["recommended_mitigations"]:
        st.markdown(f"- {m}")

    st.markdown("---")
    st.markdown("### Carbon Footprint per Simulation / Training")

    energy_summary = aia_report["energy_and_carbon"]
    st.write(
        f"**Total estimated energy**: {energy_summary['total_kwh']:.4f} kWh"
    )
    st.write(
        f"**Total estimated emissions**: {energy_summary['total_kg_co2e']:.4f} kg CO₂e"
    )

    by_label = energy_summary["by_label"]
    if by_label:
        df_energy = pd.DataFrame(by_label).T
        df_energy["kWh"] = df_energy["wh"] / 1000.0
        st.markdown("#### Energy use by operation label")
        st.dataframe(df_energy, use_container_width=True)

        fig2, ax2 = plt.subplots()
        ax2.bar(df_energy.index, df_energy["kWh"])
        ax2.set_ylabel("kWh")
        ax2.set_title("Energy consumption by operation")
        ax2.set_xticklabels(df_energy.index, rotation=45, ha="right")
        st.pyplot(fig2)
    else:
        st.info("No energy usage events recorded yet.")

    st.markdown("---")
    st.markdown("#### Download full AIA JSON")
    st.download_button(
        label="Download AIA JSON",
        data=json.dumps(aia_report, indent=2),
        file_name="aia_report.json",
        mime="application/json",
    )
