import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported cleanly anywhere
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import json
import streamlit as st
import pandas as pd

from src.models import ProfileModel
from src.recommender import InfluencerRecommender
import json
from pathlib import Path
import streamlit as st
import pandas as pd

from src.models import ProfileModel
from src.recommender import InfluencerRecommender

# Page Configuration
st.set_page_config(
    page_title="LinkedIn School Influencer Discovery",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 10px;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 5px;
        background-color: #e3f2fd;
        color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_recommender():
    return InfluencerRecommender()


@st.cache_data
def load_reference_profiles():
    with open("data/reference_profiles.json", "r", encoding="utf-8") as f:
        return [ProfileModel(**item) for item in json.load(f)]


# Initialize Engine
try:
    recommender = get_recommender()
    seed_profiles = load_reference_profiles()
except Exception as e:
    st.error(f"Failed to load application data: {e}")
    st.stop()

# Header & Banner
st.title("🎓 LinkedIn School Influencer Discovery Tool")
st.caption("AI-Powered Discovery & Contextual Ranking Engine for the K-12 School Ecosystem (EDXSO MVP)")

# Top Metric Proof
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric(label="Total Ecosystem Candidates", value=len(recommender.candidates))
with col_m2:
    st.metric(label="Seed Reference Profiles", value=len(seed_profiles))
with col_m3:
    st.metric(label="Persona Categories", value=len(set(c.persona_type for c in recommender.candidates)))
with col_m4:
    st.metric(label="Min Target Discovery", value="≥ 20 Profiles")

st.divider()

# Sidebar: Reference Profile Selection or Custom Input
st.sidebar.header("🔍 Reference Profile Input")
input_mode = st.sidebar.radio("Input Mode", ["Select Preset Reference Profile", "Create Custom Reference Profile"])

if input_mode == "Select Preset Reference Profile":
    preset_names = [f"{p.name} ({p.current_role} - {p.organization})" for p in seed_profiles]
    selected_idx = st.sidebar.selectbox("Choose a Reference Leader:", range(len(seed_profiles)), format_func=lambda i: preset_names[i])
    current_ref: ProfileModel = seed_profiles[selected_idx]

else:
    st.sidebar.subheader("Custom Reference Details")
    c_name = st.sidebar.text_input("Full Name", value="Dr. Preeti Sharma")
    c_role = st.sidebar.text_input("Current Role", value="Senior K-12 Advisor & Trustee")
    c_org = st.sidebar.text_input("Organization", value="EduVision Foundation")
    c_loc = st.sidebar.text_input("Location", value="New Delhi, India")
    c_exp = st.sidebar.slider("Years of Experience", min_value=1, max_value=45, value=25)
    c_persona = st.sidebar.selectbox("Persona Type", ["School Director", "Education Consultant", "Former Principal", "Education Entrepreneur"])
    c_indicators = st.sidebar.text_area("Ecosystem Indicators (comma separated)", value="CBSE Accreditation, Teacher Capacity Building, Policy Governance, K-12 Advisory")
    c_affiliations = st.sidebar.text_area("Key Affiliations (comma separated)", value="FICCI ARISE, CBSE, NPSC")
    c_bio = st.sidebar.text_area("Bio Summary", value="Strategic advisor to private school boards, focusing on CBSE pedagogical compliance, leadership mentoring, and campus modernization.")

    current_ref = ProfileModel(
        id="CUSTOM_REF",
        name=c_name,
        linkedin_url="https://www.linkedin.com/in/custom-reference",
        current_role=c_role,
        organization=c_org,
        location=c_loc,
        experience_years=c_exp,
        persona_type=c_persona,
        ecosystem_indicators=[i.strip() for i in c_indicators.split(",") if i.strip()],
        bio_summary=c_bio,
        key_affiliations=[a.strip() for a in c_affiliations.split(",") if a.strip()]
    )

# Execution Controls
top_k_val = st.sidebar.slider("Number of Matches to Discover", min_value=10, max_value=35, value=20)
persona_filter = st.sidebar.multiselect(
    "Filter by Persona Type:",
    options=list(set(c.persona_type for c in recommender.candidates)),
    default=[]
)

# Reference Profile Overview Card
with st.expander("📌 Active Reference Profile Inspection", expanded=True):
    col_r1, col_r2 = st.columns([1, 2])
    with col_r1:
        st.markdown(f"### {current_ref.name}")
        st.write(f"**Role:** {current_ref.current_role}")
        st.write(f"**Organization:** {current_ref.organization}")
        st.write(f"**Location:** {current_ref.location}")
        st.write(f"**Seniority:** ~{current_ref.experience_years} years in education")
        st.write(f"**Persona:** `{current_ref.persona_type}`")
        st.markdown(f"[View Verified LinkedIn Profile]({current_ref.linkedin_url})")

    with col_r2:
        st.markdown("**Executive Summary:**")
        st.write(current_ref.bio_summary)
        st.markdown("**Ecosystem Indicators:**")
        st.markdown(" ".join([f"<span class='badge'>{ind}</span>" for ind in current_ref.ecosystem_indicators]), unsafe_allow_html=True)
        st.markdown("<br>**Affiliations / Institutional Ties:**", unsafe_allow_html=True)
        st.markdown(" ".join([f"<span class='badge'>{aff}</span>" for aff in current_ref.key_affiliations]), unsafe_allow_html=True)

# Run Ranking Pipeline
matches = recommender.rank_similar_profiles(current_ref, top_k=35)

if persona_filter:
    matches = [m for m in matches if m.profile.persona_type in persona_filter]

matches = matches[:top_k_val]

# Display Discovery Output Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Discovered Matches & Explanations", "📊 Analytics & Comparative Matrix", "📋 Data Table View"])

with tab1:
    st.subheader(f"Discovered {len(matches)} Relevant School Ecosystem Profiles")
    st.caption("Ranked by contextual relevance: 70% Semantic Embedding Similarity + 15% Persona Alignment + 15% Network/Affiliation Overlap.")

    for idx, match in enumerate(matches, 1):
        prof = match.profile
        with st.container():
            st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0; color: #0d47a1;">#{idx}. {prof.name}</h4>
                    <span style="background-color: #2e7d32; color: white; padding: 4px 10px; border-radius: 12px; font-weight: bold;">
                        {match.relevance_score}% Match
                    </span>
                </div>
                <p style="margin: 4px 0 8px 0; color: #424242;">
                    <strong>{prof.current_role}</strong> at <strong>{prof.organization}</strong> | 📍 {prof.location}
                </p>
            </div>
            """, unsafe_allow_html=True)

            col_c1, col_c2 = st.columns([2, 1])
            with col_c1:
                st.write(f"**Context:** {prof.bio_summary}")
                st.markdown("**Why this person was selected:**")
                for r in match.match_reasons:
                    st.markdown(f"- {r}")

            with col_c2:
                st.write(f"**Persona:** `{prof.persona_type}`")
                st.write(f"**Vector Similarity:** `{match.similarity_score}%`")
                st.write(f"**Experience:** ~{prof.experience_years} years")
                st.markdown(f"[Open LinkedIn Profile]({prof.linkedin_url})")

            with st.expander("View Indicators & Institutional Ties"):
                st.write("**Ecosystem Indicators:**", ", ".join(prof.ecosystem_indicators))
                st.write("**Key Affiliations:**", ", ".join(prof.key_affiliations))

            st.write("")

with tab2:
    st.subheader("Match Score Distribution & Breakdown")
    chart_df = pd.DataFrame([
        {
            "Name": m.profile.name,
            "Relevance Score": m.relevance_score,
            "Semantic Sim": m.similarity_score,
            "Persona": m.profile.persona_type
        }
        for m in matches
    ])

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("**Top Match Score Comparison**")
        st.bar_chart(chart_df.set_index("Name")["Relevance Score"])

    with col_chart2:
        st.markdown("**Persona Representation in Top Matches**")
        st.bar_chart(chart_df["Persona"].value_counts())

with tab3:
    st.subheader("Raw Profile Data Export")
    export_rows = [
        {
            "Rank": i,
            "Name": m.profile.name,
            "Relevance (%)": m.relevance_score,
            "Current Role": m.profile.current_role,
            "Organization": m.profile.organization,
            "Persona Type": m.profile.persona_type,
            "Location": m.profile.location,
            "Experience (Yrs)": m.profile.experience_years,
            "LinkedIn URL": m.profile.linkedin_url,
            "Primary Match Reason": m.match_reasons[0] if m.match_reasons else ""
        }
        for i, m in enumerate(matches, 1)
    ]
    st.dataframe(pd.DataFrame(export_rows), use_container_width=True)