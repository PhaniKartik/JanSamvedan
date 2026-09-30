import os
import io
import json
import math
import urllib.request
import urllib.parse
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from dotenv import load_dotenv
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# 1. Environment & Client Setup
# -----------------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="JanSamvedan | Multi-Agent DPI Decision Cockpit",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.sidebar.warning("⚠️ GEMINI_API_KEY not found in environment or .env file.")
    api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")

client = genai.Client(api_key=api_key) if api_key else None

# -----------------------------------------------------------------------------
# 2. Dynamic Spatial Telemetry Engine (OSM Geocoding)
# -----------------------------------------------------------------------------
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two GPS coordinates in kilometers."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2)**2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

@st.cache_data(ttl=600)
def detect_real_ip_location() -> Tuple[float, float, str, str, str]:
    """Dynamically detects the user's network IP location."""
    try:
        req = urllib.request.Request(
            "http://ip-api.com/json/",
            headers={"User-Agent": "JanSamvedan-DPI-Platform/1.0"}
        )
        with urllib.request.urlopen(req, timeout=4) as response:
            data = json.loads(response.read().decode("utf-8"))
            if data.get("status") == "success":
                lat = float(data["lat"])
                lon = float(data["lon"])
                city = data.get("city", "Urban Centre")
                region = data.get("regionName", "Telangana")
                country = data.get("country", "India")
                suburb_detail, dist, st_name = reverse_geocode_osm(lat, lon)
                full_desc = suburb_detail if suburb_detail else f"{city}, {region}, {country}"
                return lat, lon, city, region, full_desc
    except Exception:
        pass
    return 17.3850, 78.4867, "Hyderabad", "Telangana", "Ward 63 Mangalhat, Hyderabad, Telangana, India"

def reverse_geocode_osm(lat: float, lon: float) -> Tuple[str, str, str]:
    """Reverse geocodes coordinates to administrative hierarchy (Panchayat/Ward, District, State)."""
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lon}"
        req = urllib.request.Request(url, headers={"User-Agent": "JanSamvedan-DPI-Platform/1.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            addr = data.get("address", {})
            locality = addr.get("suburb") or addr.get("neighbourhood") or addr.get("village") or addr.get("residential") or addr.get("road") or "Urban Ward"
            district = addr.get("city") or addr.get("state_district") or addr.get("county") or "District"
            state = addr.get("state") or "India"
            return f"{locality}, {district}, {state}", district, state
    except Exception:
        return "Unknown Locality, Hyderabad, Telangana", "Hyderabad", "Telangana"

def forward_geocode_osm(place_query: str) -> Optional[Tuple[float, float, str, str, str]]:
    """Forward geocodes a spoken locality into target coordinates, district, and state."""
    try:
        clean_query = f"{place_query}, India"
        encoded_query = urllib.parse.quote(clean_query)
        url = f"https://nominatim.openstreetmap.org/search?format=json&q={encoded_query}&addressdetails=1&limit=1"
        req = urllib.request.Request(url, headers={"User-Agent": "JanSamvedan-DPI-Platform/1.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data and len(data) > 0:
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                addr = data[0].get("address", {})
                district = addr.get("city") or addr.get("state_district") or addr.get("county") or place_query
                state = addr.get("state") or "India"
                display_name = data[0].get("display_name", place_query)
                return lat, lon, district, state, display_name
    except Exception:
        pass
    return None

# -----------------------------------------------------------------------------
# 3. Domain Data Schemas
# -----------------------------------------------------------------------------
class InfrastructureSector(str, Enum):
    ROADS_BRIDGES = "Roads & Bridges"
    WATER_SANITATION = "Water & Sanitation"
    POWER_ENERGY = "Power & Energy"
    HEALTHCARE = "Primary Healthcare"
    EDUCATION = "Education Infrastructure"
    DRAINAGE_FLOOD = "Drainage & Flood Control"

class UrgencyLevel(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

class CitizenGrievanceExtraction(BaseModel):
    detected_language: str = Field(description="Language or dialect detected.")
    english_translation: str = Field(description="Strict verbatim English translation without adding unmentioned items.")
    sector: InfrastructureSector = Field(description="Classified infrastructure sector.")
    asset_type: str = Field(description="Specific asset involved (e.g., culvert, bitumen road, borewell, PHC sub-center, overhead line).")
    failure_mode: str = Field(description="Precise ground failure mode.")
    urgency: UrgencyLevel = Field(description="Assessed severity level based on ground disruption.")
    landmark_entities: List[str] = Field(description="Extracted local landmarks, habitations, or localities explicitly spoken.")
    reported_location_name: str = Field(description="Administrative entity query for the incident site.")
    confidence_score: float = Field(description="Extraction confidence between 0.0 and 1.0.")

class EnvironmentalAssessment(BaseModel):
    terrain_type: str = Field(description="Likely terrain (e.g., Hilly, Riverine, Dense Urban, Plains).")
    environmental_clearances_required: List[str] = Field(description="Statutory clearances (e.g., Forest, Coastal, TSPCB, Groundwater).")
    land_acquisition_complexity: str = Field(description="High, Medium, or Low complexity with reasoning.")
    gis_risk_flags: List[str] = Field(description="Key spatial risks.")

class SchemeAssessment(BaseModel):
    eligible_schemes: List[str] = Field(description="List of applicable GoI or State schemes.")
    recommended_scheme: str = Field(description="Primary scheme recommended.")
    financial_risk_flags: List[str] = Field(description="Funding overlap or pipeline constraints.")
    budget_availability_status: str = Field(description="Summary of PFMS budget availability.")

class ProjectConceptNote(BaseModel):
    project_title: str = Field(description="Formal civil works project title.")
    target_administrative_unit: str = Field(description="Gram Panchayat / Ward, District, State.")
    recommended_central_scheme: str = Field(description="Applicable Government of India or State scheme.")
    estimated_capital_outlay_inr_lakhs: float = Field(description="Capital expenditure estimate in Lakhs INR.")
    estimated_beneficiaries: int = Field(description="Target direct population served.")
    executive_problem_statement: str = Field(description="Formal administrative summary of ground deficits.")
    socio_economic_impact_justification: str = Field(description="Direct rationale linking project to poverty alleviation and access.")
    environmental_and_gis_clearances: List[str] = Field(description="Clearances aggregated from Agent 1.")
    preliminary_timeline_months: int = Field(description="Estimated execution timeframe in months.")

# -----------------------------------------------------------------------------
# 4. Federated Public Data Lake Mock
# -----------------------------------------------------------------------------
class PublicDataLake:
    @staticmethod
    def get_lgd_and_mpi_table() -> pd.DataFrame:
        data = [
            {"lgd_code": "203948", "panchayat": "Gangavaram", "block": "Rampachodavaram", "district": "Alluri Sitharama Raju", "state": "Andhra Pradesh", "lat": 17.5834, "lon": 81.7921, "mpi_score": 0.54, "tribal_pop_pct": 78.4, "total_population": 4200, "source": "MoPR LGD & NITI Aayog MPI"},
            {"lgd_code": "194821", "panchayat": "Chunar Dehat", "block": "Chunar", "district": "Mirzapur", "state": "Uttar Pradesh", "lat": 25.1242, "lon": 82.8711, "mpi_score": 0.42, "tribal_pop_pct": 14.1, "total_population": 8900, "source": "MoPR LGD & NITI Aayog MPI"},
            {"lgd_code": "239102", "panchayat": "Banjara Hills W-10", "block": "Khairatabad", "district": "Hyderabad", "state": "Telangana", "lat": 17.4156, "lon": 78.4350, "mpi_score": 0.04, "tribal_pop_pct": 1.2, "total_population": 34000, "source": "GHMC Ward Directory & Census"},
            {"lgd_code": "128490", "panchayat": "Kabisuryanagar", "block": "Kabisuryanagar", "district": "Ganjam", "state": "Odisha", "lat": 19.6642, "lon": 84.7731, "mpi_score": 0.39, "tribal_pop_pct": 22.8, "total_population": 6100, "source": "MoPR LGD & NITI Aayog MPI"},
            {"lgd_code": "156782", "panchayat": "Bishnupur Gram", "block": "Bishnupur", "district": "Bankura", "state": "West Bengal", "lat": 23.0678, "lon": 87.3176, "mpi_score": 0.33, "tribal_pop_pct": 16.5, "total_population": 5400, "source": "MoPR LGD & NITI Aayog MPI"}
        ]
        return pd.DataFrame(data)

    @staticmethod
    def get_asset_deficit_table() -> pd.DataFrame:
        data = [
            {"block": "Rampachodavaram", "sector": "Roads & Bridges", "dist_to_nearest_facility_km": 11.2, "asset_registry_ref": "PMGSY OMMS ID: AP-ASR-09"},
            {"block": "Chunar", "sector": "Water & Sanitation", "dist_to_nearest_facility_km": 6.8, "asset_registry_ref": "JJM IMIS Source: UP-MZP-082"},
            {"block": "Khairatabad", "sector": "Drainage & Flood Control", "dist_to_nearest_facility_km": 0.5, "asset_registry_ref": "GHMC Nala Channel 12"},
            {"block": "Khairatabad", "sector": "Water & Sanitation", "dist_to_nearest_facility_km": 0.6, "asset_registry_ref": "HMWSSB Water Feeder Line 04"},
            {"block": "Kabisuryanagar", "sector": "Primary Healthcare", "dist_to_nearest_facility_km": 9.4, "asset_registry_ref": "ABDM HFR Code: OD-GNJ-44"},
            {"block": "Bishnupur", "sector": "Education Infrastructure", "dist_to_nearest_facility_km": 4.8, "asset_registry_ref": "UDISE+ School Building Code: 191301001"}
        ]
        return pd.DataFrame(data)

    @staticmethod
    def get_public_investments_table() -> pd.DataFrame:
        data = [
            {"block": "Rampachodavaram", "sector": "Roads & Bridges", "scheme_name": "Unallocated / None", "sanctioned_budget_lakhs": 0.0, "status": "Unbudgeted Deficit"},
            {"block": "Chunar", "sector": "Water & Sanitation", "scheme_name": "Unallocated / None", "sanctioned_budget_lakhs": 0.0, "status": "Unbudgeted Deficit"},
            {"block": "Khairatabad", "sector": "Drainage & Flood Control", "scheme_name": "AMRUT 2.0 Stormwater", "sanctioned_budget_lakhs": 85.0, "status": "In Execution"},
            {"block": "Khairatabad", "sector": "Water & Sanitation", "scheme_name": "Unallocated / None", "sanctioned_budget_lakhs": 0.0, "status": "Unbudgeted Deficit"},
            {"block": "Kabisuryanagar", "sector": "Primary Healthcare", "scheme_name": "Unallocated / None", "sanctioned_budget_lakhs": 0.0, "status": "Unbudgeted Deficit"},
            {"block": "Bishnupur", "sector": "Education Infrastructure", "scheme_name": "Samagra Shiksha", "sanctioned_budget_lakhs": 45.0, "status": "Sanctioned"}
        ]
        return pd.DataFrame(data)

# -----------------------------------------------------------------------------
# 5. MCDA Scoring Engine (Configurable Policy Weights)
# -----------------------------------------------------------------------------
def calculate_composite_mcda_score(
    complaints: int, mpi: float, dist_km: float, budget_lakhs: float,
    w_demand: float = 30.0, w_equity: float = 30.0, w_deficit: float = 25.0, w_unfunded: float = 15.0
) -> Dict[str, float]:
    demand_sub = round((min(complaints, 50) / 50.0) * w_demand, 2)
    equity_sub = round(max(0.0, min(mpi, 1.0)) * w_equity, 2)
    deficit_sub = round((min(max(dist_km, 0.0), 10.0) / 10.0) * w_deficit, 2)
    unfunded_sub = 0.0 if budget_lakhs > 0 else float(w_unfunded)
    total = round(demand_sub + equity_sub + deficit_sub + unfunded_sub, 2)
    return {
        "demand": demand_sub,
        "equity": equity_sub,
        "deficit": deficit_sub,
        "unfunded_gap": unfunded_sub,
        "total": total
    }

# -----------------------------------------------------------------------------
# 6. Session State Initialization
# -----------------------------------------------------------------------------
def reconcile_hotspot_registry():
    lgd_mpi = PublicDataLake.get_lgd_and_mpi_table()
    assets = PublicDataLake.get_asset_deficit_table()
    budgets = PublicDataLake.get_public_investments_table()

    feedback_events = [
        {"block": "Rampachodavaram", "sector": "Roads & Bridges", "complaints": 28, "critical_complaints": 14, "dominant_issue": "Submerged box culvert cutting off 4 tribal habitations during monsoons", "sample_voice": "వంతెన కొట్టుకుపోయింది. స్కూలు పిల్లలు వెళ్లలేకపోతున్నారు."},
        {"block": "Chunar", "sector": "Water & Sanitation", "complaints": 39, "critical_complaints": 21, "dominant_issue": "Arsenic contamination in shallow handpumps; missing piped water distribution main", "sample_voice": "पानी में आर्सेनिक की समस्या है, नई पाइपलाइन चाहिए।"},
        {"block": "Khairatabad", "sector": "Drainage & Flood Control", "complaints": 48, "critical_complaints": 2, "dominant_issue": "Stormwater drain siltation and road waterlogging", "sample_voice": "Road number 12 is flooded again after 30 minutes of rain."},
        {"block": "Kabisuryanagar", "sector": "Primary Healthcare", "complaints": 19, "critical_complaints": 10, "dominant_issue": "Dilapidated PHC sub-center building without solar cold chain or labor room", "sample_voice": "ଏଠାରେ ଡାକ୍ତରଖାନା ନାହିଁ, ୧୦ କିଲୋମିଟର ଦୂର ଯିବାକୁ ପଡୁଛି।"},
    ]
    fb_df = pd.DataFrame(feedback_events)
    merged = pd.merge(lgd_mpi, fb_df, on="block")
    merged = pd.merge(merged, assets, on=["block", "sector"])
    merged = pd.merge(merged, budgets, on=["block", "sector"])

    records = []
    for idx, row in merged.iterrows():
        scores = calculate_composite_mcda_score(
            row["complaints"], row["mpi_score"], row["dist_to_nearest_facility_km"], row["sanctioned_budget_lakhs"]
        )
        row_dict = row.to_dict()
        row_dict.update(scores)
        row_dict["id"] = f"HS-{str(idx+1).zfill(3)}"
        row_dict["confidence"] = 0.95
        records.append(row_dict)
    return pd.DataFrame(records)

if "hotspots_df" not in st.session_state:
    st.session_state.hotspots_df = reconcile_hotspot_registry()

# -----------------------------------------------------------------------------
# 7. Sidebar Controls & Policy Profiles
# -----------------------------------------------------------------------------
st.sidebar.title("🏛️ JanSamvedan")
st.sidebar.caption("Agentic Urban Infrastructure AI | Track 1: AI for DPI")

detected_lat, detected_lon, detected_city, detected_region, detected_full_addr = detect_real_ip_location()
st.sidebar.markdown("### 📡 Live Geolocation Status")
st.sidebar.info(f"🌐 **IP Auto-Detected:**\n\n`{detected_full_addr}`\n\n**Coordinates:** `{detected_lat:.4f}° N, {detected_lon:.4f}° E`")

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Governance Levers (MCDA Policy Profile)")
policy_profile = st.sidebar.selectbox(
    "Active Policy Framework:",
    [
        "⚖️ Balanced National Framework (30/30/25/15)",
        "🌱 Equity-First (Aspirational Districts 15/45/25/15)",
        "🚧 Infrastructure Deficit-First (20/20/45/15)",
        "📢 Citizen Demand-First (50/20/15/15)"
    ]
)

if "Equity-First" in policy_profile:
    w_dem, w_eq, w_def, w_unf = 15.0, 45.0, 25.0, 15.0
elif "Deficit-First" in policy_profile:
    w_dem, w_eq, w_def, w_unf = 20.0, 20.0, 45.0, 15.0
elif "Citizen Demand-First" in policy_profile:
    w_dem, w_eq, w_def, w_unf = 50.0, 20.0, 15.0, 15.0
else:
    w_dem, w_eq, w_def, w_unf = 30.0, 30.0, 25.0, 15.0

# Recalculate scores dynamically when policy framework changes
for idx in range(len(st.session_state.hotspots_df)):
    r = st.session_state.hotspots_df.iloc[idx]
    updated_scores = calculate_composite_mcda_score(
        int(r["complaints"]), float(r["mpi_score"]), float(r["dist_to_nearest_facility_km"]), float(r["sanctioned_budget_lakhs"]),
        w_dem, w_eq, w_def, w_unf
    )
    for k, v in updated_scores.items():
        st.session_state.hotspots_df.at[idx, k] = v

st.sidebar.header("🎛️ Governance Filters")
selected_sector = st.sidebar.selectbox("Infrastructure Sector:", ["All Sectors"] + [s.value for s in InfrastructureSector])
min_score = st.sidebar.slider("Min MCDA Priority Score (0-100):", 0, 100, 25)

active_df = st.session_state.hotspots_df.copy()
if selected_sector != "All Sectors":
    active_df = active_df[active_df["sector"] == selected_sector]
active_df = active_df[active_df["total"] >= min_score].sort_values(by="total", ascending=False)

# -----------------------------------------------------------------------------
# 8. Navigation Tabs
# -----------------------------------------------------------------------------
tab_map, tab_ingest, tab_public_data, tab_audit = st.tabs([
    "🗺️ Multi-Agent GIS Decision Cockpit",
    "🎙️ Live Mic & Spatial Resolution Engine",
    "📂 Public Dataset Integration",
    "📑 MCDA Score Matrix & Audit Ledger"
])

# -----------------------------------------------------------------------------
# TAB 1: Multi-Agent GIS Cockpit
# -----------------------------------------------------------------------------
with tab_map:
    col_map, col_details = st.columns([1.2, 1.0])

    with col_map:
        st.subheader("📍 Demand Hotspot Cartography")
        center_lat = active_df["lat"].mean() if not active_df.empty else detected_lat
        center_lon = active_df["lon"].mean() if not active_df.empty else detected_lon
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=5)

        for _, row in active_df.iterrows():
            marker_color = "red" if row["total"] >= 60 else "orange" if row["total"] >= 45 else "blue"
            popup_html = f"<b>{row['panchayat']}, {row['district']}</b><br>Score: <b>{row['total']}/100</b><br>Complaints: {row['complaints']}<br>Deficit: {row['dist_to_nearest_facility_km']} km"
            folium.Marker(
                location=[row["lat"], row["lon"]],
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"{row['panchayat']} ({row['district']}) - Priority: {row['total']}/100",
                icon=folium.Icon(color=marker_color, icon="exclamation-sign" if row["total"] >= 60 else "info-sign")
            ).add_to(m)
        st_folium(m, width="100%", height=530)

    with col_details:
        st.subheader("📋 Selected Hotspot Telemetry")
        if active_df.empty:
            st.info("No active hotspots match the filter threshold.")
        else:
            selected_id = st.selectbox(
                "Select Hotspot for Multi-Agent Review:",
                options=active_df["id"].tolist(),
                format_func=lambda x: f"{x} - {active_df.loc[active_df['id'] == x, 'panchayat'].values[0]} ({active_df.loc[active_df['id'] == x, 'district'].values[0]}) | Score: {active_df.loc[active_df['id'] == x, 'total'].values[0]}"
            )

            hotspot = active_df[active_df["id"] == selected_id].iloc[0]

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Priority Score", f"{hotspot['total']}/100")
            k2.metric("Citizen Feedback", f"{hotspot['complaints']} Reports")
            k3.metric("MPI Deprivation", f"{hotspot['mpi_score']}")
            k4.metric("Active Sanction", f"₹{hotspot['sanctioned_budget_lakhs']}L")

            st.markdown(f"**Identified Deficit:** {hotspot['dominant_issue']}")
            st.markdown(f"**Administrative Entity:** `{hotspot['panchayat']}, {hotspot['district']}, {hotspot['state']}`")
            st.markdown(f"**Statutory Asset Reference:** `{hotspot.get('asset_registry_ref', 'Decoupled Spatial Telemetry')}`")

            st.markdown("---")
            if st.button("⚡ Trigger Multi-Agent PCN Synthesis", use_container_width=True, type="primary"):
                if not client:
                    st.error("Please configure a valid GEMINI_API_KEY.")
                else:
                    with st.status("🤖 Multi-Agent Policy Workflow Initiated...", expanded=True) as status:
                        try:
                            # AGENT 1: Environmental & GIS Planner 
                            st.write("🕵️‍♂️ **Agent 1 (GIS & Environmental Expert):** Assessing terrain and land-use clearances...")
                            a1_prompt = f"Analyze environmental and land constraints for a {hotspot['sector']} project in {hotspot['panchayat']}, {hotspot['district']}, {hotspot['state']}. Coordinates: {hotspot['lat']}, {hotspot['lon']}."
                            a1_resp = client.models.generate_content(
                                model="gemini-3.1-pro-preview",
                                contents=a1_prompt,
                                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=EnvironmentalAssessment, temperature=0.1)
                            )
                            env_data = EnvironmentalAssessment.model_validate_json(a1_resp.text)
                            st.write(f"↳ *Result:* Terrain: {env_data.terrain_type}. Required Clearances: {', '.join(env_data.environmental_clearances_required)}")

                            # AGENT 2: Finance & Scheme Auditor (Semantic Anchoring)
                            st.write("🧑‍💼 **Agent 2 (Finance Auditor):** Validating GoI scheme eligibility and budget gates...")
                            a2_prompt = f"""
Determine the most eligible, currently active Government of India or State scheme for a {hotspot['sector']} deficit in {hotspot['district']}, {hotspot['state']}. 
MPI score is {hotspot['mpi_score']}, current sanctioned budget is INR {hotspot['sanctioned_budget_lakhs']} Lakhs.

GUIDELINES:
- Prioritize major flagship infrastructure schemes if logically aligned (e.g. AMRUT 2.0 for urban drainage/water, PMGSY for rural connectivity, Jal Jeevan Mission for rural water, PM-ABHIM for health).
- If context warrants a specific state or central program, select the most accurate active scheme. Do not invent non-existent programs.
"""
                            a2_resp = client.models.generate_content(
                                model="gemini-3.1-pro-preview",
                                contents=a2_prompt,
                                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=SchemeAssessment, temperature=0.1)
                            )
                            fin_data = SchemeAssessment.model_validate_json(a2_resp.text)
                            st.write(f"↳ *Result:* Matched Scheme: {fin_data.recommended_scheme}. Status: {fin_data.budget_availability_status}")

                            # AGENT 3: Policy Synthesizer
                            st.write("👨‍‍⚖️ **Agent 3 (Policy Synthesizer):** Drafting the Project Concept Note...")
                            pcn_prompt = f"""
You are the Principal Secretary. Synthesize the final PCN using inputs from Agent 1 (Environment) and Agent 2 (Finance):
[LOCATION]: {hotspot['panchayat']}, {hotspot['district']}, {hotspot['state']}
[DEFICIT]: {hotspot['dominant_issue']}
[AGENT 1 ENVIRONMENTAL]: Terrain: {env_data.terrain_type}, Clearances: {env_data.environmental_clearances_required}, Risks: {env_data.gis_risk_flags}
[AGENT 2 FINANCE]: Recommended Scheme: {fin_data.recommended_scheme}, Budget Status: {fin_data.budget_availability_status}
Draft the formal executive PCN integrating these parameters.
"""
                            final_resp = client.models.generate_content(
                                model="gemini-3.1-pro-preview",
                                contents=pcn_prompt,
                                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=ProjectConceptNote, temperature=0.2)
                            )
                            pcn = ProjectConceptNote.model_validate_json(final_resp.text)
                            
                            status.update(label="✅ Project Concept Note Synthesized Successfully!", state="complete", expanded=False)

                            with st.expander("📄 AI-Generated Draft Project Concept Note (Pending Human Review)", expanded=True):
                                st.markdown(f"### {pcn.project_title}")
                                st.markdown(f"**Recommended Funding Vehicle:** `{pcn.recommended_central_scheme}`")
                                
                                m1, m2, m3 = st.columns(3)
                                m1.metric("Estimated Outlay", f"₹{pcn.estimated_capital_outlay_inr_lakhs:.1f} Lakhs")
                                m2.metric("Target Beneficiaries", f"{pcn.estimated_beneficiaries:,}")
                                m3.metric("Execution Window", f"{pcn.preliminary_timeline_months} Months")

                                st.markdown("#### Executive Problem Statement")
                                st.write(pcn.executive_problem_statement)

                                st.markdown("#### Socio-Economic Impact Justification")
                                st.write(pcn.socio_economic_impact_justification)

                                st.markdown("#### Statutory & Environmental Clearances (Agent 1)")
                                for flag in pcn.environmental_and_gis_clearances:
                                    st.markdown(f"- ⚠️ {flag}")
                                    
                        except Exception as e:
                            status.update(label="❌ Multi-Agent Workflow Failed", state="error", expanded=True)
                            st.error(str(e))

# -----------------------------------------------------------------------------
# TAB 2: Live Mic & Spatial Resolution Engine
# -----------------------------------------------------------------------------
with tab_ingest:
    st.subheader("🎙️ Live Multilingual Ingress & Spatial Clustering Engine")

    st.markdown("##### 📱 Citizen Reporter Origin (Device / Network Telemetry)")
    loc_mode = st.radio("Reporter Telemetry Source:", ["🌐 Dynamic Live IP Detection", "🏛️ Select Benchmark Administrative Block", "✏️ Manual Override Coordinates"], horizontal=True)

    if loc_mode == "🌐 Dynamic Live IP Detection":
        reporter_gps_lat, reporter_gps_lon, reporter_location_name = detected_lat, detected_lon, detected_full_addr
        rep_dist, rep_state = detected_city, detected_region
    elif loc_mode == "🏛️ Select Benchmark Administrative Block":
        chosen_block = st.selectbox("Select Reporter Origin Block:", ["Khairatabad (Hyderabad, Telangana)", "Rampachodavaram (ASR District, Andhra Pradesh)", "Chunar (Mirzapur, Uttar Pradesh)", "Bishnupur (Bankura, West Bengal)"])
        if "Rampachodavaram" in chosen_block: reporter_gps_lat, reporter_gps_lon, rep_dist, rep_state = 17.5834, 81.7921, "Alluri Sitharama Raju", "Andhra Pradesh"
        elif "Chunar" in chosen_block: reporter_gps_lat, reporter_gps_lon, rep_dist, rep_state = 25.1242, 82.8711, "Mirzapur", "Uttar Pradesh"
        elif "Bishnupur" in chosen_block: reporter_gps_lat, reporter_gps_lon, rep_dist, rep_state = 23.0678, 87.3176, "Bankura", "West Bengal"
        else: reporter_gps_lat, reporter_gps_lon, rep_dist, rep_state = 17.4156, 78.4350, "Hyderabad", "Telangana"
        reporter_location_name = chosen_block
    else:
        c1, c2 = st.columns(2)
        reporter_gps_lat = c1.number_input("Reporter Lat:", value=detected_lat, format="%.5f")
        reporter_gps_lon = c2.number_input("Reporter Lon:", value=detected_lon, format="%.5f")
        addr_str, rep_dist, rep_state = reverse_geocode_osm(reporter_gps_lat, reporter_gps_lon)
        reporter_location_name = addr_str
    
    st.success(f"📱 **Reporter Origin Bound:** `{reporter_location_name}` | **Telemetry:** `{reporter_gps_lat:.5f}° N, {reporter_gps_lon:.5f}° E`")
    st.markdown("---")

    input_channel = st.radio("Choose Ingress Modality:", ["🔴 Live Microphone", "⌨️ Dialect Text Input", "📑 Benchmark Transcripts (8 Indic Languages)"], horizontal=True)

    audio_bytes_data = None
    audio_mime_type = "audio/wav"
    text_content_data = None

    if input_channel == "🔴 Live Microphone":
        live_recording = st.audio_input("Record Voice (Speaks in your mother tongue):")
        if live_recording:
            audio_bytes_data = live_recording.read()
            audio_mime_type = live_recording.type
    elif input_channel == "⌨️ Dialect Text Input":
        text_content_data = st.text_area("Type or paste complaint:", height=100)
    elif input_channel == "📑 Benchmark Transcripts (8 Indic Languages)":
        presets = {
            "Telugu (Flooding & Drainage - Ward 63)": "వర్షం పడిన ప్రతిసారీ ఈ ప్రాంతంలో నీరు నిలిచిపోతోంది. డ్రైనేజీ పూర్తిగా బ్లాక్ అయిపోయింది.",
            "Hindi (Handpump & Drinking Water - Mirzapur)": "हमारे गांव चुनार में हैंडपंप से आर्सेनिक वाला पानी आ रहा है, पीने का साफ पानी नहीं है।",
            "Bengali (Dilapidated School - Bankura)": "বিষ্ণুপুর গ্রামে প্রাথমিক বিদ্যালয়ের ছাদ ভেঙে পড়ছে, বাচ্চারা ক্লাসে যেতে পারছে না।",
            "Odia (PHC Sub-Center - Ganjam)": "ଏଠାରେ ଡାକ୍ତରଖାନା ନାହିଁ, ଗର୍ଭବତୀ ମହିଳାମାନଙ୍କୁ ୧୦ କିଲୋମିଟର ଦୂର ଯିବାକୁ ପଡୁଛି।",
            "Tamil (Bridge Washout - Vellore)": "எங்கள் ஊர் பாலம் மழையில் அடித்து செல்லப்பட்டது, போக்குவரத்து துண்டிக்கப்பட்டுள்ளது.",
            "Marathi (Broken Rural Feeder Road - Vidarbha)": "आमच्या गावातील मुख्य रस्ता पावसामुळे वाहून गेला आहे, शेतमालाची वाहतूक ठप्प झाली आहे.",
            "Kannada (Borewell Failure - Raichur)": "ನಮ್ಮ ಹಳ್ಳಿಯಲ್ಲಿ ಕುಡಿಯುವ ನೀರಿನ ಕೊಳವೆಬಾವಿ ಕೆಟ್ಟುಹೋಗಿ ಎರಡು ವಾರವಾಯಿತು, ನೀರಿಗೆ ತೊಂದರೆಯಾಗಿದೆ.",
            "Santali (Tribal Habitation Access - Mayurbhanj)": "ᱟᱞᱮᱭᱟᱜ ᱟᱛᱳ ᱨᱮ ᱰᱟᱦᱟᱨ ᱵᱟᱹᱱᱩᱜᱼᱟ, ᱫᱟᱜ ᱫᱤᱱ ᱟᱹᱰᱤ ᱦᱟᱨᱠᱮᱛ ᱦᱩᱭᱩᱜ ᱠᱟᱱᱟ।"
        }
        chosen_preset = st.selectbox("Select Multilingual Ground Benchmark:", list(presets.keys()))
        text_content_data = st.text_area("Citizen Telemetry:", value=presets[chosen_preset], height=100)

    st.markdown("---")
    
    if st.button("🚀 Process via Gemini 3.8 Flash Pipeline", type="primary", use_container_width=True):
        if not client:
            st.error("Please configure GEMINI_API_KEY in your .env file or sidebar.")
        elif not (audio_bytes_data or text_content_data):
            st.warning("Please record voice or provide text.")
        else:
            with st.spinner("Gemini 3.8 Flash analyzing dialect (Zero-Hallucination) and decoupling incident coordinates..."):
                try:
                    payload_parts = []
                    geo_instructions = f"""
STRICT TRANSLATION & GEO-RESOLUTION INSTRUCTIONS:
1. ZERO-HALLUCINATION TRANSLATION: Transcribe strictly verbatim. DO NOT invent unmentioned utilities or assets.
2. LOCATION RESOLUTION HIERARCHY:
   - If citizen explicitly speaks a locality (e.g. 'Chunar', 'Banjara Hills', 'Chintal Basti'): Extract into 'reported_location_name' and 'landmark_entities'.
   - ONLY if citizen mentions ZERO localities (e.g. 'near my house', 'in this area'): Fall back to reporter origin: '{reporter_location_name}'.
"""
                    if audio_bytes_data:
                        payload_parts.append(types.Part.from_bytes(data=audio_bytes_data, mime_type=audio_mime_type))
                        payload_parts.append(f"Listen, translate to formal English, and extract infrastructure parameters.\n{geo_instructions}")
                    else:
                        payload_parts.append(f"Analyze this feedback, translate, and extract parameters:\n{geo_instructions}\n\n{text_content_data}")

                    response = client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=payload_parts,
                        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=CitizenGrievanceExtraction, temperature=0.0)
                    )
                    extracted = CitizenGrievanceExtraction.model_validate_json(response.text)

                    # Dynamic Geocoding Resolution (Incident vs. Reporter)
                    incident_lat, incident_lon = None, None
                    target_district, target_state = rep_dist, rep_state
                    incident_location_name = extracted.reported_location_name
                    is_remote = False

                    if extracted.landmark_entities:
                        query_term = extracted.landmark_entities[0]
                        geo_res = forward_geocode_osm(f"{query_term}, {extracted.reported_location_name}")
                        if not geo_res:
                            geo_res = forward_geocode_osm(query_term)
                        if geo_res:
                            incident_lat, incident_lon, target_district, target_state, display_name = geo_res
                            incident_location_name = display_name
                            is_remote = True

                    if not incident_lat or not incident_lon:
                        incident_lat, incident_lon = reporter_gps_lat, reporter_gps_lon
                        target_district, target_state = rep_dist, rep_state
                        if not incident_location_name or "unspecified" in incident_location_name.lower():
                            incident_location_name = reporter_location_name

                    st.session_state.pending_ticket = {
                        "extracted": extracted,
                        "incident_lat": incident_lat,
                        "incident_lon": incident_lon,
                        "incident_location_name": incident_location_name,
                        "target_district": target_district,
                        "target_state": target_state,
                        "reporter_location_name": reporter_location_name,
                        "is_remote_report": is_remote
                    }
                except Exception as e:
                    st.error(f"Pipeline Error: {str(e)}")

    # Ingestion Block with Spatial Clustering
    if "pending_ticket" in st.session_state:
        pt = st.session_state.pending_ticket
        ext = pt["extracted"]
        
        st.success("✅ Grievance Structured, Translated & Spatially Resolved!")
        loc_c1, loc_c2 = st.columns(2)
        with loc_c1:
            st.markdown("📍 **Target Incident Site (Asset):**")
            st.markdown(f"**Location:** `{pt['incident_location_name']}`")
            st.markdown(f"**Jurisdiction:** `{pt['target_district']}, {pt['target_state']}`")
            st.markdown(f"**Coords:** `{pt['incident_lat']:.5f}° N, {pt['incident_lon']:.5f}° E`")
        with loc_c2:
            st.markdown("📱 **Reporter Origin (Citizen Telemetry):**")
            st.markdown(f"**Origin:** `{pt['reporter_location_name']}`")
            st.markdown(f"**Dialect Detected:** `{ext.detected_language}` (Confidence: `{ext.confidence_score * 100:.1f}%`)")
            if pt['is_remote_report']:
                st.info("ℹ️ *Remote Reporting Identified: Spoken incident site resolved independently of citizen device coordinates.*")

        st.markdown(f"**Translation:** {ext.english_translation}")
        st.markdown(f"**Asset & Failure:** `{ext.asset_type}` — *{ext.failure_mode}*")
        
        st.markdown("---")
        if st.button("➕ Ingest / Aggregate into Master Hotspot Engine", use_container_width=True, type="primary"):
            # Check for spatial clustering (< 15 km in same infrastructure sector)
            matched_hotspot_idx = None
            for idx, row in st.session_state.hotspots_df.iterrows():
                if row["sector"] == ext.sector.value:
                    d_km = haversine_distance(pt["incident_lat"], pt["incident_lon"], float(row["lat"]), float(row["lon"]))
                    if d_km <= 15.0:
                        matched_hotspot_idx = idx
                        break

            if matched_hotspot_idx is not None:
                # Aggregate into existing cluster
                current_cnt = int(st.session_state.hotspots_df.at[matched_hotspot_idx, "complaints"]) + 1
                st.session_state.hotspots_df.at[matched_hotspot_idx, "complaints"] = current_cnt
                if ext.urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.HIGH]:
                    st.session_state.hotspots_df.at[matched_hotspot_idx, "critical_complaints"] = int(st.session_state.hotspots_df.at[matched_hotspot_idx, "critical_complaints"]) + 1
                
                # Recalculate MCDA score with aggregated volume
                row_ref = st.session_state.hotspots_df.iloc[matched_hotspot_idx]
                new_scores = calculate_composite_mcda_score(
                    current_cnt, float(row_ref["mpi_score"]), float(row_ref["dist_to_nearest_facility_km"]), float(row_ref["sanctioned_budget_lakhs"]),
                    w_dem, w_eq, w_def, w_unf
                )
                for k, v in new_scores.items():
                    st.session_state.hotspots_df.at[matched_hotspot_idx, k] = v
                
                clustered_id = st.session_state.hotspots_df.at[matched_hotspot_idx, "id"]
                del st.session_state.pending_ticket
                st.success(f"⚡ Spatial Aggregation Complete! Merged into cluster {clustered_id}. Feedback count elevated to {current_cnt}.")
                st.rerun()

            else:
                # Create a new Hotspot
                new_id = f"HS-{str(len(st.session_state.hotspots_df) + 1).zfill(3)}"
                panchayat_name = ext.landmark_entities[0] if ext.landmark_entities else pt["incident_location_name"].split(',')[0]
                
                scores = calculate_composite_mcda_score(1, 0.15, 3.5, 0.0, w_dem, w_eq, w_def, w_unf)
                new_row = {
                    "id": new_id,
                    "lgd_code": f"LGD-{new_id}",
                    "panchayat": panchayat_name,
                    "block": panchayat_name,
                    "district": pt["target_district"],
                    "state": pt["target_state"],
                    "lat": pt["incident_lat"],
                    "lon": pt["incident_lon"],
                    "mpi_score": 0.15,
                    "tribal_pop_pct": 5.0,
                    "total_population": 15000,
                    "sector": ext.sector.value,
                    "complaints": 1,
                    "critical_complaints": 1 if ext.urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.HIGH] else 0,
                    "dist_to_nearest_facility_km": 3.5,
                    "asset_registry_ref": "Decoupled Spatial Telemetry",
                    "scheme_name": "Unallocated / None",
                    "sanctioned_budget_lakhs": 0.0,
                    "status": "Unbudgeted Deficit",
                    "dominant_issue": f"{ext.asset_type}: {ext.failure_mode}",
                    "sample_voice": ext.english_translation,
                    "confidence": ext.confidence_score,
                    **scores
                }
                
                st.session_state.hotspots_df = pd.concat([st.session_state.hotspots_df, pd.DataFrame([new_row])], ignore_index=True)
                del st.session_state.pending_ticket
                st.success(f"✅ Integrated as new Demand Hotspot {new_id}! Check Tab 1 to generate the PCN.")
                st.rerun()

# -----------------------------------------------------------------------------
# TAB 3: Public Dataset Integration
# -----------------------------------------------------------------------------
with tab_public_data:
    st.subheader("📂 Public Dataset Integration Gateway (Data Provenance)")
    st.markdown("Federated indicators linking citizen demand to authoritative Digital Public Infrastructure (DPI) registries.")
    
    st.markdown("##### 1. Local Government Directory (LGD) & NITI Aayog Multidimensional Poverty Index (MPI)")
    st.dataframe(PublicDataLake.get_lgd_and_mpi_table(), use_container_width=True)

    st.markdown("##### 2. Statutory Asset Infrastructure Deficit Matrix (PM GatiShakti / PMGSY / JJM / ABDM)")
    st.dataframe(PublicDataLake.get_asset_deficit_table(), use_container_width=True)

    st.markdown("##### 3. Public Financial Management System (PFMS) Budget Pipeline")
    st.dataframe(PublicDataLake.get_public_investments_table(), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: MCDA Score Matrix & Audit Ledger
# -----------------------------------------------------------------------------
with tab_audit:
    st.subheader("📑 100-Point MCDA Scoring Engine & Decision Audit Ledger")
    st.markdown(f"**Active Formula:** `Priority = Demand ({w_dem}%) + Equity ({w_eq}%) + Deficit ({w_def}%) + Unfunded Gap ({w_unf}%)`")
    
    audit_cols = ["id", "panchayat", "district", "state", "sector", "complaints", "total", "demand", "equity", "deficit", "unfunded_gap"]
    st.dataframe(st.session_state.hotspots_df[audit_cols], use_container_width=True)