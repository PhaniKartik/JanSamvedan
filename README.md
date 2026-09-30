# 🏛️ JanSamvedan: Agentic Urban Infrastructure AI

**Track 1 Submission: AI for Digital Public Infrastructure (DPI) & Governance**  

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit)](https://jansamvedan-87fcufobvxvnyrwe67s8tt.streamlit.app/)

## 📌 The Problem
Urban infrastructure planning in India suffers from a massive disconnect between grassroots citizen grievances (OpEx/maintenance) and state-level capital expenditure (CapEx) pipelines. Citizen complaints are localized, multilingual, and often spatially ambiguous, making it difficult for District Magistrates and Nodal Agencies to identify genuine infrastructure deficits and route them to appropriate Centrally Sponsored Schemes (CSS).

## 🚀 The Solution: JanSamvedan
JanSamvedan is an autonomous, multi-agent Digital Public Infrastructure (DPI) decision cockpit. It ingests unstructured, multilingual citizen grievances (voice or text), resolves spatial ambiguity, clusters demand dynamically, and uses a multi-agent LLM workflow to draft Cabinet-ready **Project Concept Notes (PCNs)** aligned with actual government funding vehicles.

### ✨ Core Architecture & Features

1. **🎙️ Multilingual Ingress & Zero-Hallucination Extraction (Gemini 3.8 Flash)**
   - Ingests dialect-heavy voice and text across 8 Indic languages.
   - Extracts exact failure modes, assets, and urgency while strictly preventing AI hallucinations (temperature = 0.0).

2. **🗺️ Spatial Decoupling & 15km Demand Clustering**
   - **Decoupling:** Separates the *citizen's device location* from the *spoken incident location* using OpenStreetMap (Nominatim) forward/reverse geocoding.
   - **Clustering:** Automatically calculates Haversine distance to aggregate duplicate reports within a 15km radius, preventing duplicate tickets and mathematically elevating priority scores.

3. **⚖️ Configurable MCDA Prioritization Engine**
   - Replaces arbitrary AI prioritizing with a deterministic 100-point Multi-Criteria Decision Analysis (MCDA) matrix.
   - Computes priority based on Citizen Demand, NITI Aayog MPI (Equity), Infrastructure Deficits, and Unfunded Budget Gaps.
   - Features adjustable Governance Levers (e.g., *Equity-First*, *Deficit-First*).

4. **🤖 Multi-Agent Policy Synthesis (Gemini 3.1 Pro)**
   - **Agent 1 (GIS & Environmental):** Assesses terrain complexity and flags required statutory clearances (e.g., TSPCB, Groundwater).
   - **Agent 2 (Finance Auditor):** Uses semantic anchoring to map the deficit to the exact active Central or State scheme (e.g., AMRUT 2.0, PMGSY).
   - **Agent 3 (Policy Synthesizer):** Drafts the final, structured Project Concept Note detailing socio-economic justification, estimated outlay, and execution timelines.

## 🛠️️ Tech Stack
* **AI/LLM:** Google Gemini 3.8 Flash (Data Extraction), Gemini 3.1 Pro (Agentic Reasoning)
* **Frontend/UI:** Streamlit, Streamlit-Folium
* **Geospatial Processing:** OpenStreetMap Nominatim API, Folium, Haversine routing
* **Data Layer:** Pandas, Python Pydantic (Structured JSON Schema Enforcement)

## 💻 Local Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PhaniKartik/JanSamvedan.git
   cd JanSamvedan
   
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate

   pip install -r requirements.txt

   GEMINI_API_KEY="your_api_key_here"

   streamlit run app.py
