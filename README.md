# 🎓 Student Segmentation & Personalized Course Recommendation System for EduPro

A data-driven machine learning system and interactive Streamlit web dashboard designed to transform one-size-fits-all online learning into adaptive, learner-centric educational journeys.

---

## 📌 Project Overview

Online learners are heterogeneous in their motivations, pace, and domain interests:
- Some explore introductory courses across diverse disciplines.
- Some specialize deeply in advanced technical domains.
- Others focus on career-oriented multi-course certifications.

Traditional platforms rely on generic, popularity-driven recommendations that fail to retain students or improve completion rates. This project introduces **student-centric machine learning intelligence** to the EduPro platform by segmenting learners into distinct behavioral archetypes and delivering cluster-aware hybrid course recommendations.

---

## 🚀 Key Features

1. **High-Dimensional Feature Engineering**:
   - **Engagement Metrics**: Total courses enrolled, active enrollment frequency, total learning hours.
   - **Preference Indicators**: Dominant category, level preferences (Beginner, Intermediate, Advanced), free vs. paid ratio.
   - **Behavioral Fingerprints**: *Learning Depth Index* ($[0.33, 1.00]$), *Curriculum Diversity Score* ($[0, 1]$), *Shannon Category Entropy*, and *Specialization Ratio*.

2. **Unsupervised Learner Segmentation (4 Discovered Archetypes)**:
   - **Cluster 0: Foundational & Entry-Level Starters (21.8%)**: 100% beginner courses, exploratory entry into learning.
   - **Cluster 1: Intermediate Applied Practitioners (40.9%)**: 62.4% intermediate courses, steady workplace upskilling.
   - **Cluster 2: High-Engagement Super-Learners (15.1%)**: 13.4 courses enrolled, 367+ learning hours, $1,248 avg spend, cross-disciplinary power users.
   - **Cluster 3: Deep Domain Specialists (22.2%)**: 100% advanced mastery courses, direct-to-expert technical focus.

3. **Cluster-Aware Hybrid Recommendation Engine**:
   - Multi-criteria scoring: TF-IDF Content Cosine Similarity (35%), Intra-Cluster Peer Velocity (35%), Quality Rating (15%), and Pedagogical Level-Progression Fit (15%).
   - Generates sequenced **3-stage learning roadmaps** (*Foundation $\rightarrow$ Core $\rightarrow$ Mastery*).

4. **Interactive Streamlit Web Dashboard**:
   - 🏠 **Platform Overview & KPIs**: Platform health, enrollment distributions, revenue breakdown.
   - 👤 **Learner Profile Explorer**: Search 3,000 learners, demographic stats, behavioral radar charts, and enrollment history.
   - 📊 **Cluster Visualizer & Personas**: 2D/3D PCA & t-SNE scatter plots, centroid heatmaps, archetype cards.
   - 🎯 **Personalized Recommendations**: Live recommendations with match scores, filter by category/level/type, explanation rationales.
   - 📈 **Segment Comparison Analytics**: Cross-segment spend distributions, depth vs diversity boxplots.
   - 🧪 **What-If Learner Simulator**: Live sliders to simulate new learner profiles and predict their cluster and recommendations.
   - 📑 **Model Validation & Research**: Silhouette scores, Calinski-Harabasz, intra-cluster cohesion, and lift projections.

---

## 📂 Project Structure

```
├── EduPro Online Platform.xlsx          # Multi-sheet dataset (Users, Courses, Transactions, Teachers)
├── src/
│   ├── __init__.py                      # Package initialization
│   ├── data_loader.py                   # Data ingestion, schema validation & joins
│   ├── feature_engineering.py          # Engagement, preference, behavioral feature engineering
│   ├── clustering.py                    # K-Means, Hierarchical, PCA 2D/3D, t-SNE, profiling
│   ├── recommender.py                   # Cluster-aware hybrid recommender & learning paths
│   └── evaluation.py                    # Silhouette, Cohesion, Hold-out Hit Rate@5, Lift
├── data/                                # Generated pipeline artifacts
│   ├── processed_learners.csv           # 3,000 learner feature vectors with cluster labels
│   ├── cluster_profiles.csv             # Centroid comparison across segments
│   ├── courses_processed.csv            # Course catalog metadata
│   ├── evaluation_metrics.json          # Quantitative ML evaluation metrics
│   └── k_sweep_metrics.json             # Diagnostic sweep across K=2..8
├── reports/
│   ├── research_paper.md                # Full academic/technical research paper
│   └── executive_summary.md             # Policy brief for government stakeholders
├── app.py                               # Interactive Streamlit Web Application
├── run_pipeline.py                      # End-to-end data processing & model training script
├── requirements.txt                     # Python package dependencies
└── README.md                            # Documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ How to Run

### Run the Interactive Streamlit Web App:
```bash
streamlit run app.py
```
*(or on Windows: `python -m streamlit run app.py`)*

Open your browser at **`http://localhost:8501`**.

### Re-Run the End-to-End ML Pipeline:
```bash
python run_pipeline.py
```

---

## 📊 Experimental Evaluation Highlights

| Metric | Result | Purpose |
| :--- | :---: | :--- |
| **Calinski-Harabasz Index** | **1,221.3** | Global mathematical separation peak for $K=4$ clusters |
| **Intra-Cluster Cohesion (Super-Learners)** | **0.9027** | High behavioral consistency among power learners |
| **Hold-out Hit Rate @ 5** | **18.0%** | Exact course prediction accuracy on held-out transactions |
| **Category Match Precision @ 5** | **48.67%** | Precision matching user's subsequent domain choice |
| **Projected Course Completion Lift** | **+28.4%** | Estimated completion rate improvement over baseline |
| **Projected Multi-Course Retention Lift** | **+34.2%** | Estimated increase in repeat enrollments |

---

## 📄 License & Attribution
Developed for EduPro Educational Analytics & Research. Distributed under the MIT License.
