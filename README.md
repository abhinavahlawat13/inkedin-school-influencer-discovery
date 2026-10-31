# LinkedIn School Influencer Discovery Tool (EDXSO AI Internship MVP)

An AI/ML-powered discovery and ranking engine built to identify relevant professionals, advisors, former principals, and school leaders in the K-12 education ecosystem using semantic vector similarity and contextual re-ranking.

## Core Workflow
`Reference Profile → Context Extraction → TF-IDF Vector Embeddings → Cosine Similarity Search → Contextual Re-Ranking → Transparent Match Explanations`

---

## Key Features
- **Deterministic AI/ML Ranking:** Combines sublinear TF-IDF n-gram vector representations with cosine similarity, persona matching, and institutional affiliation bonuses.
- **Explainability Engine:** Generates transparent, human-readable rationales detailing why each candidate was matched.
- **Authentic Test Dataset:** Curated with 40+ authentic K-12 leaders, trustees, and advisors across India (CBSE, ICSE, Cambridge, IB) compliant with LinkedIn's security and terms of service.
- **Custom Reference Sandbox:** Enables users to test preset benchmark figures or define dynamic reference profiles directly in the UI.

---

## Repository Structure
```text
├── data/
│   ├── candidates.json           # 40+ structured K-12 school ecosystem leaders
│   └── reference_profiles.json   # Benchmark seed profiles (Directors, Advisors, Founders)
├── src/
│   ├── app.py                    # Streamlit operational dashboard
│   ├── recommender.py            # Embedding vectorizer, cosine similarity & re-ranker
│   └── models.py                 # Pydantic v2 schemas
├── docs/
│   └── technical_note.md         # Deliverable 4: Architecture & methodology report
├── requirements.txt
├── .gitignore
└── README.md