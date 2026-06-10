# Skincare Product Recommendation — ML Project

**Lënda:** Machine Learning | **Viti Akademik:** 2025-2026
**Dataset:** AI Skincare Personalization (Kaggle)

---

## Përshkrimi

Sistem inteligjent për rekomandimin e rutinës skincare bazuar në profilin e lëkurës së përdoruesit. Projekti implementon dhe krahason 7 modele ML (klasifikim + clustering) mbi një dataset të bashkuar nga 3 burime: users, products dhe interactions.

5 klasa (rutina): Acne · Hydration · Brightening · Anti-Aging · Sensitive Skin

---

## Struktura e Projektit

```
ml-project/
├── skincare_ml.ipynb     # Notebook kryesor
├── users.csv             # Profili i lëkurës së përdoruesve
├── skincare_100.csv      # 100 produkte skincare (dataset i prelucruar)
├── interactions.csv      # Vlerësimet e produkteve nga users
├── ml_models/
│   └── model.pkl         # Modeli i eksportuar me joblib
├── requirements.txt
└── README.md
```

---

## Struktura e Notebook-it

| # | Seksioni |
|---|----------|
| 1 | Ngarkimi i bibliotekave dhe te dhenave |
| 2 | Paraperpunimi (Preprocessing) |
| 3 | Analiza Eksploratore (EDA) |
| 4 | Inxhinieria e Vecorive (Feature Engineering) |
| 5 | Klasifikimi — 5 modele + Hyperparameter Tuning |
| 6 | Rrjeta Neurale (ANN/MLP) — 2 arkitektura |
| 7 | Krahasimi i Modeleve |
| 8 | Grupimi (K-Means Clustering) |
| 9 | Konkluzionet Finale |

---

## Instalimi

### Kerkesat

- Python >= 3.10
- Anaconda (rekomandohet)
- Jupyter Notebook

### Instalimi i bibliotekave

```bash
pip install -r requirements.txt
```

ose me Anaconda:

```bash
conda install --file requirements.txt
```

### Ekzekutimi

```bash
cd ml-project
jupyter notebook skincare_ml.ipynb
```

> Shenim: Skedaret `users.csv`, `skincare_100.csv` dhe `interactions.csv` duhet te jene ne te njejten dosje me notebook-in.

---

## Modelet e Implementuara

### Klasifikimi (Supervised Learning)

| # | Modeli | Qasja | Tuning |
|---|--------|-------|--------|
| 1 | Decision Tree | Pema vendimmarrese | GridSearchCV (max_depth, criterion) |
| 2 | Naive Bayes | Probabilistik (Bayes theorem) | GridSearchCV (var_smoothing) |
| 3 | Random Forest | Ansambel pemesh | GridSearchCV (n_estimators, max_depth) |
| 4 | KNN | K fqinjet me te afert | GridSearchCV (n_neighbors, metric) |
| 5 | SVM | Kufi vendimmarres optimal | GridSearchCV (C, kernel, gamma) |
| 6 | ANN Arkitektura A | MLP 128->64, 50 epoka | warm_start per-epoch tracking |
| 7 | ANN Arkitektura B | MLP 256->128->64->32, L2, 50 epoka | warm_start + alpha=0.001 |

### Clustering (Unsupervised Learning)

| Modeli | K | Metoda |
|--------|---|--------|
| K-Means | 5 | k-means++, Elbow + Silhouette |

---

## Metrikat e Vleresimit

Cdo model vlerësohet me:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Feature Engineering

| Grupi | Vecorite |
|-------|----------|
| Profili i lekures | Age, Skin_Type, Skin_Tone, Climate, Diet, Hormonal_Status, Budget_Level |
| Severity-et | Acne, Dryness, Pigmentation, Aging, Sensitivity (0-10) |
| Produkti | Brand, Price, product_mean_rating, product_popularity |
| Historia e userit | user_mean_rating, user_rating_std, user_interactions |
| Ingredientet | One-Hot Encoding (MultiLabelBinarizer) |

**Target:** Severity-i dominant i lekures percakton rutinen skincare (logjike dermatologjike)

**Split:** 80% train / 20% test | `random_state=42` | `stratify=y`

---

## Eksportimi i Modelit

Modeli me i mire eksportohet me `joblib` per integrim me FastAPI (Lab Course 2):

```python
import joblib

model_data = {
    'model':          rf,
    'encoders':       enc_dt,
    'target_encoder': le_target_dt,
    'feature_cols':   feature_cols_dt,
}
joblib.dump(model_data, 'ml_models/model.pkl')
```

Ngarkimi:

```python
import joblib
model_data = joblib.load('ml_models/model.pkl')
model = model_data['model']
```

---

## Vizualizimet

Notebook-i gjeneron:
- Boxplot i severity-eve te lekures
- Histogram + Pie chart i vleresimeve (class distribution)
- Bar charts: tipi i lekures, buxheti, statusi hormonal
- Heatmap e korrelacionit ndermjet vecorive numerike
- Frekuenca e ingredienteve
- Confusion Matrix per secilin model
- Feature Importance (Decision Tree, Random Forest)
- Decision Tree vizual (max_depth=3)
- F1-Score sipas K (KNN)
- Evolucioni i Loss/Accuracy/Precision/Recall gjate epokave (ANN)
- Krahasimi final i modeleve — bar chart + radar chart
- K-Means: Elbow Method, Silhouette, PCA 2D, profili i grupeve

---

## Dataset

| Skedari | Rreshtat | Pershkrimi |
|---------|----------|------------|
| `users.csv` | ~1,000 | Profili i lekures: skin type, severities, demographics |
| `skincare_100.csv` | 100 | Produkte: Brand, Category, Price, Ingredients |
| `interactions.csv` | ~10,000 | User-Product ratings (1-5) |

`skincare_100.csv` eshte i gjeneruar me `preprocessing.py` — 20 produkte per secilen nga 5 kategorite, `RANDOM_SEED=42`.
