# 🚀 SpaceX Falcon 9 Landing Success Predictor

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Complete](https://img.shields.io/badge/Status-Complete-success.svg)]()

> **End-to-end machine learning pipeline achieving 94.4% accuracy in predicting Falcon 9 first-stage landing outcomes**

An IBM Data Science Capstone project analyzing 90+ SpaceX launches to identify key factors in successful rocket landings and enable cost-optimized bidding strategies for competitive launch services.

---

## 🎯 Key Results

| Metric | Achievement |
|--------|-------------|
| **Prediction Accuracy** | 94.4% (Decision Tree) |
| **Launches Analyzed** | 90+ missions (2010-2020) |
| **Cost Implications** | $30M+ savings per successful landing |
| **Success Rate Improvement** | 0% → 80% over decade |
| **Best Launch Site** | KSC LC-39A (76.9% success rate) |
| **Optimal Orbits** | VLEO, SSO (near 100% success) |

---

## 📊 Project Highlights

### Data Collection & Engineering
- **Dual data sources**: SpaceX REST API + Wikipedia web scraping (BeautifulSoup)
- **Complex target engineering**: Converted multi-category landing outcomes (True Ocean, False ASDS, etc.) into binary classification
- **Data wrangling**: Handled missing values, one-hot encoded categorical variables, created calculated features

### Exploratory Data Analysis
- **SQL-based analysis**: 10+ custom queries on launch patterns, payload distributions, site performance
- **Statistical visualizations**: Matplotlib/Seaborn correlation analysis, trend identification
- **Geospatial mapping**: Folium interactive maps showing launch sites, success/failure clusters, proximity analysis
- **Interactive dashboards**: Plotly Dash app with dynamic filtering by site and payload mass

### Machine Learning
- **4 optimized models**: Logistic Regression, SVM (RBF kernel), Decision Tree, K-Nearest Neighbors
- **Hyperparameter tuning**: GridSearchCV with 10-fold cross-validation across 100+ parameter combinations
- **Feature standardization**: StandardScaler preprocessing for distance-based algorithms
- **Performance metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix analysis
- **Best model**: Decision Tree (94.4% accuracy, 96% F1-score, 0 false negatives)

---

## 🛠️ Technical Stack

**Data Collection & Processing:**
- `requests` - REST API integration
- `BeautifulSoup` - HTML parsing and web scraping
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing

**Database & Queries:**
- `sqlite3` - SQL database operations
- `sqlalchemy` - Database connectivity

**Visualization:**
- `matplotlib` - Static visualizations
- `seaborn` - Statistical graphics
- `folium` - Interactive geospatial maps
- `plotly` - Interactive charts and dashboards

**Machine Learning:**
- `scikit-learn` - Model training and evaluation
  - `LogisticRegression`
  - `SVC` (Support Vector Classifier)
  - `DecisionTreeClassifier`
  - `KNeighborsClassifier`
  - `GridSearchCV`
  - `train_test_split`
  - `StandardScaler`

---

## 📁 Project Structure

```
SpaceX_Capstone_Project/
│
├── Data_Collection_API_SpaceX.ipynb
│   └── Fetch launch data from SpaceX REST API
│       • 90+ launches collected
│       • JSON parsing and DataFrame creation
│       • Handling missing payload values
│
├── Data_Collection_Web_Scraping_SpaceX.ipynb
│   └── Scrape Falcon 9 launch records from Wikipedia
│       • BeautifulSoup HTML parsing
│       • Regex-based data cleaning
│       • Unicode normalization
│
├── Data_Wrangling_SpaceX.ipynb
│   └── Clean and transform raw data
│       • Binary target variable creation
│       • One-hot encoding categorical features
│       • Success rate calculations
│
├── EDA_SQL_Queries_SpaceX.ipynb
│   └── Database analysis with SQL
│       • 10+ custom analytical queries
│       • Payload mass aggregations
│       • Launch site comparisons
│
├── EDA_Matplotlib_Visualization_SpaceX.ipynb
│   └── Statistical visualizations
│       • Scatter plots (Flight # vs Payload, Orbit analysis)
│       • Bar charts (Success by orbit type)
│       • Line charts (Success rate over time)
│
├── Interactive_Visualizations_Folium_SpaceX.ipynb
│   └── Geospatial analysis
│       • Interactive launch site maps
│       • Success/failure marker clusters
│       • Proximity measurements (coastline, cities, highways)
│
└── Machine_Learning_Prediction_SpaceX.ipynb
    └── Classification model training and evaluation
        • 4 algorithms with GridSearchCV tuning
        • Confusion matrix analysis
        • Model comparison and selection
```

---

## 🔬 Methodology

### 1. Data Collection
```python
# SpaceX API
response = requests.get("https://api.spacexdata.com/v4/launches/past")
data = response.json()

# Web Scraping
soup = BeautifulSoup(html_text, 'html.parser')
falcon9_table = soup.find_all('table')[0]
```

### 2. Data Wrangling
```python
# Binary classification target
landing_outcomes = df['Landing_Outcome'].value_counts()
df['Class'] = df['Landing_Outcome'].apply(
    lambda x: 1 if 'Success' in x else 0
)
```

### 3. Exploratory Analysis
```sql
-- SQL Analysis Example
SELECT Launch_Site, 
       COUNT(*) as Total_Launches,
       SUM(Class) as Successful_Landings,
       ROUND(100.0 * SUM(Class) / COUNT(*), 2) as Success_Rate
FROM SPACEXTBL
GROUP BY Launch_Site
ORDER BY Success_Rate DESC;
```

### 4. Machine Learning
```python
# GridSearchCV with 10-fold CV
parameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 4, 6, 8, 10, 12, 14, 16, 18],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

tree_cv = GridSearchCV(
    estimator=DecisionTreeClassifier(),
    param_grid=parameters,
    cv=10
).fit(X_train, Y_train)

# Best model: 94.4% test accuracy
```

---

## 📈 Model Performance

| Model | Accuracy | Best Parameters | Notes |
|-------|----------|----------------|-------|
| **Decision Tree** ⭐ | **94.4%** | criterion='gini', max_depth=6 | Best overall, 0 false negatives |
| Logistic Regression | 83.3% | C=0.1, penalty='l2' | Strong baseline |
| SVM (RBF) | 83.3% | C=1.0, gamma=0.1 | Good generalization |
| K-Nearest Neighbors | 83.3% | n_neighbors=7, p=1 | Tied with LR/SVM |

### Confusion Matrix (Decision Tree)
```
              Predicted
              Fail  Success
Actual Fail     5      1
       Success  0     12
```

**Metrics:**
- Precision: 92.3% (12/13 positive predictions correct)
- Recall: 100% (12/12 actual successes identified)
- F1-Score: 96.0% (harmonic mean of precision/recall)

---

## 💡 Key Insights

### Launch Site Analysis
1. **KSC LC-39A** has highest success rate (76.9%) - newer site with upgraded infrastructure
2. **CCAFS LC-40** early site with lower success (26.9%) - older technology baseline
3. Sites located near equator maximize Earth's rotational velocity for fuel efficiency
4. Strategic coastal positioning minimizes collateral damage risk

### Payload Patterns
- Payloads <5,500 kg show 80%+ success rates
- Heavy payloads (>13,000 kg) exclusive to VLEO orbit, mostly successful
- Optimal range: 2,000-6,000 kg (highest success density)

### Orbit Type Impact
- **100% Success:** SSO (Sun-Synchronous), VLEO (Very Low Earth Orbit), GEO
- **50-85% Success:** LEO, ISS, GTO (Geostationary Transfer Orbit)
- **0% Success:** SO (Single Orbit) - limited sample size

### Temporal Trends
- 2010-2013: 0% success (developmental phase)
- 2014-2016: Rapid improvement to 60% (iterative engineering)
- 2017-2020: Plateau at 80-90% (mature technology)

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/brent-smith-4/IBM_Data_Science_Repo.git
cd IBM_Data_Science_Repo/Capstone_Project_SpaceX

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn folium plotly beautifulsoup4 requests
```

### Running Notebooks
```bash
# Launch Jupyter
jupyter notebook

# Open notebooks in sequence:
# Data_Collection_API → Web_Scraping → Data_Wrangling → 
# EDA_SQL → EDA_Visualization → Folium_Maps → Machine_Learning
```

---

## 🎓 Skills Demonstrated

### Data Engineering
- REST API integration with error handling
- Web scraping with BeautifulSoup (HTML parsing, regex)
- ETL pipeline design (Extract, Transform, Load)
- Data cleaning and missing value imputation

### Statistical Analysis
- SQL database querying and aggregation
- Correlation analysis
- Trend identification
- Distribution analysis

### Machine Learning
- Binary classification problem formulation
- Train-test split methodology
- Hyperparameter tuning with GridSearchCV
- Cross-validation (k-fold)
- Model evaluation metrics (accuracy, precision, recall, F1)
- Confusion matrix interpretation
- Type I/II error tradeoff analysis

### Data Visualization
- Static charts (Matplotlib, Seaborn)
- Interactive maps (Folium with marker clusters)
- Web dashboards (Plotly Dash)
- Geospatial analysis

### Software Engineering
- Jupyter notebook organization
- Modular code structure
- Documentation and commenting
- Version control best practices

---

## 🔮 Future Enhancements

### Data Expansion
- [ ] Incorporate weather data (wind speed, precipitation) at launch time
- [ ] Add booster reuse count as feature (1st flight vs. 10th flight)
- [ ] Include landing platform conditions (drone ship wave height)

### Model Improvements
- [ ] Ensemble methods (Random Forest, XGBoost, Gradient Boosting)
- [ ] Neural networks for non-linear pattern detection
- [ ] Time-series forecasting for launch success trends
- [ ] Feature importance analysis (SHAP values)

### Deployment
- [ ] Deploy Dash app to Heroku/Render
- [ ] Create REST API for predictions
- [ ] Build Streamlit interface with user input
- [ ] Docker containerization

---

## 📚 References

- [SpaceX API Documentation](https://github.com/r-spacex/SpaceX-API)
- [Falcon 9 Wikipedia](https://en.wikipedia.org/wiki/Falcon_9)
- [IBM Data Science Professional Certificate](https://www.coursera.org/professional-certificates/ibm-data-science)
- [Project Presentation (PDF)](./A_Race_To_SpaceY_Presentation.pdf)

---

## 👤 Author

**Brent Smith**
- Email: brents434@gmail.com
- LinkedIn: [linkedin.com/in/brent-smith-4d](https://linkedin.com/in/brent-smith-4d)
- GitHub: [@brent-smith-4](https://github.com/brent-smith-4)

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **IBM Data Science Professional Certificate** for project framework and guidance
- **SpaceX** for public API access
- **Wikipedia contributors** for comprehensive launch records

---

<div align="center">
  
### ⭐ If you found this project helpful, please consider giving it a star!

**Built with Python 🐍 | Powered by Machine Learning 🤖**

</div>
