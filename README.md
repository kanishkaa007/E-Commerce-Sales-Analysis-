# E-Commerce-Sales-Analysis-
End-to-end data pipeline simulating e-commerce order data, performing EDA, and training predictive models (Linear Regression &amp; Random Forest) to optimize inventory and supply chain strategy.

## Project Overview
Managing inventory effectively is a critical challenge in e-commerce. Overstocking increases storage costs, while understocking leads to lost revenue. 
This project addresses that challenge by:
1. **Generating a realistic synthetic dataset** of 1,000 orders containing product types, pricing, order dates, and regions.
2. **Performing deep-dive EDA** to visualize temporal trends, top-performing items, and regional revenue share.
3. **Training Machine Learning models** to forecast unit demand (`Units_Sold`) based on product attributes and time factors.
4. **Deriving business insights** to help supply chain managers make data-driven inventory decisions.

---
## Tech Stack & Libraries Used
* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn` (Linear Regression, Random Forest Regressor)
---
## Exploratory Data Analysis (EDA)
The project generates four key visual insights:
1. **Monthly Sales Trend:** Tracks overall revenue fluctuations across the year.
2. **Top-Selling Products:** Identifies highest volume items using horizontal bar plots.
3. **Regional Distribution:** Analyzes pie-chart breakdowns of total revenue across North, East, South, and West regions.
4. **Seasonal Product Demand:** Evaluates quarterly demand per product category to pinpoint seasonal spikes.
---
## ML Pipeline

### Feature Engineering & Preprocessing
* Extracted temporal features (`Month`, `DayOfWeek`, `Quarter`).
* Applied **One-Hot Encoding** (`pd.get_dummies`) to categorical variables (`Product`, `Region`).
* Split data into **80% Training** and **20% Testing** sets.

### Models Implemented
* **Linear Regression:** Baseline model for linear relationships.
* **Random Forest Regressor:** Non-linear ensemble model with 100 decision trees.

### Evaluation Metrics
Models were evaluated using:
* **Root Mean Squared Error (RMSE)**
* **R² Score**
---
Key  Business Insights

* **Stocking Strategy:** Focus inventory capital on high-volume products identified in the demand analysis.
* **Logistics Focus:** Allocate supply chain resources to top-revenue regions highlighted by geographic distribution.
* **Seasonal Buffers:** Build stock buffers approximately 1 month prior to high-volume quarters detected in seasonal plots.

---
