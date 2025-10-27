# Tennis Match Prediction using Machine Learning

This is my data science project to analyze and predict ATP tennis match outcomes. I use a gradient boosting model (XGBoost) and tested its performance via different metrics including F1 scores, direct application to a Grand Slam tournament and a backtesting simulation of different betting strategies. Whilst I arrived at first preliminary results this is still a work in progress.

# Project Goal

The goal of this project is to execute the entire data science workflow, from data acquisition to model implementation. It aimes to investigate whether modern machine learning models can identify a statistical advantage over the implied probabilities of betting odds.

# Workflow & Methodology

The process is divided into several Jupyter Notebooks:

1.  Exploratory Data Analysis (EDA):
    - Data inspection, analysis of missing values, and visualization of feature distributions.

2.  Data Preparation & Feature Engineering:
    - Cleaning the raw data and handling missing values.
    - Creation of new, model-ready features (e.g., `Rank_diff`, `ELO`, player form statistics, surface statistics, head2head statistics, fatigue).

3.  Model Training:
    - Training a Gradient Boosting Model (Model: XGBoost) to predict the match winner.
    - Hyperparameter tuning to optimize model accuracy.

4.  Model Evaluation:
    - Evaluating the model on an unseen test set.
    - Analysis of metrics such as Accuracy, Log-Loss, and F1-scores.
    - Iteration through different combinations of included features.

5.  Backtesting & Simulation: 
    * Simulation of different betting strategies based on the model's predictions.
    * Analysis of potential profitability combining betting strategies with different model training techniques.

# Analysis & Results

- Model Performance: The current model achieves an accuracy of 67% on the test set (including ATP data from 2024 and 2025), which is comparable to   IBM predictions and comparabe scientific literature which hover at around 70% accuracy.
  Notably the model is able to recreate the predictiveness of playing strenght metrics like rank or betting odds by only using in-game player        statistics which makes the encoded information in these proxies more comprehensive.
  --> It turns out that tennis is a surprisingly unpredictable sport with most prediction accuracies capped at around 70%, which reflects the   profound mental dimension of the game.
- Balancing relevance: The current best model is very biased towards predicting the favourite player (by rank). Balancing the training dataset
  can mitigate this but results in an accuracy trade-off between the expected and upset class. The overall F1-score and log-loss stay more or less   constant. This opens the possibility to customize the model for whichever class should be accurately predicted.
  <img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/e040213e-0616-439a-b95c-35c4da99c0b1" />

- US-Open 2025 performance: The model performed reasonably even though a series of upsets lowered its accuracy:
  <img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/c6ca885d-8a28-46b6-bd72-a6f5dc72cb60" />

- Backtesting Result: The simulation showed that only "smart" betting is a profitable strategy. That means that betting a constant fraction of the   bankroll on every game yields 0 net profit in the best case.
  <img width="796" height="455" alt="image" src="https://github.com/user-attachments/assets/e565ed61-51ec-45fd-a64a-53b5d7acc7f2" />
  Tilting the choice of taken bet opportunities towards higher model confidence however yielded a first significant increase. For a constant         betting fraction strategy this improved the profit to 40% of the initial investment:
  <img width="799" height="455" alt="image" src="https://github.com/user-attachments/assets/0174e457-5f24-49d1-9aff-9c89c97b2ce9" />
  Finally we also explored the impact of the different model balancings on the betting performance. Here we found that different betting          strategies prefer different data balancings. Whilst i) the constant fraction strategy preferred modest balancing ratios, ii) the Kelly criterion   strategy thrived on a perfectly balanced set:
  i):
  <img width="808" height="455" alt="image" src="https://github.com/user-attachments/assets/d3b79991-fee0-40e9-90bd-14bea2bcb749" />
  ii):
  <img width="808" height="455" alt="image" src="https://github.com/user-attachments/assets/e8226dc2-46b9-404f-831d-1e875ff58401" />


# Data used
  - ATP dataset from Tennismylife / TML - Database
  - betting odds from http://www.tennis-data.co.uk
