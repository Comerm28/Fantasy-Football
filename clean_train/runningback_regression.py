# cols: Age,G,GS,Rush_Att,Rush_Yds,Rush_TD,Rush_Y/A,Rec_Rec,Rec_Yds,Rec_TD,Rec_Y/R,Fmb,FmbLost,Year,games_played_pct,games_started_pct,ProBowl,AllPro,Touches,Rush_Att_per_game,Rush_Yds_per_game,Rush_TD_per_game,Rec_Rec_per_game,Rec_Yds_per_game,Rec_TD_per_game,Fmb_per_game,FmbLost_per_game,Touches_per_game,Points_half-ppr,PPG_half-ppr,player_display_name,season,week,efficiency,percent_attempts_gte_eight_defenders,avg_time_to_los,rush_attempts,rush_yards,avg_rush_yards,rush_touchdowns,player_jersey_number,expected_rush_yards,rush_yards_over_expected,rush_yards_over_expected_per_att,rush_pct_over_expected,next_year_Rush_Yds,next_year_Rec_Yds,next_year_Points_half-ppr,next_year_PPG_half-ppr

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import joblib

training_data = pd.read_csv('premodel_data/rb_training.csv')
training_data = training_data[training_data['rush_yards_over_expected_per_att'].notna()]
scaler = MinMaxScaler()

# feature engineering
training_data['td_per_touch'] = training_data['Rush_TD'] / training_data['Touches'].replace(0, 1)
training_data['losttofmb'] = training_data['FmbLost'] / training_data['Fmb'].replace(0, 1)
training_data['Rec_to_Run'] = (training_data['Rec_Rec_per_game'] * training_data['Rec_Yds_per_game']) / (training_data['Rush_Att'].replace(0, 1) * training_data['Rush_Yds_per_game'].replace(0, 1))
training_data['epr_per'] = training_data['expected_rush_yards'] / training_data['Rush_Att'].replace(0, 1)
training_data['stacked_coeff'] = training_data['percent_attempts_gte_eight_defenders'] / training_data['Rush_Att'].replace(0, 1)
training_data['wear_and_tear'] = (training_data['Age'] * training_data['Touches'].replace(0, 1)) / training_data['G'].replace(0, 1)
print(f"Number of players with at least 50% games started: {len(training_data)}")

X = training_data[['wear_and_tear', 'rush_yards_over_expected_per_att',
                   'td_per_touch', 'losttofmb', 'Touches_per_game',
                   'avg_time_to_los', 'efficiency', 'Rec_to_Run',
                   'epr_per', 'rush_pct_over_expected', 'stacked_coeff']]
# X = scaler.fit_transform(X)
y = training_data['next_year_PPG_half-ppr']

#polynomial fitting
# poly = PolynomialFeatures(degree=2, include_bias=False)
# poly_features = poly.fit_transform(X[['Age']])
# poly_df = pd.DataFrame(poly_features, columns=['Age', 'Age^2'])
# X = pd.concat([X.drop(columns=['Age']), poly_df], axis=1)

#hyper parameter tuning
# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 5, 10],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5, scoring='r2')
# grid_search.fit(X_train, y_train)

# best_model = grid_search.best_estimator_
# print("Best Parameters:", grid_search.best_params_)

#model selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = LinearRegression(), try thielsenregressor
model = GradientBoostingRegressor(n_estimators=20, learning_rate=0.1, max_depth=6, min_samples_split=10, min_samples_leaf=8, subsample=1.0, random_state=42)
# model = VotingRegressor(estimators=[
#     ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
#     ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
#     ('ts', TheilSenRegressor())
# ])
# model = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_split=10, max_features='sqrt', random_state=42)
# model = TheilSenRegressor(n_jobs=-1)
model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
train_mse = mean_squared_error(y_train, train_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)
print("\n===== Model Evaluation =====")
print(f"Training MSE: {train_mse:.4f}")
print(f"Testing MSE: {test_mse:.4f}")
print(f"Training R²: {train_r2:.4f}")
print(f"Testing R²: {test_r2:.4f}")

joblib.dump(model, 'models/rb.pkl')

#random forest(more important part as forests have importance) regressor
# importances = model.feature_importances_
# feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
# print(feature_importance_df.sort_values(by='Importance', ascending=False))