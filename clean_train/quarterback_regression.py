# cols: Name,Team,Pos,Age,G,GS,Pass_Cmp,Pass_Att,Pass_Yds,Pass_TD,Pass_Int,Rush_Att,Rush_Yds,Rush_TD,Rush_Y/A,Fmb,FmbLost,Year,games_played_pct,games_started_pct,ProBowl,AllPro,Touches,Pass_Cmp%,Pass_Cmp_per_game,Pass_Att_per_game,Pass_Yds_per_game,Pass_TD_per_game,Pass_Int_per_game,Rush_Att_per_game,Rush_Yds_per_game,Rush_TD_per_game,Fmb_per_game,FmbLost_per_game,Touches_per_game,Points_half-ppr,PPG_half-ppr,player_display_name,season,week,avg_time_to_throw,avg_completed_air_yards,avg_intended_air_yards,avg_air_yards_differential,aggressiveness,max_completed_air_distance,avg_air_yards_to_sticks,attempts,pass_yards,pass_touchdowns,interceptions,passer_rating,completions,completion_percentage,expected_completion_percentage,completion_percentage_above_expectation,avg_air_distance,max_air_distance,player_jersey_number,next_year_Pass_Yds,next_year_Rush_Yds,next_year_Points_half-ppr,next_year_PPG_half-ppr

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

training_data = pd.read_csv('premodel_data/qb_training.csv')
scaler = MinMaxScaler()

# feature engineering
training_data['tdratio'] = training_data['Pass_TD'] / training_data['interceptions'].replace(0, 1)
training_data['ypc'] = training_data['Pass_Yds'] / training_data['Pass_Cmp'].replace(0, 1)
training_data['agressivetoint'] = training_data['aggressiveness'] / training_data['Pass_Int_per_game'].replace(0, 1)
training_data['ry_per_td'] = training_data['Rush_Yds'] / training_data['Pass_TD'].replace(0, 1)
training_data = training_data[training_data['GS'] / training_data['G'] >= .5]
print(f"Number of players with at least 50% games started: {len(training_data)}")

# 'PPG_half-ppr',
X = training_data[['Age', 'Pass_Yds_per_game', 'Rush_Y/A', 'Rush_Att_per_game',
                   'avg_completed_air_yards', 'avg_intended_air_yards', 'avg_air_distance',
                   'tdratio', 'ypc', 'agressivetoint', 'ry_per_td']]
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

joblib.dump(model, 'models/qb.pkl')

#random forest(more important part as forests have importance) regressor
# importances = model.feature_importances_
# feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
# print(feature_importance_df.sort_values(by='Importance', ascending=False))