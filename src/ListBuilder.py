import pandas as pd
import joblib

qb_input = pd.read_csv('premodel_data/qb_input.csv')
qb_input['tdratio'] = qb_input['Pass_TD'] / qb_input['interceptions'].replace(0, 1)
qb_input['ypc'] = qb_input['Pass_Yds'] / qb_input['Pass_Cmp'].replace(0, 1)
qb_input['agressivetoint'] = qb_input['aggressiveness'] / qb_input['Pass_Int_per_game'].replace(0, 1)
qb_input['ry_per_td'] = qb_input['Rush_Yds'] / qb_input['Pass_TD'].replace(0, 1)
qb_input = qb_input[qb_input['GS'] / qb_input['G'] >= .5]

te_input = pd.read_csv('premodel_data/te_input.csv')
te_input['targets_per_game'] = te_input['targets'] / te_input['G'].replace(0, 1)

rb_input = pd.read_csv('premodel_data/rb_input.csv')
rb_input = rb_input[rb_input['rush_yards_over_expected_per_att'].notna()]
rb_input['td_per_touch'] = rb_input['Rush_TD'] / rb_input['Touches'].replace(0, 1)
rb_input['losttofmb'] = rb_input['FmbLost'] / rb_input['Fmb'].replace(0, 1)
rb_input['Rec_to_Run'] = (rb_input['Rec_Rec_per_game'] * rb_input['Rec_Yds_per_game']) / (rb_input['Rush_Att'].replace(0, 1) * rb_input['Rush_Yds_per_game'].replace(0, 1))
rb_input['epr_per'] = rb_input['expected_rush_yards'] / rb_input['Rush_Att'].replace(0, 1)
rb_input['stacked_coeff'] = rb_input['percent_attempts_gte_eight_defenders'] / rb_input['Rush_Att'].replace(0, 1)
rb_input['wear_and_tear'] = (rb_input['Age'] * rb_input['Touches'].replace(0, 1)) / rb_input['G'].replace(0, 1)

wr_input = pd.read_csv('premodel_data/wr_input.csv')
wr_input['targets_per'] = wr_input['targets'] / wr_input['G'].replace(0, 1)
wr_input['td_to_catch'] = wr_input['rec_touchdowns'] / wr_input['Touches'].replace(0, 1)
wr_input = wr_input.dropna()
print(f"Number of qbs: {len(qb_input)}")
print(f"Number of tes: {len(te_input)}")
print(f"Number of rbs: {len(rb_input)}")
print(f"Number of wrs: {len(wr_input)}")

qb_model = joblib.load('models/qb.pkl')
te_model = joblib.load('models/te.pkl')
rb_model = joblib.load('models/rb.pkl')
wr_model = joblib.load('models/wr.pkl')

qb_list = []
te_list = []
rb_list = []
wr_list = []

qb_rows = ['Age', 'Pass_Yds_per_game', 'Rush_Y/A', 'Rush_Att_per_game',
               'avg_completed_air_yards', 'avg_intended_air_yards', 'avg_air_distance',
               'tdratio', 'ypc', 'agressivetoint', 'ry_per_td']
for index, row in qb_input.iterrows():
    prediction = qb_model.predict([row[qb_rows]])[0]
    qb_list.append({
        'Name': row['Name'],
        'Predicted_PPG': prediction
    })

rb_rows = ['wear_and_tear', 'rush_yards_over_expected_per_att',
                   'td_per_touch', 'losttofmb', 'Touches_per_game',
                   'avg_time_to_los', 'efficiency', 'Rec_to_Run',
                   'epr_per', 'rush_pct_over_expected', 'stacked_coeff']
for index, row in rb_input.iterrows():
    prediction = rb_model.predict([row[rb_rows]])[0]
    rb_list.append({
        'Name': row['Name'],
        'Predicted_PPG': prediction
    })


te_rows = ['Age', 'avg_cushion', 'targets_per_game', 'Rec_Yds_per_game',
                   'avg_separation','percent_share_of_intended_air_yards',
                   'Rec_TD_per_game', 'catch_percentage', 'avg_yac_above_expectation']
for index, row in te_input.iterrows():
    prediction = te_model.predict([row[te_rows]])[0]
    te_list.append({
        'Name': row['Name'],
        'Predicted_PPG': prediction
    })

wr_rows = ['catch_percentage', 'Rec_Rec_per_game', 'targets_per', 
                   'avg_yac_above_expectation', 'percent_share_of_intended_air_yards',
                   'td_to_catch', 'avg_separation', 'avg_cushion']
for index, row in wr_input.iterrows():
    prediction = wr_model.predict([row[wr_rows]])[0]
    wr_list.append({
        'Name': row['Name'],
        'Predicted_PPG': prediction
    })

qb_list = sorted(qb_list, key=lambda x: x['Predicted_PPG'], reverse=True)
rb_list = sorted(rb_list, key=lambda x: x['Predicted_PPG'], reverse=True)
te_list = sorted(te_list, key=lambda x: x['Predicted_PPG'], reverse=True)
wr_list = sorted(wr_list, key=lambda x: x['Predicted_PPG'], reverse=True)

print("Quarterback Predictions:")
for player in qb_list[:10]:
    print(f"{player['Name']}: {player['Predicted_PPG']} PPG")

print("\nRunning Back Predictions:")
for player in rb_list[:10]:
    print(f"{player['Name']}: {player['Predicted_PPG']} PPG")

print("\nTight End Predictions:")
for player in te_list[:10]:
    print(f"{player['Name']}: {player['Predicted_PPG']} PPG")

print("\nWide Receiver Predictions:")
for player in wr_list[:10]:
    print(f"{player['Name']}: {player['Predicted_PPG']} PPG")