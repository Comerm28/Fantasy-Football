import pandas as pd

def top_300_players_plus_df():
    top_300_df = pd.read_csv('data/espn_top_300.csv')
    players_df = pd.read_csv('data/fantasy_data.csv')
    result = pd.merge(top_300_df, players_df, on='Name', how='left')
    result.to_csv('premodel_data/base_300_stats.csv', index=False)

def ngs_preplus_qb_df():
    qb_df = pd.read_csv('data/ngs_passing_data.csv')
    qb_df = qb_df[qb_df['season_type'] == 'REG']
    avg_qb_df = qb_df.groupby(['player_display_name', 'season']).mean(numeric_only=True).reset_index()
    return avg_qb_df

def ngs_preplus_rb_df():
    rb_df = pd.read_csv('data/ngs_rushing_data.csv')
    rb_df = rb_df[rb_df['season_type'] == 'REG']
    avg_rb_df = rb_df.groupby(['player_display_name', 'season']).mean(numeric_only=True).reset_index()
    return avg_rb_df

def ngs_preplus_receiver_df():
    wr_df = pd.read_csv('data/ngs_receiving_data.csv')
    wr_df = wr_df[wr_df['season_type'] == 'REG']
    avg_wr_df = wr_df.groupby(['player_display_name', 'season']).mean(numeric_only=True).reset_index()
    return avg_wr_df

def separate_out_positions():
    top_300_players_plus_df = pd.read_csv('premodel_data/base_300_stats.csv')
    grouped = top_300_players_plus_df.groupby('Pos')

    qb_df = grouped.get_group('QB')
    rb_df = grouped.get_group('RB')
    wr_df = grouped.get_group('WR')
    te_df = grouped.get_group('TE')

    qb_df['Year'] = qb_df['Year'].astype(int)
    rb_df['Year'] = rb_df['Year'].astype(int)
    wr_df['Year'] = wr_df['Year'].astype(int)
    te_df['Year'] = te_df['Year'].astype(int)

    qb_df = qb_df[['Name', 'Team', 'Pos', 'Age', 'G', 'GS', 'Pass_Cmp', 'Pass_Att', 'Pass_Yds', 'Pass_TD', 'Pass_Int', 'Rush_Att', 'Rush_Yds', 'Rush_TD', 'Rush_Y/A', 'Fmb', 'FmbLost', 'Year', 'games_played_pct', 'games_started_pct', 'ProBowl', 'AllPro', 'Touches', 'Pass_Cmp%', 'Pass_Cmp_per_game', 'Pass_Att_per_game', 'Pass_Yds_per_game', 'Pass_TD_per_game', 'Pass_Int_per_game', 'Rush_Att_per_game', 'Rush_Yds_per_game', 'Rush_TD_per_game', 'Fmb_per_game', 'FmbLost_per_game', 'Touches_per_game', 'Points_half-ppr', 'PPG_half-ppr']]
    rb_df = rb_df[['Name', 'Team', 'Pos', 'Age', 'G', 'GS', 'Rush_Att', 'Rush_Yds', 'Rush_TD', 'Rush_Y/A', 'Rec_Rec', 'Rec_Yds', 'Rec_TD', 'Rec_Y/R', 'Fmb', 'FmbLost', 'Year', 'games_played_pct', 'games_started_pct', 'ProBowl', 'AllPro', 'Touches', 'Rush_Att_per_game', 'Rush_Yds_per_game', 'Rush_TD_per_game', 'Rec_Rec_per_game', 'Rec_Yds_per_game', 'Rec_TD_per_game', 'Fmb_per_game', 'FmbLost_per_game', 'Touches_per_game', 'Points_half-ppr', 'PPG_half-ppr']]
    wr_df = wr_df[['Name', 'Team', 'Pos', 'Age', 'G', 'GS', 'Rec_Rec', 'Rec_Yds', 'Rec_TD', 'Rec_Y/R', 'Rush_Att', 'Rush_Yds', 'Rush_TD', 'Fmb', 'FmbLost', 'Year', 'games_played_pct', 'games_started_pct', 'ProBowl', 'AllPro', 'Touches', 'Rec_Rec_per_game', 'Rec_Yds_per_game', 'Rec_TD_per_game', 'Rush_Att_per_game', 'Rush_Yds_per_game', 'Rush_TD_per_game', 'Fmb_per_game', 'FmbLost_per_game', 'Touches_per_game', 'Points_half-ppr', 'PPG_half-ppr']]
    te_df = te_df[['Name', 'Team', 'Pos', 'Age', 'G', 'GS', 'Rec_Rec', 'Rec_Yds', 'Rec_TD', 'Rec_Y/R', 'Fmb', 'FmbLost', 'Year', 'games_played_pct', 'games_started_pct', 'ProBowl', 'AllPro', 'Touches', 'Rec_Rec_per_game', 'Rec_Yds_per_game', 'Rec_TD_per_game', 'Fmb_per_game', 'FmbLost_per_game', 'Touches_per_game', 'Points_half-ppr', 'PPG_half-ppr']]

    ngs_preplus_qb = ngs_preplus_qb_df()
    ngs_preplus_rb = ngs_preplus_rb_df()
    ngs_preplus_receiver = ngs_preplus_receiver_df()

    pd.merge(qb_df, ngs_preplus_qb, left_on=['Name', 'Year'], right_on=['player_display_name', 'season'], how='left').to_csv('premodel_data/qb_pre.csv', index=False)
    pd.merge(rb_df, ngs_preplus_rb, left_on=['Name', 'Year'], right_on=['player_display_name', 'season'], how='left').to_csv('premodel_data/rb_pre.csv', index=False)
    pd.merge(wr_df, ngs_preplus_receiver, left_on=['Name', 'Year'], right_on=['player_display_name', 'season'], how='left').to_csv('premodel_data/wr_pre.csv', index=False)
    pd.merge(te_df, ngs_preplus_receiver, left_on=['Name', 'Year'], right_on=['player_display_name', 'season'], how='left').to_csv('premodel_data/te_pre.csv', index=False)

def shift_player_data():
    qb_df = pd.read_csv('premodel_data/qb_pre.csv')
    rb_df = pd.read_csv('premodel_data/rb_pre.csv')
    wr_df = pd.read_csv('premodel_data/wr_pre.csv')
    te_df = pd.read_csv('premodel_data/te_pre.csv')
   
    qb_df = qb_df.sort_values(by=['Name', 'Year'])
    rb_df = rb_df.sort_values(by=['Name', 'Year'])
    wr_df = wr_df.sort_values(by=['Name', 'Year'])
    te_df = te_df.sort_values(by=['Name', 'Year'])

    columns_to_shift_qb = ['Pass_Yds', 'Rush_Yds', 'Points_half-ppr', 'PPG_half-ppr']
    columns_to_shift_rb = ['Rush_Yds', 'Rec_Yds', 'Points_half-ppr', 'PPG_half-ppr']
    columns_to_shift_rec = ['Rec_Yds', 'Points_half-ppr', 'PPG_half-ppr']
    for col in columns_to_shift_qb:
        qb_df[f'next_year_{col}'] = qb_df.groupby('Name')[col].shift(-1)
    for col in columns_to_shift_rb:
        rb_df[f'next_year_{col}'] = rb_df.groupby('Name')[col].shift(-1)
    for col in columns_to_shift_rec:
        wr_df[f'next_year_{col}'] = wr_df.groupby('Name')[col].shift(-1)
        te_df[f'next_year_{col}'] = te_df.groupby('Name')[col].shift(-1)

    qb_df.to_csv('premodel_data/qb_pre_shifted.csv', index=False)
    rb_df.to_csv('premodel_data/rb_pre_shifted.csv', index=False)
    wr_df.to_csv('premodel_data/wr_pre_shifted.csv', index=False)
    te_df.to_csv('premodel_data/te_pre_shifted.csv', index=False)

def finalize_and_get_input_data():
    #need to remove injuries

    qb_df = pd.read_csv('premodel_data/qb_pre_shifted.csv')
    rb_df = pd.read_csv('premodel_data/rb_pre_shifted.csv')
    wr_df = pd.read_csv('premodel_data/wr_pre_shifted.csv')
    te_df = pd.read_csv('premodel_data/te_pre_shifted.csv')

    qb_df_input = qb_df[~qb_df['next_year_Pass_Yds'].notnull()]
    qb_df_input = qb_df_input.drop(columns=['next_year_Pass_Yds', 'next_year_Rush_Yds', 'next_year_PPG_half-ppr', 'next_year_Points_half-ppr'])
    qb_df_training = qb_df[qb_df['next_year_Pass_Yds'].notnull()]
    qb_df_training = qb_df_training[~qb_df_training['avg_time_to_throw'].isnull()]

    rb_df_input = rb_df[~rb_df['next_year_Rush_Yds'].notnull()]
    rb_df_input = rb_df_input.drop(columns=['next_year_Rush_Yds', 'next_year_Rec_Yds', 'next_year_PPG_half-ppr', 'next_year_Points_half-ppr'])
    rb_df_training = rb_df[rb_df['next_year_Rush_Yds'].notnull()]
    rb_df_training = rb_df_training[~rb_df_training['efficiency'].isnull()]

    wr_df_input = wr_df[~wr_df['next_year_Rec_Yds'].notnull()]
    wr_df_input = wr_df_input.drop(columns=['next_year_Rec_Yds', 'next_year_PPG_half-ppr', 'next_year_Points_half-ppr'])
    wr_df_training = wr_df[wr_df['next_year_Rec_Yds'].notnull()]
    wr_df_training = wr_df_training[~wr_df_training['avg_cushion'].isnull()]

    te_df_input = te_df[~te_df['next_year_Rec_Yds'].notnull()]
    te_df_input = te_df_input.drop(columns=['next_year_Rec_Yds', 'next_year_PPG_half-ppr', 'next_year_Points_half-ppr'])
    te_df_training = te_df[te_df['next_year_Rec_Yds'].notnull()]
    te_df_training = te_df_training[~te_df_training['avg_cushion'].isnull()]

    qb_df_input.to_csv('premodel_data/qb_input.csv', index=False)
    qb_df_training.to_csv('premodel_data/qb_training.csv', index=False)

    rb_df_input.to_csv('premodel_data/rb_input.csv', index=False)
    rb_df_training.to_csv('premodel_data/rb_training.csv', index=False)

    wr_df_input.to_csv('premodel_data/wr_input.csv', index=False)
    wr_df_training.to_csv('premodel_data/wr_training.csv', index=False)

    te_df_input.to_csv('premodel_data/te_input.csv', index=False)
    te_df_training.to_csv('premodel_data/te_training.csv', index=False)

    print("Data preparation complete. Input and training datasets are ready.")
    print("Shape: ")
    print("QB Input Shape:", qb_df_input.shape)
    print("QB Training Shape:", qb_df_training.shape)
    print("RB Input Shape:", rb_df_input.shape)
    print("RB Training Shape:", rb_df_training.shape)
    print("WR Input Shape:", wr_df_input.shape)
    print("WR Training Shape:", wr_df_training.shape)
    print("TE Input Shape:", te_df_input.shape)
    print("TE Training Shape:", te_df_training.shape)

if __name__ == "__main__":
    top_300_players_plus_df()
    separate_out_positions()
    shift_player_data()
    finalize_and_get_input_data()




