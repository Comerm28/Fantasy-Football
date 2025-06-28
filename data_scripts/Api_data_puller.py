import espn_api as espn
import nfl_data_py as  nfl
import pandas as pd

seasons = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

def ngs_data():
    passing_df = nfl.import_ngs_data('passing', seasons)
    rushing_df = nfl.import_ngs_data('rushing', seasons)
    receiving_df = nfl.import_ngs_data('receiving', seasons)

    passing_df.to_csv('data/ngs_passing_data.csv', index=False)
    rushing_df.to_csv('data/ngs_rushing_data.csv', index=False)
    receiving_df.to_csv('data/ngs_receiving_data.csv', index=False)

def seasonal_data():
    seasons = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    passing_df = nfl.import_season_data(seasons, 'passing')
    rushing_df = nfl.import_season_data(seasons, 'rushing')
    receiving_df = nfl.import_season_data(seasons, 'receiving')
    qbr_df = nfl.import_qbr(seasons)

    passing_df.to_csv('data/seasonal_passing_data.csv', index=False)
    rushing_df.to_csv('data/seasonal_rushing_data.csv', index=False)
    receiving_df.to_csv('data/seasonal_receiving_data.csv', index=False)
    qbr_df.to_csv('data/seasonal_qbr_data.csv', index=False)

def team_data():
    teams = nfl.import_win_totals(seasons)
    teams.to_csv('data/team_data.csv', index=False)

def pfr_data():
    pass_pfr = nfl.import_seasonal_pfr('pass', seasons)
    rush_pfr = nfl.import_seasonal_pfr('rush', seasons)
    rec_pfr = nfl.import_seasonal_pfr('receiving', seasons)

    pass_pfr.to_csv('data/pfr_pass_data.csv', index=False)
    rush_pfr.to_csv('data/pfr_rush_data.csv', index=False)
    rec_pfr.to_csv('data/pfr_receiving_data.csv', index=False)

def snap_counts():
    snap_counts = nfl.import_snap_counts(seasons)
    snap_counts.to_csv('data/snap_counts.csv', index=False)

# still need to pull espn data and clean for models, each model should handle its own dataset building


if __name__ == "__main__":
    ngs_data()
    seasonal_data()
    team_data()