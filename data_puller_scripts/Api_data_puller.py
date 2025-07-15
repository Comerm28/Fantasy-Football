
import nfl_data_py as  nfl
import pandas as pd

seasons = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

def ngs_data():
    passing_df = nfl.import_ngs_data('passing', seasons)
    rushing_df = nfl.import_ngs_data('rushing', seasons)
    receiving_df = nfl.import_ngs_data('receiving', seasons)

    passing_df.to_csv('data/ngs_passing_data.csv', index=False)
    rushing_df.to_csv('data/ngs_rushing_data.csv', index=False)
    receiving_df.to_csv('data/ngs_receiving_data.csv', index=False)

def seasonal_data():
    season_df = nfl.import_seasonal_data(seasons, 'REG')
    # rushing_df = nfl.import_seasonal_data(seasons, 'rush')
    # receiving_df = nfl.import_seasonal_data(seasons, '')
    qbr_df = nfl.import_qbr(seasons)

    season_df.to_csv('data/seasonal_data.csv', index=False)
    # rushing_df.to_csv('data/seasonal_rushing_data.csv', index=False)
    # receiving_df.to_csv('data/seasonal_receiving_data.csv', index=False)
    qbr_df.to_csv('data/seasonal_qbr_data.csv', index=False)

def team_data():
    teams = nfl.import_schedules(seasons)
    teams.to_csv('data/team_data.csv', index=False)

def pfr_data():
    pass_pfr = nfl.import_seasonal_pfr('pass', seasons)
    rush_pfr = nfl.import_seasonal_pfr('rush', seasons)
    rec_pfr = nfl.import_seasonal_pfr('rec', seasons)

    pass_pfr.to_csv('data/pfr_pass_data.csv', index=False)
    rush_pfr.to_csv('data/pfr_rush_data.csv', index=False)
    rec_pfr.to_csv('data/pfr_receiving_data.csv', index=False)

def snap_counts():
    snap_counts = nfl.import_snap_counts(seasons)
    snap_counts.to_csv('data/snap_counts.csv', index=False)

def ids():
    id = nfl.import_ids()
    id.to_csv('data/id_to_player.csv', index=False)

def injuries():
    # for outlier purposes
    inj = nfl.import_injuries(seasons)
    inj.to_csv('data/inj.csv', index=False)

if __name__ == "__main__":
    ngs_data()
    # seasonal_data()
    # team_data()
    # pfr_data()
    # snap_counts()
    # ids()
    # injuries()