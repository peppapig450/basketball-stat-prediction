import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from nba_api.stats.endpoints import PlayerGameLogs, PlayerIndex
    from nba_api.stats.static.teams import get_teams

    return PlayerGameLogs, PlayerIndex, get_teams, pd


@app.cell
def _():
    SEASON = '2025-26'
    OUT_FILE = 'data/nba_gamelogs_2025_26.parquet'
    return OUT_FILE, SEASON


@app.cell
def _(PlayerIndex, SEASON):
    roster_idx = PlayerIndex(season=SEASON).get_data_frames()[0]
    roster = roster_idx.rename(columns={'PERSON_ID': 'PLAYER_ID'})

    roster_subset = roster[[
        'PLAYER_ID', 'POSITION', 'HEIGHT', 'COLLEGE', 'COUNTRY',
        'DRAFT_YEAR', 'DRAFT_ROUND', 'DRAFT_NUMBER', 'FROM_YEAR', 'TO_YEAR'
    ]]
    return roster, roster_subset


@app.cell
def _(get_teams):
    nba_teams = get_teams()

    nba_team_ids = [team['id'] for team in nba_teams]
    return (nba_team_ids,)


@app.cell
def _(PlayerGameLogs, SEASON, nba_team_ids, pd, roster_subset):
    logs_raw = PlayerGameLogs(
        league_id_nullable="00",
        season_nullable=SEASON,
    ).get_data_frames()[0]

    # filter the logs for valid team ids to ensure that we don't have any all-star game stuff in there
    clean_logs = logs_raw[logs_raw['TEAM_ID'].isin(nba_team_ids)]

    master_data = pd.merge(
        clean_logs,
        roster_subset,
        on="PLAYER_ID",
        how="inner"
    )
    master_data['GAME_DATE'] = pd.to_datetime(master_data["GAME_DATE"])

    # We convert draft columns to nullable integers to ensure that undrafted players don't have null values
    draft_cols = ['DRAFT_YEAR', 'DRAFT_ROUND', 'DRAFT_NUMBER']
    master_data[draft_cols] = master_data[draft_cols].apply(pd.to_numeric, errors='coerce').astype('Int64')

    # Convert reptitive types to categories
    category_cols = ['TEAM_ABBREVIATION', 'POSITION', 'WL', 'SEASON_YEAR']
    master_data = master_data.astype({col: 'category' for col in category_cols})

    # Get rid of unnecessary rank columns as we do not need them
    master_data = master_data.loc[:, ~master_data.columns.str.contains('_RANK')]

    # Drop other redundant columns 
    to_drop = ['WNBA_FANTASY_PTS', 'SEASON_YEAR', 'TO_YEAR']
    master_data = master_data.drop(columns=[col for col in to_drop if col in to_drop])

    # Use booleans for columns that have a 0 or 1 for false and true respectively
    bool_cols = ['DD2', 'TD3', 'AVAILABLE_FLAG']
    master_data.astype({col: bool for col in bool_cols})

    master_data
    return logs_raw, master_data


@app.cell
def _(logs_raw):
    logs_raw.info()
    return


@app.cell
def _(roster):
    roster.info()
    return


@app.cell
def _(master_data):
    master_data.info()
    return


@app.cell
def _(master_data):
    master_data.nunique()
    return


@app.cell
def _(OUT_FILE, master_data):
    master_data.to_parquet(
        OUT_FILE,
        engine='pyarrow',
        compression='zstd',
        index=False,
    )
    return


if __name__ == "__main__":
    app.run()
