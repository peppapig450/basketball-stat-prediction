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

    team_id_map = {team['abbreviation']: team['id'] for team in nba_teams}

    nba_team_ids = [team['id'] for team in nba_teams]
    return nba_team_ids, team_id_map


@app.cell
def _(PlayerGameLogs, SEASON, nba_team_ids, pd, roster_subset, team_id_map):
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

    # Convert the year column to Int16 for memory purposes
    career_cols = ['FROM_YEAR']
    master_data[career_cols] = master_data[career_cols].apply(pd.to_numeric, errors='coerce').astype('Int16')

    # We convert draft columns to nullable integers to ensure that undrafted players don't have null values
    draft_cols = ['DRAFT_YEAR', 'DRAFT_ROUND', 'DRAFT_NUMBER', 'FROM_YEAR']
    master_data[draft_cols] = master_data[draft_cols].apply(pd.to_numeric, errors='coerce').astype('Int16')

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
    master_data[bool_cols] = master_data[bool_cols].astype(bool)

    # Split the matchup column into a boolean for Home games and one for the opponent
    master_data['IS_HOME'] = master_data['MATCHUP'].str.contains('vs.')
    master_data['OPPONENT'] = master_data['MATCHUP'].str.split().str[-1]

    # Since this winds up as an object we need to convert it to a category
    master_data['OPPONENT'] = master_data['OPPONENT'].astype('category')

    # Map the opponent's abbreviation to their ID to allow for filtering by opponents
    master_data['OPP_TEAM_ID'] = master_data['OPPONENT'].map(team_id_map)

    # Drop any rows where the mapping failed (Guangzhou, Melbourne, Tel Aviv, etc..)
    # for some reason even tho it should be fetching regular season games only the api
    # returns pre-season games against international opponents
    master_data = master_data.dropna(subset=['OPP_TEAM_ID'])

    # Cast to integer instead of a float since a float is unneccessary
    master_data['OPP_TEAM_ID'] = master_data['OPP_TEAM_ID'].astype(int)

    master_data = master_data.drop(columns=['MATCHUP'])

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
