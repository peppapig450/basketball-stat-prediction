import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from nba_api.stats.endpoints import PlayerGameLogs, PlayerIndex

    return PlayerGameLogs, PlayerIndex, pd


@app.cell
def _():
    SEASON = '2025-26'
    OUT_FILE = 'nba_gamelogs_2025_26.parquet'
    return (SEASON,)


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
def _(PlayerGameLogs, SEASON, pd, roster_subset):
    logs_raw = PlayerGameLogs(
        league_id_nullable="00",
        season_nullable=SEASON,
        last_n_games_nullable=5
    ).get_data_frames()[0]

    master_data = pd.merge(
        logs_raw,
        roster_subset,
        on="PLAYER_ID",
        how="inner"
    )
    master_data['GAME_DATE'] = pd.to_datetime(master_data["GAME_DATE"]).dt.date

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
    null_teams = master_data[master_data['TEAM_ABBREVIATION'].isna()]
    null_teams[['GAME_DATE', 'PLAYER_NAME', 'MATCHUP', 'TEAM_ID']].head(20)
    return


if __name__ == "__main__":
    app.run()
