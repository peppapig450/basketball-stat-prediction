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
    OUT_FILE = 'test_nba_subset.csv'
    return (SEASON,)


@app.cell
def _(PlayerIndex, SEASON):
    roster_idx = PlayerIndex(season=SEASON).get_data_frames()[0]
    roster = roster_idx.rename(columns={'PERSON_ID': 'PLAYER_ID'})
    return (roster,)


@app.cell
def _(PlayerGameLogs, SEASON, pd, roster):
    logs_raw = PlayerGameLogs(
        season_nullable=SEASON,
        last_n_games_nullable=5
    ).get_data_frames()[0]

    master_data = pd.merge(logs_raw, roster, on="PLAYER_ID", how="left")
    master_data['GAME_DATE'] = pd.to_datetime(master_data["GAME_DATE"]).dt.date
    master_data
    return


if __name__ == "__main__":
    app.run()
