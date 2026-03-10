import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from nba_api.stats.endpoints import PlayerGameLogs, PlayerIndex

    return (PlayerIndex,)


@app.cell
def _():
    SEASON = '2025-26'
    OUT_FILE = 'test_nba_subset.csv'
    return (SEASON,)


@app.cell
def _(PlayerIndex, SEASON):
    roster_idx = PlayerIndex(season=SEASON).get_data_frames()[0]
    roster = roster_idx.rename(columns={'PERSON_ID': 'PLAYER_ID'})
    return


app._unparsable_cell(
    r"""
    logs_raw = PlayerGameLogs(
        season_nullable=SEASON,
        seas
    )
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
