import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from datetime import datetime
    import pandas as pd
    from nba_api.stats.endpoints import ScoreboardV3 as Scoreboard, LeagueDashPlayerStats

    return LeagueDashPlayerStats, Scoreboard, datetime


@app.cell
def _(Scoreboard, datetime):
    today = datetime.now().strftime('%Y-%m-%d')
    board = Scoreboard(game_date=today)

    teams_today_df = board.line_score.get_data_frame()
    active_team_ids = teams_today_df['teamId'].unique().tolist()

    return (active_team_ids,)


@app.cell
def _(LeagueDashPlayerStats, active_team_ids):
    recent_stats = LeagueDashPlayerStats(
        last_n_games=10,
        per_mode_detailed='PerGame',
    ).get_data_frames()[0]

    candidates = recent_stats[recent_stats['TEAM_ID'].isin(active_team_ids)].copy()

    # Logic: Look for players averaging > 3 makes (FG3M) and > 8 attempts (FG3A)
    high_probability = candidates.query(
        'FG3M >= 3.0 and FG3A >= 8.0 and GP >= 4'
    ).sort_values(by='FG3M', ascending=False)

    high_probability
    return


@app.cell
def _(high_prob):
    high_prob
    return


if __name__ == "__main__":
    app.run()
