import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import pandas as pd
    from datetime import datetime
    from nba_api.stats.endpoints import ScoreboardV3 as Scoreboard, LeagueDashPlayerStats


@app.cell
def _():
    today = datetime.now().strftime('%Y-%m-%d')
    board = Scoreboard(game_date='2026-03-09')

    teams_today_df = board.line_score.get_data_frame()
    active_team_ids = teams_today_df['teamId'].unique().tolist()
    return (active_team_ids,)


@app.cell
def _():
    # Slider for the number of 3s we are looking for
    target_3s = mo.ui.slider(start=1.0, stop=6.0, step=0.5, value=3.0, label="Target 3PM")

    # Slider for Bayesian Weight which controls how much we trust the season average
    confidence_weight = mo.ui.slider(start=1, stop=30, step=1, value=10, label="Season Weight (C)")

    mo.vstack([target_3s, confidence_weight])
    return confidence_weight, target_3s


@app.cell
def _(active_team_ids, confidence_weight, target_3s):
    season_stats_df = LeagueDashPlayerStats(
        per_mode_detailed='PerGame'
    ).get_data_frames()[0][['PLAYER_ID', 'PLAYER_NAME', 'FG3M', 'FG3A']]

    recent_stats_df = LeagueDashPlayerStats(
        last_n_games=10,
        per_mode_detailed='PerGame'
    ).get_data_frames()[0][['PLAYER_ID', 'GP', 'FG3M', 'FG3A', 'TEAM_ID']]

    combined_stats_df = (
        pd.merge(
            recent_stats_df,
            season_stats_df,
            on='PLAYER_ID',
            suffixes=('_recent', '_season')
        )
        .loc[lambda x: x['TEAM_ID'].isin(active_team_ids)]
        .copy()
    )

    # Apply Bayesian Shrinkage using our confidence value
    C = confidence_weight.value
    combined_stats_df['Bayesian_FG3M'] = (
        (combined_stats_df['GP'] * combined_stats_df['FG3M_recent']) + (C * combined_stats_df['FG3M_season'])
    ) / (combined_stats_df['GP'] + C)

    # Filter our results based on our dynamic target, looking for players who Bayesian average is >= than our target
    results_df = combined_stats_df[combined_stats_df['Bayesian_FG3M'] >= target_3s.value].copy()
    results_df['Bayesian_FG3M'] = results_df['Bayesian_FG3M'].round(2)

    # Sort by the highest likelihood
    results_df = results_df.sort_values(by='Bayesian_FG3M', ascending=False)
    results_df
    return


if __name__ == "__main__":
    app.run()
