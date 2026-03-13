import marimo

__generated_with = "0.21.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import altair as alt

    return mo, np, pd


@app.cell
def _(mo):
    run_name_input = mo.ui.text(
        label="Run name",
        placeholder="testing",
    )
    run_name_input
    return (run_name_input,)


@app.cell
def _(run_name_input):
    run_name = run_name_input.value or "unnamed_run"
    return (run_name,)


@app.cell
def _(pd):
    # Make sure we sort the data by player and date so the rolling/expanding windows work properly
    df_backtest = pd.read_parquet('data/nba_gamelogs_2025_26.parquet')
    df_backtest = df_backtest.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
    return (df_backtest,)


@app.cell
def _(df_backtest):
    # Generate Point-in-Time features
    # We shift by 1 so that the prediction for tonight only knows about past games
    gp = df_backtest.groupby('PLAYER_ID')['FG3M']

    # Historical Season Average (expanding mean)
    df_backtest['season_avg_before'] = gp.transform(lambda x: x.expanding().mean().shift(1))

    # Recent form (last 10 games mean)
    df_backtest['recent_avg_before'] = gp.transform(lambda x: x.rolling(window=10).mean().shift(1))

    # Count how many games have been played in the recent window (this handles players with less than 10 games played)
    df_backtest['recent_gp_before'] = gp.transform(lambda x: x.rolling(window=10).count().shift(1))

    # Drop the first game of the season for every player (where season_avg is NaN)
    # Also drop games where recent_gp_before is 0
    clean_test_df = df_backtest.dropna(subset=['season_avg_before', 'recent_avg_before']).copy()
    return (clean_test_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    since our most optimal team spans and league spans are 20 and 200, we can try going even lower to see if that offers further improvements
    """)
    return


@app.cell
def _(clean_test_df, df_backtest, np, pd):
    # Test C values, defensive weights, and spans for EWMA

    c_values = range(1,31)

    att_weights = np.linspace(0.1, 0.8, 12) # Test volume importance

    # The span value is reflective of how much we want each N amount of games to dominate the average
    # so naturally it is much lower for individual teams than league-wide
    team_spans = np.arange(10, 35, 5)
    league_spans = [150, 200, 250, 300]

    team_game_stats = (
        df_backtest
            .groupby(['OPP_TEAM_ID', 'GAME_ID', 'GAME_DATE'], as_index=False)
            .agg(
                total_fg3a=('FG3A', 'sum'),
            )
    )

    team_game_stats = team_game_stats.sort_values('GAME_DATE')

    optimization_results = []

    team_gp = team_game_stats.groupby('OPP_TEAM_ID')

    for league_span in league_spans:

        # First we calculate the EWMA for each span value for the league
        league_avg_3pa = (
            team_game_stats['total_fg3a']
            .ewm(span=league_span, adjust=False)
            .mean()
            .shift(1)
        )

        for team_span in team_spans:

            # Team defensive EWMA
            opp_prev_3pa_allowed = (
                team_gp['total_fg3a']
                .transform(lambda x: x.ewm(span=team_span, adjust=False).mean().shift(1))
            )

            # Caclulate the defense multipliers at the team-game level
            team_game_stats['att_mult'] = (opp_prev_3pa_allowed / league_avg_3pa).fillna(1.0)

            # Map our multipliers back to the player-level dataframe
            # We use a temporary merge to align the multipliers with clean_test_df
            multiplier_map = team_game_stats[['GAME_ID', 'OPP_TEAM_ID', 'att_mult']]
            merged_df = clean_test_df.merge(multiplier_map, on=['GAME_ID', 'OPP_TEAM_ID'], how='left')

            curr_att_mult = merged_df['att_mult'].values

            # We test C values from 1 to 30
            for c_val in c_values:
                # Apply our Bayesian Formula
                # Pred = (n_recent * avg_recent + C * avg_reason) / (n_recent + C)
                # TODO: we can make our prediction more accurate by incorporating more priors
                base_pred = (
                        (clean_test_df['recent_gp_before'] * clean_test_df['recent_avg_before']) + 
                        (c_val * clean_test_df['season_avg_before'])
                ) / (clean_test_df['recent_gp_before'] + c_val)

                for w_att in att_weights:

                    # Apply power-law damping
                    adj_att_mult = curr_att_mult ** w_att

                    final_predictions = base_pred * adj_att_mult

                    # Calculate MAE (Mean Absolute Error)
                    mae = (clean_test_df['FG3M'] - final_predictions).abs().mean()

                    # Calculate MSE (Mean Squared Error)
                    mse = ((clean_test_df['FG3M'] - final_predictions) ** 2).mean()

                    optimization_results.append({
                        'C': c_val,
                        'W_Att': w_att,
                        'Team_Span': team_span,
                        'League_Span': league_span,
                        'MAE': mae,
                        'MSE': mse
                    })

    # Find the best combination
    results_df = pd.DataFrame(optimization_results)

    best_row = results_df.loc[results_df['MAE'].idxmin()]

    best_c = best_row['C']
    best_w_att = best_row['W_Att']
    best_team_span = best_row['Team_Span']
    best_league_span = best_row['League_Span']

    results_df
    return (results_df,)


@app.cell
def _(results_df, run_name):
    from pathlib import Path
    import subprocess
    from datetime import datetime, UTC

    output_dir = Path("experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Grab the Git HEAD hash to use in the filename
    git_hash = subprocess.run(
        ['git', 'rev-parse', '--short', 'HEAD'],
        capture_output=True,
        text=True,
    ).stdout.strip()

    # We use UTC to be extra proper
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M")
    filename = f"{run_name}_backtest_3point_{git_hash}_{timestamp}.csv"
    file_path = output_dir / filename

    results_df.to_csv(file_path, index=False)
    return git_hash, output_dir, timestamp


@app.cell
def _(git_hash, output_dir, run_name, timestamp):
    # We export in the Vega-Lite spec
    chart_filename = f"{run_name}_backtest_3point_chart_{git_hash}_{timestamp}.json"
    chart_path = output_dir / chart_filename

    # chart.save(str(chart_path))
    return


if __name__ == "__main__":
    app.run()
