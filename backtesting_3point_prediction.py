import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import altair as alt

    return alt, pd


@app.cell
def _(pd):
    # Make sure we sort the data by player and date so the rolling/expanding windows work properly
    df_backtest = pd.read_parquet('data/nba_gamelogs_2025_26.parquet')
    df_backtest = df_backtest.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
    return (df_backtest,)


@app.cell
def _(df_backtest):
    # Calculate team's defensive performances over time
    # We look at what the opponent of each team did defensively
    team_gp = df_backtest.groupby('OPP_TEAM_ID')

    # Calculate the rolling averages for both the league and the teams
    # We shift by 1 so we only know the stats PRIOR to the game
    df_backtest['opp_prev_3pa_allowed'] = team_gp['FG3A'].transform(lambda x: x.expanding().mean().shift(1))
    df_backtest['opp_prev_3p_pct_allowed'] = team_gp['FG3_PCT'].transform(lambda x: x.expanding().mean().shift(1))

    # Calculate league averages at each point time (we take the expanding mean of all games)
    df_backtest['league_avg_3pa'] = df_backtest['FG3A'].expanding().mean().shift(1)
    df_backtest['league_avg_3p_pct'] = df_backtest['FG3_PCT'].expanding().mean().shift(1)

    # Create the multiplies we'll use in our formula
    df_backtest['def_att_mult'] = df_backtest['opp_prev_3pa_allowed'] / df_backtest['league_avg_3pa']
    df_backtest['def_pct_mult'] = df_backtest['opp_prev_3p_pct_allowed'] / df_backtest['league_avg_3p_pct']

    # Fill any NaNs that may exist in early season games where averages aren't stable with 1.0 (Neutral)
    df_backtest[['def_att_mult', 'def_pct_mult']] = df_backtest[['def_att_mult', 'def_pct_mult']].fillna(1.0)

    df_backtest
    return


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


@app.cell
def _(clean_test_df, pd):
    # Setup our optimization loop
    optimization_results = []

    # We test C values from 1 to 30
    for c_val in range(1, 31):
        # Apply our Bayesian Formula
        # Pred = (n_recent * avg_recent + C * avg_reason) / (n_recent + C)
        # TODO: we can make our prediction more accurate by incorporating more priors
        base_pred = (
                (clean_test_df['recent_gp_before'] * clean_test_df['recent_avg_before']) + 
                (c_val * clean_test_df['season_avg_before'])
        ) / (clean_test_df['recent_gp_before'] + c_val)

        # Apply our defensive context
        # Final Pred = Base * Volume Multiplier * Efficiency Multiplier
        final_predictions = base_pred * clean_test_df['def_att_mult'] * clean_test_df['def_pct_mult']

        # Calculate MAE (Mean Absolute Error)
        mae = (clean_test_df['FG3M'] - final_predictions).abs().mean()

        # Calculate MSE (Mean Squared Error)
        mse = ((clean_test_df['FG3M'] - final_predictions) ** 2).mean()

        optimization_results.append({'C': c_val, 'MAE': mae, 'MSE': mse})

    # Find our winner
    results_df = pd.DataFrame(optimization_results)
    best_c_mae = results_df.loc[results_df['MAE'].idxmin(), 'C']
    best_c_mse = results_df.loc[results_df['MSE'].idxmin(), 'C']

    results_df
    return best_c_mae, results_df


@app.cell
def _(alt, best_c_mae, results_df):
    chart = alt.Chart(results_df).mark_line(point = True).encode(
        x='C:Q',
        y=alt.Y('MAE:Q', scale=alt.Scale(zero=False)),
        tooltip=['C', 'MAE']
    ).properties(
        title=f"Optimization Curve: Best C is {best_c_mae}",
        width=500
    )

    chart
    return (chart,)


@app.cell
def _(results_df):
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
    filename = f"backtest_3point_c_val_{git_hash}_{timestamp}.csv"
    file_path = output_dir / filename

    results_df.to_csv(file_path, index=False)
    return git_hash, output_dir, timestamp


@app.cell
def _(chart, git_hash, output_dir, timestamp):
    # We export in the Vega-Lite spec
    chart_filename = f"backtest_3point_c_val_chart_{git_hash}_{timestamp}.json"
    chart_path = output_dir / chart_filename

    chart.save(str(chart_path))
    return


if __name__ == "__main__":
    app.run()
