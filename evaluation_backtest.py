import marimo

__generated_with = "0.21.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt
    import numpy as np

    from pathlib import Path
    from datetime import datetime

    return Path, alt, datetime, mo, np, pd


@app.cell
def _(Path, datetime, pd):
    def get_run_history(directory: str = "experiments"):
        path = Path(directory)
        history = []

        for file in path.glob("*.csv"):
            # Format: NAME_backtest_3point_HASH_YYYYMMDD_HHMM.csv
            match file.stem.split('_'):
                case [*name_parts, "backtest", "3point", _, date_str, time_str]:
                    try:
                        dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M")

                        # Handle the case where the name parts contain underscores
                        run_name = "_".join(name_parts)

                        history.append({"run_name": run_name, "timestamp": dt, "path": file})

                    except ValueError:
                        continue

        return pd.DataFrame(history)

    return (get_run_history,)


@app.cell
def _(get_run_history, mo):
    history_df = get_run_history()

    unique_names = sorted(history_df["run_name"].unique())

    run_dropdown = mo.ui.dropdown(
        options=unique_names,
        label="Select Experiment Series",
        value=unique_names[-1] if unique_names else None
    )
    return history_df, run_dropdown


@app.cell
def _(history_df, mo, run_dropdown):
    versions = history_df[history_df["run_name"] == run_dropdown.value].sort_values("timestamp", ascending=False)

    version_picker = mo.ui.table(
        versions,
        selection="single",
        label="Select specific version to analyze"
    )

    mo.vstack([
        run_dropdown,
        mo.md("### Available Timestamps"),
        version_picker
    ])
    return (version_picker,)


@app.cell
def _(mo, pd, version_picker):
    # Since the table is backed by a DataFrame and marimo struggles with the ambiguity we have to do it like this
    val = version_picker.value

    if (not val.empty) if hasattr(val, 'empty') else val:
        selected_path = val.iloc[0]["path"] if hasattr(val, 'iloc') else val[0]["path"]
        results_df = pd.read_csv(selected_path)
    else:
        mo.md("Please select a version from the table above.")

    results_df
    return (results_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    copy some of the logic from the backtest notebook to get clean_test_df and df_backtest
    """)
    return


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


@app.cell
def _(clean_test_df, df_backtest, np, pd, results_df):
    # Calibration loop
    best_row = results_df.loc[results_df['combined_score'].idxmin()]
    c_val = best_row['C']
    w_att = best_row['W_Att']
    t_span = best_row['Team_Span']
    l_span = best_row['League_Span']

    # Perform the same operations we do in our optimization loop
    team_game_stats = (
        df_backtest
            .groupby(['OPP_TEAM_ID', 'GAME_ID', 'GAME_DATE'], as_index=False)
            .agg(
                total_fg3a=('FG3A', 'sum')
            )
    )

    league_avg = (
        team_game_stats['total_fg3a']
            .ewm(span=l_span, adjust=False)
            .mean()
            .shift(1)
    )

    team_prev_allowed = (
        team_game_stats
            .groupby('OPP_TEAM_ID')['total_fg3a']
            .transform(
                lambda x: x.ewm(span=t_span, adjust=False)
                            .mean()
                            .shift(1)
            )
    )

    team_game_stats['att_mult'] = (team_prev_allowed / league_avg).fillna(1.0)

    merged_calib = (
        clean_test_df
            .merge(
                team_game_stats[['GAME_ID', 'OPP_TEAM_ID', 'att_mult']],
                on=['GAME_ID', 'OPP_TEAM_ID'],
                how='left'
            )
    )

    base_pred = (
            (merged_calib['recent_gp_before'] * merged_calib['recent_avg_before']) + 
            (c_val * merged_calib['season_avg_before'])
    ) / (merged_calib['recent_gp_before'] + c_val)

    merged_calib['final_pred'] = base_pred * (merged_calib['att_mult'] ** w_att)
    merged_calib['actual'] = merged_calib['FG3M']

    # First bin approach that is simplified
    merged_calib['bin_fixed'] = np.floor(merged_calib['final_pred'] * 2) / 2
    stats_fixed = (
        merged_calib
            .groupby('bin_fixed', as_index=False)
            .agg(
                avg_actual=('actual', 'mean'),
                avg_pred=('final_pred', 'mean'),
                sample_size=('actual', 'count'),
            )
            .rename(columns={'bin_fixed': 'bin'})
    )

    # Second bin approach, doing it quantile style with equally sized bins
    merged_calib['bin_q'] = pd.qcut(merged_calib['final_pred'], q=10, labels=False, duplicates='drop')
    stats_q = (
        merged_calib
            .groupby('bin_q', as_index=False)
            .agg(
                avg_actual=('actual', 'mean'),
                avg_pred=('final_pred', 'mean'),
                sample_size=('actual', 'count'),
            )
            .rename(columns={'bin_q': 'bin'})
    )
    return stats_fixed, stats_q


@app.cell
def _(alt, pd, stats_fixed, stats_q):
    def make_chart(data, title, x_label):
        line = alt.Chart(pd.DataFrame({'x': [0, 5], 'y': [0, 5]})).mark_line(color='red', strokeDash=[5,5]).encode(x='x', y='y')
        points = alt.Chart(data[data['sample_size'] > 15]).mark_circle(size=70).encode(
            x=alt.X('avg_pred', title=x_label),
            y=alt.Y('avg_actual', title='Actual Made 3s'),
            size='sample_size',
            tooltip=['avg_actual', 'avg_pred', 'sample_size']
        )

        return (line + points).properties(title=title, width=300, height=300)

    chart_fixed = make_chart(stats_fixed, "Fixed Bins (Matches Prop Lines)", "Predicted 3PM")
    chart_q = make_chart(stats_q, "Quantile Bins (Equal Data)", "Predicted 3PM")
    return chart_fixed, chart_q


@app.cell
def _(chart_fixed):
    chart_fixed
    return


@app.cell
def _(chart_q):
    chart_q
    return


if __name__ == "__main__":
    app.run()
