import pandas as pd
from nba_api.stats.endpoints import PlayerGameLogs, PlayerIndex

SEASON = "2025-26"
OUT_FILE = "nba_gamelogs_2025_26.parquet"


def main():
    print(f"Fetching data for {SEASON}...")

    roster_idx: pd.DataFrame = PlayerIndex(season=SEASON).get_data_frames()[0]
    roster = roster_idx.rename(columns={"PERSON_ID": "PLAYER_ID"})

    logs_raw = PlayerGameLogs(
        season_nullable=SEASON,
    ).get_data_frames()[0]

    master_data = pd.merge(logs_raw, roster, on="PLAYER_ID", how="left")
    master_data["GAME_DATE"] = pd.to_datetime(master_data["GAME_DATE"])

    print(f"Saving to {OUT_FILE}...")
    master_data.to_parquet(OUT_FILE, engine="pyarrow", compression="zstd", index=False)
    print("Done!")


if __name__ == "__main__":
    main()
