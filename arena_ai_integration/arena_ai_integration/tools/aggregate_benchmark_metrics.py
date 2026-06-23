#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import pandas as pd

STRICT_GOAL_RADIUS_M = 0.5
TIMEOUT_SEC = 180.0
TIMEOUT_MARGIN_SEC = 1.0
LOW_COLLISION_THRESHOLD = 2
ARENA_MAX_COLLISIONS = 3
SOCIAL_INTRUSION_RATIO_THRESHOLD = 5.0
SOCIAL_INTRUSION_TIME_THRESHOLD = 5.0

BOOLEAN_METRIC_COLUMNS = {
    "completed",
    "completed_hunav",
    "completed_goal_strict",
    "low_collision",
    "timeout_like",
    "success",
    "social_success",
}

RATE_COLUMNS = {
    "completed_hunav": "completed_hunav_rate_pct",
    "completed_goal_strict": "goal_completion_rate_pct",
    "low_collision": "low_collision_rate_pct",
    "timeout_like": "timeout_rate_pct",
    "success": "success_rate_pct",
    "social_success": "social_success_rate_pct",
}


def parse_experiment_tag(tag: str) -> tuple[str, str, int]:
    if "__" in tag and "__ep" in tag:
        contestant, stage, episode = tag.rsplit("__", 2)
        if episode.startswith("ep"):
            return contestant, stage, int(episode[2:])

    match = re.match(r"^(?P<prefix>.+)_ep(?P<episode>\d+)$", tag)
    if not match:
        return "unknown", tag, -1

    prefix = match.group("prefix")
    episode = int(match.group("episode"))
    if "_" in prefix:
        contestant, stage = prefix.split("_", 1)
        return contestant, stage, episode
    return prefix, "unknown", episode


def to_bool_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def numeric_column(data: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in data.columns:
        return pd.Series(default, index=data.index, dtype=float)
    return pd.to_numeric(data[column], errors="coerce")


def derive_episode_metrics(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    if "recorded_episode_duration" not in data.columns and "time_to_reach_goal" in data.columns:
        # Legacy CSVs used time_to_reach_goal for the whole recording duration.
        data["recorded_episode_duration"] = numeric_column(data, "time_to_reach_goal")

    if "completed" in data.columns:
        data["completed"] = to_bool_series(data["completed"])

    if "completed_hunav" in data.columns:
        data["completed_hunav"] = to_bool_series(data["completed_hunav"])
    elif "completed" in data.columns:
        data["completed_hunav"] = data["completed"]

    if "completed_goal_strict" in data.columns:
        data["completed_goal_strict"] = to_bool_series(data["completed_goal_strict"])
    elif "final_distance_to_target" in data.columns:
        data["completed_goal_strict"] = (
            numeric_column(data, "final_distance_to_target") <= STRICT_GOAL_RADIUS_M
        ).fillna(False)
    elif "completed_hunav" in data.columns:
        data["completed_goal_strict"] = data["completed_hunav"]
    else:
        data["completed_goal_strict"] = False

    robot_collisions = numeric_column(data, "robot_on_person_collision")
    person_collisions = numeric_column(data, "person_on_robot_collision")
    if "total_collisions" in data.columns:
        data["total_collisions"] = numeric_column(data, "total_collisions").fillna(
            robot_collisions.add(person_collisions, fill_value=0.0)
        )
    else:
        data["total_collisions"] = robot_collisions.add(person_collisions, fill_value=0.0)

    data["low_collision"] = data["total_collisions"] < LOW_COLLISION_THRESHOLD

    duration = numeric_column(data, "recorded_episode_duration", default=float("nan"))
    timeout_by_duration = duration >= (TIMEOUT_SEC - TIMEOUT_MARGIN_SEC)

    if "result_arena_like" not in data.columns:
        result = pd.Series("TIMEOUT", index=data.index, dtype=object)
        collision = data["total_collisions"] >= ARENA_MAX_COLLISIONS
        goal = data["completed_goal_strict"]
        result.loc[collision & ~timeout_by_duration] = "COLLISION"
        result.loc[goal & ~timeout_by_duration & ~collision] = "GOAL_REACHED"
        data["result_arena_like"] = result
    else:
        data["result_arena_like"] = data["result_arena_like"].astype(str).str.upper()

    data["timeout_like"] = (
        data["result_arena_like"].astype(str).str.upper().eq("TIMEOUT")
        | timeout_by_duration.fillna(False)
    )

    data["success"] = (
        data["completed_goal_strict"]
        & data["low_collision"]
        & ~data["timeout_like"]
    )

    social_safe = pd.Series(True, index=data.index, dtype=bool)
    if "intrusion_ratio_08m" in data.columns:
        social_safe &= (
            numeric_column(data, "intrusion_ratio_08m")
            <= SOCIAL_INTRUSION_RATIO_THRESHOLD
        ).fillna(False)
    if "intrusion_time_08m" in data.columns:
        social_safe &= (
            numeric_column(data, "intrusion_time_08m")
            <= SOCIAL_INTRUSION_TIME_THRESHOLD
        ).fillna(False)

    data["social_success"] = data["success"] & social_safe

    for column in BOOLEAN_METRIC_COLUMNS:
        if column in data.columns:
            data[column] = to_bool_series(data[column])

    return data


def aggregate_metrics(data: pd.DataFrame) -> pd.DataFrame:
    parsed = data["experiment_tag"].astype(str).map(parse_experiment_tag)
    data = data.copy()
    data["contestant"] = parsed.map(lambda item: item[0])
    data["stage"] = parsed.map(lambda item: item[1])
    data["episode"] = parsed.map(lambda item: item[2])

    data = derive_episode_metrics(data)

    excluded = {
        "experiment_tag",
        "run_id",
        "contestant",
        "stage",
        "episode",
        "result_arena_like",
    }
    metric_columns = [column for column in data.columns if column not in excluded]

    rows = []
    for (contestant, stage), group in data.groupby(["contestant", "stage"], sort=False):
        row = {
            "contestant": contestant,
            "stage": stage,
            "episodes": int(len(group)),
        }
        for column in metric_columns:
            if column in BOOLEAN_METRIC_COLUMNS:
                row[column] = float(group[column].mean())
                continue

            numeric = pd.to_numeric(group[column], errors="coerce")
            if numeric.notna().any():
                row[column] = float(numeric.mean())

        for column, rate_name in RATE_COLUMNS.items():
            if column in group.columns:
                row[rate_name] = float(group[column].mean() * 100.0)

        if "result_arena_like" in group.columns:
            result = group["result_arena_like"].astype(str).str.upper()
            row["result_goal_reached_rate_pct"] = float(
                result.eq("GOAL_REACHED").mean() * 100.0
            )
            row["result_collision_rate_pct"] = float(
                result.eq("COLLISION").mean() * 100.0
            )
            row["result_timeout_rate_pct"] = float(
                result.eq("TIMEOUT").mean() * 100.0
            )
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {input_path}")

    data = pd.read_csv(input_path)
    if "experiment_tag" not in data.columns:
        raise KeyError("Expected column 'experiment_tag' in metrics file")
    if args.run_id is not None and "run_id" in data.columns:
        run_id = pd.to_numeric(pd.Series([args.run_id]), errors="coerce").iloc[0]
        data = data[pd.to_numeric(data["run_id"], errors="coerce") == run_id]
    if data.empty:
        raise RuntimeError("No metrics rows matched the requested run")

    aggregated = aggregate_metrics(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(output_path, index=False)
    print(f"Wrote aggregated benchmark metrics to {output_path}")


if __name__ == "__main__":
    main()
