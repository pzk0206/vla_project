"""不调用 VLM 的动态 smoke 案例筛选。"""


def classify_candidate(step_rows, settings, error=None):
    """按 clear、位置误差和最终距离对一条候选轨迹分类。"""
    rows = list(step_rows)
    if error is not None:
        return {
            "qualified": False,
            "reason": "candidate_error",
            "error": error,
        }

    for row in rows:
        if (
            row["block_visibility_ratio"]
            < settings["clear_visibility_threshold"]
        ):
            return {
                "qualified": False,
                "reason": "visibility_below_threshold",
            }
        if row["target_error_3d"] > settings["max_pose_error"]:
            reason = (
                "start_pose_error"
                if row["observation_step"] == 0
                else "motion_target_error"
            )
            return {"qualified": False, "reason": reason}

    expected_observations = settings["num_actions"] + 1
    if len(rows) != expected_observations:
        return {"qualified": False, "reason": "incomplete_trace"}

    if rows[-1]["true_distance_xy"] > settings["max_final_distance_xy"]:
        return {"qualified": False, "reason": "final_distance_error"}
    return {"qualified": True, "reason": "qualified"}


def select_qualified_cases(candidate_rows, directions):
    """按方向顺序选择 seed 最小且互不重复的合格案例。"""
    rows = list(candidate_rows)
    selected = []
    used_seeds = set()
    for direction in directions:
        eligible = sorted(
            (
                row
                for row in rows
                if row["qualified"]
                and row["direction"] == direction
                and row["seed"] not in used_seeds
            ),
            key=lambda row: row["seed"],
        )
        if not eligible:
            raise ValueError(f"没有可选的 {direction} 合格案例")
        selected.append(eligible[0])
        used_seeds.add(eligible[0]["seed"])
    return selected
