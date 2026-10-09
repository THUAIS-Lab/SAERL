"""Extend the upstream scoring table through its custom reward interface."""


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source in {"zwhe99/DeepMath-103K", "math-ai/minervamath", "knoveleng/OlympiadBench"}:
        from saerl.rewards.deepmath import compute_score as score_deepmath

        return score_deepmath(solution_str, ground_truth)
    if data_source in {"Maxwell-Jia/AIME_2024", "knoveleng/AMC-23"}:
        from verl.utils.reward_score.math_reward import compute_score as score_math

        return score_math(solution_str, ground_truth)
    from verl.utils.reward_score import default_compute_score

    return default_compute_score(
        data_source=data_source, solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info, **kwargs
    )
