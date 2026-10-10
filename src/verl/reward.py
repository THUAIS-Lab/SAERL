# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset-specific math rewards loaded through the official verl custom reward interface."""

import mathruler.grader as mathruler_grader
from verl.utils.py_functional import timeout_limit

SYMPY_TIMEOUT_SECONDS = 10

# mathruler falls back to sympy.simplify for answers that do not match after
# normalization. Run only that potentially unbounded fallback in a child
# process so a pathological expression cannot block a reward worker forever.
mathruler_grader.are_equal_under_sympy = timeout_limit(seconds=SYMPY_TIMEOUT_SECONDS)(
    mathruler_grader.are_equal_under_sympy
)


def _score_deepmath(solution_str, ground_truth):
    solution_str = solution_str.strip()
    extract_output = mathruler_grader.extract_boxed_content(solution_str)

    try:
        return float(mathruler_grader.grade_answer(extract_output, ground_truth))
    except TimeoutError:
        return 0.0


# Keep the original training/validation grading rules in one reward module.
MATHRULER_SOURCES = {"zwhe99/DeepMath-103K", "math-ai/minervamath", "knoveleng/OlympiadBench"}
VERL_MATH_SOURCES = {"Maxwell-Jia/AIME_2024", "knoveleng/AMC-23"}


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Official verl custom-reward signature; preserve dataset-specific grading."""
    if data_source in MATHRULER_SOURCES:
        return _score_deepmath(solution_str, ground_truth)
    if data_source in VERL_MATH_SOURCES:
        from verl.utils.reward_score.math_reward import compute_score as grade_math

        return grade_math(solution_str, ground_truth)
    from verl.utils.reward_score import default_compute_score

    return default_compute_score(
        data_source=data_source, solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info, **kwargs
    )
