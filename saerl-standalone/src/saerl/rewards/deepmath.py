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

import mathruler.grader as mathruler_grader
from verl.utils.py_functional import timeout_limit

SYMPY_TIMEOUT_SECONDS = 10

# mathruler falls back to sympy.simplify for answers that do not match after
# normalization. Run only that potentially unbounded fallback in a child
# process so a pathological expression cannot block a reward worker forever.
mathruler_grader.are_equal_under_sympy = timeout_limit(seconds=SYMPY_TIMEOUT_SECONDS)(
    mathruler_grader.are_equal_under_sympy
)


def compute_score(solution_str, ground_truth):
    solution_str = solution_str.strip()
    extract_output = mathruler_grader.extract_boxed_content(solution_str)

    try:
        return float(mathruler_grader.grade_answer(extract_output, ground_truth))
    except TimeoutError:
        return 0.0
