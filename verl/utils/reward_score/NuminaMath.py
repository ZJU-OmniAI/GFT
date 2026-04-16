# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import re
import queue # 需要导入 queue 模块来捕获 Empty 异常
import regex
import multiprocessing
from math import isclose
from typing import Union
from collections import defaultdict

from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy2 import latex2sympy

import random
import regex
import re
import sympy
from latex2sympy2 import latex2sympy
from typing import TypeVar, Iterable, List, Union, Any, Dict
from word2number import w2n

# from .parser import choice_answer_clean, strip_string
# from parser import choice_answer_clean


def choice_answer_clean(pred: str):
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")
    return pred


def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None


def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None


def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    # print("Judge:", prediction, reference)
    if prediction is None or reference is None:
        return False
    if str(prediction.strip().lower()) == str(reference.strip().lower()):
        return True
    if (
        reference in ["A", "B", "C", "D", "E"]
        and choice_answer_clean(prediction) == reference
    ):
        return True

    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## pmatrix (amps)
    if "pmatrix" in prediction and not "pmatrix" in reference:
        reference = str_to_pmatrix(reference)

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[")
        and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        prediction.startswith("(")
        and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(
                        pred_parts[i], ref_parts[i], include_percentage, is_close
                    )
                    for i in range(len(pred_parts))
                ]
            ):
                return True
    if (
        (
            prediction.startswith("\\begin{pmatrix}")
            or prediction.startswith("\\begin{bmatrix}")
        )
        and (
            prediction.endswith("\\end{pmatrix}")
            or prediction.endswith("\\end{bmatrix}")
        )
        and (
            reference.startswith("\\begin{pmatrix}")
            or reference.startswith("\\begin{bmatrix}")
        )
        and (
            reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")
        )
    ):
        pred_lines = [
            line.strip()
            for line in prediction[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        [
                            math_equal(
                                pred_parts[i],
                                ref_parts[i],
                                include_percentage,
                                is_close,
                            )
                            for i in range(len(pred_parts))
                        ]
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        if math_equal(
            prediction.split("=")[1], reference, include_percentage, is_close
        ):
            return True
    elif (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        if math_equal(
            prediction, reference.split("=")[1], include_percentage, is_close
        ):
            return True

    # symbolic equal with sympy
    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def math_equal_process(param):
    return math_equal(param[-2], param[-1])


def numeric_equal(prediction: float, reference: float):
    # Note that relative tolerance has significant impact
    # on the result of the synthesized GSM-Hard dataset
    # if reference.is_integer():
    #     return isclose(reference, round(prediction), abs_tol=1e-4)
    # else:
    # prediction = round(prediction, len(str(reference).split(".")[-1]))
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s

    a_expr = _parse(a)
    b_expr = _parse(b)

    # --- 修复开始：转换失败直接返回 False ---
    if isinstance(a_expr, str) or isinstance(b_expr, str):
        return str(a_expr).strip().lower() == str(b_expr).strip().lower()
    # --- 修复结束 ---

    # 1. Direct string comparison (fastest)
    try:
        if str(a_expr) == str(b_expr) or a_expr == b_expr:
            return True
    except:
        pass

    # --- 修复开始：增加复杂度检测，防止 OOM ---
    # 检查是否包含阶乘(Factorial)，如果包含，视为高风险计算，跳过 N() 和 simplify
    # 或者检查原子数量，过大则跳过
    is_complex = False
    try:
        # 检查是否包含阶乘函数
        if a_expr.has(sympy.factorial) or b_expr.has(sympy.factorial):
            is_complex = True
        # 检查是否包含过大的幂运算 (Heuristic)
        if a_expr.has(sympy.Pow) or b_expr.has(sympy.Pow):
             # 简单判断：如果又有阶乘又有幂，绝对不碰
             if is_complex: 
                 # 仅做结构性判断，不做计算
                 return False 
    except:
        pass
    # --- 修复结束 ---

    if not is_complex:
        # simplify equal
        try:
            # 增加一个简单的超时保护或捕捉 RecursionError
            if simplify(a_expr - b_expr) == 0:
                return True
        except:
            pass

        # equation equal
        try:
            if hasattr(a_expr, 'lhs') and hasattr(b_expr, 'lhs'):
                if (abs(a_expr.lhs - a_expr.rhs)).equals(abs(b_expr.lhs - b_expr.rhs)):
                    return True
        except:
            pass

        # numeric equal (只有在不复杂的情况下才执行 N)
        try:
            # 使用 evalf 并指定精度，防止过度计算，但对于超大数 float() 依然会溢出
            # 这里即使溢出也会抛出异常被 except 捕获，而不是卡死
            num_a = N(a_expr)
            num_b = N(b_expr)
            if numeric_equal(float(num_a), float(num_b)):
                return True
        except:
            pass

    # matrix (保持原样)
    try:
        if hasattr(a_expr, 'shape') and hasattr(b_expr, 'shape'):
            if a_expr.shape == b_expr.shape:
                _a = a_expr.applyfunc(lambda x: round(x, 3))
                _b = b_expr.applyfunc(lambda x: round(x, 3))
                if _a.equals(_b):
                    return True
    except:
        pass

    return False


def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)




def call_with_timeout(func, *args, timeout=2, **kwargs):
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.daemon = True  # 设置为守护进程，父进程挂了子进程也会挂
    process.start()
    
    process.join(timeout)

    if process.is_alive():
        # 超时处理
        try:
            process.terminate()
            # 给一点时间让它结束，如果还不行就 kill
            process.join(0.1)
            if process.is_alive():
                # hasattr check for Python version compatibility (kill exists in 3.7+)
                if hasattr(process, 'kill'): 
                    process.kill()
                process.join()
        except Exception as e:
            print(f"Error terminating process: {e}")
        return False

    try:
        # 即使 process 结束了，如果没放进去东西（比如 crash 了），get 也会阻塞
        # 所以这里也需要 timeout
        return output_queue.get(timeout=0.1)
    except queue.Empty:
        return False
    except Exception as e:
        print(f"Error in queue get: {e}")
        return False




def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def convert_word_number(text: str) -> str:
    try:
        text = str(w2n.word_to_num(text))
    except:
        pass
    return text


# units mainly from MathQA
unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]

unit_texts.extend([t + "s" for t in unit_texts])


def strip_string(string, skip_unit=False):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    # replace \\ with \
    string = string.replace("\\!", "")
    # string = string.replace("\\ ", "")
    # string = string.replace("\\\\", "\\")

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    if not skip_unit:
        # Remove unit: texts
        for _ in range(2):
            for unit_text in unit_texts:
                # use regex, the prefix should be either the start of the string or a non-alphanumeric character
                # the suffix should be either the end of the string or a non-alphanumeric character
                _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
                if _string != "":
                    string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")

    # convert word number to digit
    string = convert_word_number(string)

    # replace "\\text{...}" to "..."
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    # string = string.replace("\\cdot", "")
    if (
        string.startswith("{")
        and string.endswith("}")
        and string.isalnum()
        or string.startswith("(")
        and string.endswith(")")
        and string.isalnum()
        or string.startswith("[")
        and string.endswith("]")
        and string.isalnum()
    ):
        string = string[1:-1]

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def extract_multi_choice_answer(pred_str):
    # TODO: SFT models
    if "Problem:" in pred_str:
        pred_str = pred_str.split("Problem:", 1)[0]
    pred_str = pred_str.replace("choice is", "answer is")
    patt = regex.search(r"answer is \(?(?P<ans>[abcde])\)?", pred_str.lower())
    if patt is not None:
        return patt.group("ans").upper()
    return "placeholder"


direct_answer_trigger_for_fewshot = ("choice is", "answer is")


def choice_answer_clean(pred: str):
    pred = pred.strip("\n")

    # Determine if this is ICL, if so, use \n\n to split the first chunk.
    ICL = False
    for trigger in direct_answer_trigger_for_fewshot:
        if pred.count(trigger) > 1:
            ICL = True
    if ICL:
        pred = pred.split("\n\n")[0]

    # Split the trigger to find the answer.
    preds = re.split("|".join(direct_answer_trigger_for_fewshot), pred)
    if len(preds) > 1:
        answer_flag = True
        pred = preds[-1]
    else:
        answer_flag = False

    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")

    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]

    if len(pred) == 0:
        pred = ""
    else:
        if answer_flag:
            # choose the first element in list ...
            pred = pred[0]
        else:
            # choose the last e
            pred = pred[-1]

    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")

    return pred


def find_box(pred_str: str):
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def clean_units(pred_str: str):
    """Clean the units in the number."""

    def convert_pi_to_number(code_string):
        code_string = code_string.replace("\\pi", "π")
        # Replace \pi or π not preceded by a digit or } with 3.14
        code_string = re.sub(r"(?<![\d}])\\?π", "3.14", code_string)
        # Replace instances where π is preceded by a digit but without a multiplication symbol, e.g., "3π" -> "3*3.14"
        code_string = re.sub(r"(\d)(\\?π)", r"\1*3.14", code_string)
        # Handle cases where π is within braces or followed by a multiplication symbol
        # This replaces "{π}" with "3.14" directly and "3*π" with "3*3.14"
        code_string = re.sub(r"\{(\\?π)\}", "3.14", code_string)
        code_string = re.sub(r"\*(\\?π)", "*3.14", code_string)
        return code_string

    pred_str = convert_pi_to_number(pred_str)
    pred_str = pred_str.replace("%", "/100")
    pred_str = pred_str.replace("$", "")
    pred_str = pred_str.replace("¥", "")
    pred_str = pred_str.replace("°C", "")
    pred_str = pred_str.replace(" C", "")
    pred_str = pred_str.replace("°", "")
    return pred_str


def extract_theoremqa_answer(pred: str, answer_flag: bool = True):
    if any([option in pred.lower() for option in ["yes", "true"]]):
        pred = "True"
    elif any([option in pred.lower() for option in ["no", "false"]]):
        pred = "False"
    elif any(
        [
            option in pred.lower()
            for option in ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
        ]
    ):
        pass
    else:
        # Some of the models somehow get used to boxed output from pre-training
        if "boxed" in pred:
            pred = find_box(pred)

        if answer_flag:
            # Extract the numbers out of the string
            pred = pred.split("=")[-1].strip()
            pred = clean_units(pred)
            try:
                tmp = str(latex2sympy(pred))
                pred = str(eval(tmp))
            except Exception:
                if re.match(r"-?[\d\.]+\s\D+$", pred):
                    pred = pred.split(" ")[0]
                elif re.match(r"-?[\d\.]+\s[^\s]+$", pred):
                    pred = pred.split(" ")[0]
        else:
            # desparate search over the last number
            preds = re.findall(r"-?\d*\.?\d+", pred)
            if len(preds) >= 1:
                pred = preds[-1]
            else:
                pred = ""

    return pred


def extract_answer(pred_str, data_name, use_last_number=True):
    pred_str = pred_str.replace("\u043a\u0438", "")
    if data_name in ["mmlu_stem", "sat_math", "aqua", "gaokao2023"]:
        # TODO check multiple choice
        return choice_answer_clean(pred_str)

    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        # minerva_math
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1] # 提取boxed后面的答案部分
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    elif "答案是" in pred_str:
        # Handle Chinese few-shot multiple choice problem answer extraction
        pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    else:  # use the last number
        if use_last_number:
            pattern = "-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ""
        else:
            pred = ""

    # choice answer
    if (
        data_name in ["sat_math", "aqua"]
        or "mmlu" in data_name
    ):
        tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
        if tmp:
            pred = tmp[-1]
        else:
            pred = pred.strip().strip(".")

    # multiple line
    # pred = pred.split("\n")[0]
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])
    return pred


STRIP_EXCEPTIONS = ["carp_en", "minerva_math"]


def parse_ground_truth(example: Dict[str, Any], data_name):
    if "gt_cot" in example and "gt" in example:
        if data_name in ["math", "math_oai"]:
            gt_ans = extract_answer(example["gt_cot"], data_name)
        elif data_name in STRIP_EXCEPTIONS:
            gt_ans = example["gt"]
        else:
            gt_ans = strip_string(example["gt"])
        return example["gt_cot"], gt_ans

    # parse ground truth
    if data_name in ["math", "minerva_math", "math_oai"]:
        gt_cot = example["solution"]
        gt_ans = extract_answer(gt_cot, data_name)
    elif data_name == "gsm8k":
        gt_cot, gt_ans = example["answer"].split("####")
    elif data_name == "svamp":
        gt_cot, gt_ans = example["Equation"], example["Answer"]
    elif data_name == "asdiv":
        gt_cot = example["formula"]
        gt_ans = re.sub(r"\(.*?\)", "", example["answer"])
    elif data_name == "mawps":
        gt_cot, gt_ans = None, example["target"]
    elif data_name == "tabmwp":
        gt_cot = example["solution"]
        gt_ans = example["answer"]
        if example["ans_type"] in ["integer_number", "decimal_number"]:
            if "/" in gt_ans:
                gt_ans = int(gt_ans.split("/")[0]) / int(gt_ans.split("/")[1])
            elif "," in gt_ans:
                gt_ans = float(gt_ans.replace(",", ""))
            elif "%" in gt_ans:
                gt_ans = float(gt_ans.split("%")[0]) / 100
            else:
                gt_ans = float(gt_ans)
    elif data_name == "carp_en":
        gt_cot, gt_ans = example["steps"], example["answer"]
    elif data_name == "mmlu_stem":
        abcd = "ABCD"
        gt_cot, gt_ans = None, abcd[example["answer"]]
    elif data_name == "sat_math":
        gt_cot, gt_ans = None, example["Answer"]
    elif data_name == "aqua":
        gt_cot, gt_ans = None, example["correct"]
    elif data_name in ["gaokao2023en", "college_math", "gaokao_math_cloze"]:
        gt_cot, gt_ans = None, example["answer"].replace("$", "").strip()
    elif data_name == "gaokao_math_qa":
        gt_cot, gt_ans = None, example["label"]
    elif data_name in ["gaokao2024_mix", "cn_middle_school"]:
        if len(example["choice_answer"]) > 0:
            gt_cot, gt_ans = None, example["choice_answer"]
        else:
            gt_cot, gt_ans = None, example["answer"]
    elif data_name == "olympiadbench":
        gt_cot, gt_ans = None, example["final_answer"][0].strip("$")
    elif data_name in [
        "aime24",
        "aime25",
        "amc23",
        "cmath",
        "gaokao2024_I",
        "gaokao2024_II",
        "imo2024",
    ]:
        gt_cot, gt_ans = None, example["answer"]
    else:
        raise NotImplementedError(f"`{data_name}`")
    # post process
    gt_cot = str(gt_cot).strip()
    if data_name not in STRIP_EXCEPTIONS:
        gt_ans = strip_string(gt_ans, skip_unit=data_name == "carp_en")
    else:
        gt_ans = (
            gt_ans.replace("\\neq", "\\ne")
            .replace("\\leq", "\\le")
            .replace("\\geq", "\\ge")
        )
    return gt_cot, gt_ans


def parse_question(example, data_name):
    question = ""
    if data_name == "asdiv":
        question = f"{example['body'].strip()} {example['question'].strip()}"
    elif data_name == "svamp":
        body = example["Body"].strip()
        if not body.endswith("."):
            body = body + "."
        question = f'{body} {example["Question"].strip()}'
    elif data_name == "tabmwp":
        title_str = (
            f'regarding "{example["table_title"]}" ' if example["table_title"] else ""
        )
        question = f"Read the following table {title_str}and answer a question:\n"
        question += f'{example["table"]}\n{example["question"]}'
        if example["choices"]:
            question += (
                f' Please select from the following options: {example["choices"]}'
            )
    elif data_name == "carp_en":
        question = example["content"]
    elif data_name == "mmlu_stem":
        options = example["choices"]
        assert len(options) == 4
        for i, (label, option) in enumerate(zip("ABCD", options)):
            options[i] = f"({label}) {str(option).strip()}"
        options = " ".join(options)
        # question = f"{example['question'].strip()}\nWhat of the following is the right choice? Explain your answer.\n{options}"
        question = f"{example['question'].strip()}\nAnswer Choices: {options}"
    elif data_name == "sat_math":
        options = example["options"].strip()
        assert "A" == options[0]
        options = "(" + options
        for ch in "BCD":
            if f" {ch}) " in options:
                options = regex.sub(f" {ch}\) ", f" ({ch}) ", options)
        # question = f"{example['question'].strip()}\nWhat of the following is the right choice? Explain your answer.\n{options.strip()}"
        question = f"{example['question'].strip()}\nAnswer Choices: {options}"
    elif "aqua" in data_name:
        options = example["options"]
        choice = "(" + "(".join(options)
        choice = choice.replace("(", " (").replace(")", ") ").strip()
        choice = "\nAnswer Choices: " + choice
        question = example["question"].strip() + choice
    elif data_name == "gaokao_math_qa":
        options_dict = example["options"]
        options = []
        for key in options_dict:
            options.append(f"({key}) {options_dict[key]}")
        options = " ".join(options)
        question = f"{example['question'].strip()}\n选项: {options}"
    else:
        for key in ["question", "problem", "Question", "input"]:
            if key in example:
                question = example[key]
                break
    # assert question != ""
    # Yes or No question
    _, gt_ans = parse_ground_truth(example, data_name)
    if isinstance(gt_ans, str):
        gt_lower = gt_ans.lower()
        if gt_lower in ["true", "false"]:
            question += " (True or False)"
        if gt_lower in ["yes", "no"]:
            question += " (Yes or No)"
    return question.strip()



def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    model_answer = extract_answer(solution_str, data_name="none")
    
    if math_equal(model_answer, ground_truth, timeout=True):
        return 1.0
    else:
        return 0.0
