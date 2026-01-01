import re

PH = re.compile(r"\{\s*([A-Za-z0-9_]+)\s*\}")

def render_numbers(s: str, vals: dict) -> str:
    # {num_digit_1} -> (123)
    return PH.sub(lambda m: f"({vals[m.group(1)]})", s)

def normalize_equality(s: str) -> str:
    # turn a lone '=' into '==', but leave '<=', '>=', '!=', '==' intact
    return re.sub(r'(?<![<>=!])=(?!=)', '==', s)

def restrictions_ok(restrictions, vals) -> bool:
    for r in restrictions or []:
        expr = normalize_equality(render_numbers(r, vals))
        if not eval(expr, {"__builtins__": {}}, {}):  # boolean result
            # print(f"failed: {expr}")  # uncomment to debug
            return False
    return True

def eval_equation(equation: str, vals: dict) -> float:
    expr = render_numbers(equation, vals)
    return eval(expr, {"__builtins__": {}}, {})

# --- paste/edit these ---------------------------------------------------------
equation = " ({num_digit_1} + (1/2)*{num_digit_1}) * 3 * {num_digit_3} "
combo = {
     "num_digit_1": 6,  "num_digit_2": 7, "num_digit_3": 5,  "answer": 135
    # include values for any vars you use in restrictions
}
restriction_list = [
    # examples:
    "{num_digit_1} <= 16",
    "{num_digit_2} <= 7"
     # "{num_digit_1} > 100",
    # "{num_digit_4} < 500",
    # "{num_digit_3} * 2 = {num_digit_2} + {num_digit_1} - 99"
]
# -----------------------------------------------------------------------------

if restrictions_ok(restriction_list, combo):
    val = eval_equation(equation, combo)
    print("value:", val)
    if "answer" in combo:
        print("check:", "OK" if abs(val - combo["answer"]) < 1e-9 else "WRONG")
else:
    print("restriction(s) not satisfied")
