import re

# --- paste/edit these ---------------------------------------------------------
equation = "(({num_digit_1} * {num_digit_2} + 2 * {num_digit_3}) - ({num_digit_2} + {num_digit_3})) * {num_digit_5} / {num_digit_4}"

#combo = {
#  "num_digit_1": 180,
#  "num_digit_2": 45,
#  "num_digit_3": 21,
#  "num_digit_4": 298,
#  "answer": 304
#}

combo = {
    "num_digit_1": 2,
    "num_digit_2": 15,
    "num_digit_3": 10,
    "num_digit_4": 1,
    "num_digit_5": 3,
    "answer": 75
}




# -----------------------------------------------------------------------------

PH = re.compile(r"\{\s*([A-Za-z0-9_]+)\s*\}")

def render_and_eval(eq: str, values: dict) -> float:
    expr = PH.sub(lambda m: f"({values[m.group(1)]})", eq)
    val = eval(expr)
    print("expr:", expr)
    print("val :", val)
    if "answer" in values:
        ok = abs(val - values["answer"]) < 1e-9
        print("check against 'answer':", values["answer"], "=>", "OK" if ok else "WRONG")
    return val

render_and_eval(equation, combo)
