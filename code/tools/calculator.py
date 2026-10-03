"""Bounded arithmetic with decimal precision, without executing Python code."""

import ast
import operator
from decimal import Decimal, DecimalException, ROUND_FLOOR, ROUND_HALF_EVEN, localcontext


class Calculator:
    OPERATIONS = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow,
    }

    @classmethod
    def evaluate(cls, expression):
        if not isinstance(expression, str) or not expression.strip() or len(expression) > 500:
            raise ValueError("Provide an arithmetic expression of 1 to 500 characters.")
        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except (SyntaxError, ValueError) as exc:
            raise ValueError("Invalid arithmetic expression.") from exc
        if len(list(ast.walk(tree))) > 100:
            raise ValueError("Expression is too complex.")
        expression = expression.strip()

        def evaluate(node, depth=0):
            if depth > 20:
                raise ValueError("Expression is too deeply nested.")
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                literal = ast.get_source_segment(expression, node).replace("_", "")
                value = Decimal(literal)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = evaluate(node.operand, depth + 1)
                if isinstance(node.op, ast.USub):
                    value = -value
            elif isinstance(node, ast.BinOp) and type(node.op) in cls.OPERATIONS:
                left = evaluate(node.left, depth + 1)
                right = evaluate(node.right, depth + 1)
                if isinstance(node.op, ast.Pow) and abs(right) > 100:
                    raise ValueError("Exponent magnitude must not exceed 100.")
                if isinstance(node.op, (ast.FloorDiv, ast.Mod)):
                    quotient = (left / right).to_integral_value(rounding=ROUND_FLOOR)
                    value = quotient if isinstance(node.op, ast.FloorDiv) else left - quotient * right
                else:
                    value = cls.OPERATIONS[type(node.op)](left, right)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id == "round" and not node.keywords and 1 <= len(node.args) <= 2):
                number = evaluate(node.args[0], depth + 1)
                digits = evaluate(node.args[1], depth + 1) if len(node.args) == 2 else Decimal(0)
                if digits != digits.to_integral_value() or not -20 <= digits <= 20:
                    raise ValueError("round digits must be an integer between -20 and 20.")
                value = round(number, int(digits))
            else:
                raise ValueError("Only numbers, +, -, *, /, //, %, **, parentheses, and round are supported.")
            if not value.is_finite() or (value and not -100 <= value.adjusted() <= 100):
                raise ValueError("Number magnitude is outside the supported range (1e-100 to 1e100).")
            return value

        try:
            with localcontext() as context:
                context.prec = 50
                context.rounding = ROUND_HALF_EVEN
                context.Emax = 100
                context.Emin = -100
                result = evaluate(tree.body)
                text = format(result, "f")
        except (DecimalException, OverflowError, ZeroDivisionError) as exc:
            raise ValueError(f"Calculation failed: {exc}") from exc
        return {"expression": expression, "result": text,
                "precision": 50, "rounding": "ROUND_HALF_EVEN",
                "note": "Result is a decimal string; units and source values must be verified separately."}
