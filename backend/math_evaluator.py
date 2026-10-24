"""
GuessMyText — V4 Mathematical Expression Evaluator

Safe, robust arithmetic evaluator for handwritten single-line math expressions.
Strictly decoupled from image recognition.

Grammar & Rules:
  - Allowed characters: digits 0-9, decimal point '.', operators: '+', '-', '*', '/', '÷', '%'.
  - Note: '=' is NOT in the dataset and must NOT be written on canvas (click Solve to compute).
  - Multiplicative operators ('*', '÷', '/', '%') share precedence and evaluate left-to-right.
  - Additive operators ('+', '-') share lower precedence and evaluate left-to-right.
  - '%' represents modulo / remainder (e.g. 17 % 5 = 2).
  - Division produces exact decimals when needed (e.g. 7 ÷ 2 = 3.5).
  - Unary plus/minus: Supported at the start of expression or following an operator (e.g. -3+5 = 2, 5*-3 = -15).
  - Safety: Pure tokenizer and recursive-descent/precedence parser. Zero use of Python eval() or exec().
  - Limits: Expression length <= 100, token count <= 50, numeric magnitude <= 1e12.
"""

from dataclasses import dataclass
from enum import Enum, auto
import math
import re
from typing import List, Optional, Tuple, Union


class TokenType(Enum):
    NUMBER = auto()
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: Union[float, int, str]
    pos: int


@dataclass
class EvaluationResult:
    status: str  # 'success', 'error', 'division_by_zero', 'unsupported_character', 'invalid_expression'
    expression: str
    result: Optional[Union[int, float]] = None
    formatted_result: Optional[str] = None
    full_equation: Optional[str] = None
    error_message: Optional[str] = None


MAX_EXPR_LENGTH = 100
MAX_TOKEN_COUNT = 50
MAX_MAGNITUDE = 1e12


class MathEvaluationError(Exception):
    def __init__(self, message: str, error_type: str = "invalid_expression"):
        super().__init__(message)
        self.error_type = error_type


def tokenize(expression_str: str) -> List[Token]:
    """
    Tokenizes raw recognized expression string into typed tokens.
    Rejects unsupported characters (e.g. letters, parentheses, unknown symbols).
    """
    if len(expression_str) > MAX_EXPR_LENGTH:
        raise MathEvaluationError(
            f"Expression exceeds maximum allowed length of {MAX_EXPR_LENGTH} characters.",
            "invalid_expression"
        )

    tokens: List[Token] = []
    i = 0
    s_eval = expression_str.strip()

    if not s_eval:
        raise MathEvaluationError("Expression is empty.", "invalid_expression")

    while i < len(s_eval):
        ch = s_eval[i]

        # Skip whitespace
        if ch.isspace():
            i += 1
            continue

        # Number: digits with optional decimal point
        if ch.isdigit() or ch == '.':
            start = i
            has_dot = (ch == '.')
            i += 1
            while i < len(s_eval) and (s_eval[i].isdigit() or s_eval[i] == '.'):
                if s_eval[i] == '.':
                    if has_dot:
                        raise MathEvaluationError(
                            f"Invalid number with multiple decimal points at position {i}.",
                            "invalid_expression"
                        )
                    has_dot = True
                i += 1
            num_str = s_eval[start:i]
            if num_str == '.':
                raise MathEvaluationError(
                    f"Lone decimal point at position {start}.",
                    "invalid_expression"
                )
            num_val = float(num_str) if has_dot else int(num_str)
            if abs(num_val) > MAX_MAGNITUDE:
                raise MathEvaluationError(
                    f"Number {num_str} exceeds maximum allowed magnitude ({MAX_MAGNITUDE:.0e}).",
                    "invalid_expression"
                )
            tokens.append(Token(TokenType.NUMBER, num_val, start))
            continue

        # Operators
        if ch == '+':
            tokens.append(Token(TokenType.PLUS, '+', i))
            i += 1
        elif ch == '-':
            tokens.append(Token(TokenType.MINUS, '-', i))
            i += 1
        elif ch in ('*', '×'):
            tokens.append(Token(TokenType.MULTIPLY, '*', i))
            i += 1
        elif ch in ('/', '÷'):
            tokens.append(Token(TokenType.DIVIDE, '÷', i))
            i += 1
        elif ch == '%':
            tokens.append(Token(TokenType.MODULO, '%', i))
            i += 1
        elif ch in ('(', ')'):
            raise MathEvaluationError(
                "Parentheses are not supported in arithmetic expressions.",
                "unsupported_character"
            )
        elif ch == '=':
            raise MathEvaluationError(
                "The equals sign '=' is not in the dataset and cannot be recognized. Do not write '=' — click Solve to compute the answer.",
                "unsupported_character"
            )
        else:
            raise MathEvaluationError(
                f"Unsupported character '{ch}' at position {i}. Only digits 0-9 and operators +, -, *, ÷, % are supported.",
                "unsupported_character"
            )

        if len(tokens) > MAX_TOKEN_COUNT:
            raise MathEvaluationError(
                f"Expression exceeds maximum allowed token count of {MAX_TOKEN_COUNT}.",
                "invalid_expression"
            )

    tokens.append(Token(TokenType.EOF, "", len(s_eval)))
    return tokens


class ExpressionParser:
    """
    Recursive-descent precedence parser:
      expr       -> term (( '+' | '-' ) term)*
      term       -> factor (( '*' | '÷' | '%' ) factor)*
      factor     -> ('+' | '-')? NUMBER
    Ensures DMAS / BODMAS left-to-right evaluation:
      - Multiplicative (*, ÷, %) evaluated first, left-to-right.
      - Additive (+, -) evaluated next, left-to-right.
    """
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.current()
        if tok.type != TokenType.EOF:
            self.pos += 1
        return tok

    def parse(self) -> Union[int, float]:
        if self.current().type == TokenType.EOF:
            raise MathEvaluationError("Expression is empty.", "invalid_expression")

        result = self.parse_expr()

        if self.current().type != TokenType.EOF:
            unexpected = self.current()
            raise MathEvaluationError(
                f"Unexpected token '{unexpected.value}' at position {unexpected.pos}. Check expression syntax.",
                "invalid_expression"
            )

        return result

    def parse_expr(self) -> Union[int, float]:
        left = self.parse_term(allow_unary=True)

        while self.current().type in (TokenType.PLUS, TokenType.MINUS):
            op_tok = self.advance()
            # Do not allow consecutive operators e.g. 2 + + 3 or 2 + - 3
            right = self.parse_term(allow_unary=False)

            if op_tok.type == TokenType.PLUS:
                left = left + right
            else:
                left = left - right

            if abs(left) > MAX_MAGNITUDE:
                raise MathEvaluationError(
                    "Intermediate calculation exceeded maximum numeric limit.",
                    "invalid_expression"
                )

        return left

    def parse_term(self, allow_unary: bool = True) -> Union[int, float]:
        left = self.parse_factor(allow_unary=allow_unary)

        while self.current().type in (TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO):
            op_tok = self.advance()
            # Multiplicative operators can be followed by a unary signed number e.g. 5 * -3
            right = self.parse_factor(allow_unary=True)

            if op_tok.type == TokenType.MULTIPLY:
                left = left * right

            elif op_tok.type == TokenType.DIVIDE:
                if right == 0:
                    raise MathEvaluationError(
                        "Division by zero is undefined.",
                        "division_by_zero"
                    )
                left = left / right

            elif op_tok.type == TokenType.MODULO:
                if right == 0:
                    raise MathEvaluationError(
                        "Modulo by zero is undefined.",
                        "division_by_zero"
                    )
                # Remainder behavior:
                # Standard truncated modulo matching 17 % 5 = 2
                left = left % right

            if abs(left) > MAX_MAGNITUDE:
                raise MathEvaluationError(
                    "Intermediate calculation exceeded maximum numeric limit.",
                    "invalid_expression"
                )

        return left

    def parse_factor(self, allow_unary: bool = True) -> Union[int, float]:
        tok = self.current()

        # Handle unary prefix (+ or -)
        unary_sign = 1
        if tok.type in (TokenType.PLUS, TokenType.MINUS):
            if not allow_unary:
                raise MathEvaluationError(
                    f"Consecutive operators at position {tok.pos}. Consecutive '+' or '-' operators are not allowed.",
                    "invalid_expression"
                )
            op = self.advance()
            if op.type == TokenType.MINUS:
                unary_sign = -1
            tok = self.current()

        if tok.type == TokenType.NUMBER:
            self.advance()
            val = tok.value * unary_sign
            return val

        if tok.type == TokenType.EOF:
            raise MathEvaluationError(
                "Expression ends with a trailing operator.",
                "invalid_expression"
            )

        raise MathEvaluationError(
            f"Expected a number at position {tok.pos}, found '{tok.value}'.",
            "invalid_expression"
        )


def format_number(val: Union[int, float]) -> str:
    """
    Formats calculated number:
      - Integers display without decimal: 14 -> '14'
      - Whole floats display as integer: 4.0 -> '4'
      - Fractional floats display cleanly: 3.5 -> '3.5', 3.333333333333 -> '3.333333'
    """
    if isinstance(val, int):
        return str(val)

    if val.is_integer():
        return str(int(val))

    # Round to at most 6 decimal places and strip trailing zeros
    rounded = round(val, 6)
    if rounded.is_integer():
        return str(int(rounded))
    s = f"{rounded:.6f}".rstrip('0').rstrip('.')
    return s


def evaluate_expression(raw_text: str) -> EvaluationResult:
    """
    Evaluates a recognized handwritten arithmetic expression string safely.
    Returns structured EvaluationResult containing status, cleaned expression,
    calculated answer, and full equation (e.g. '2 + 3 * 4 = 14').
    """
    clean_text = raw_text.strip()
    # Normalize unicode division and multiplication
    normalized = clean_text.replace("÷", "/").replace("×", "*")

    # Display equation representation with standard unicode operators
    display_expr = (
        clean_text
        .replace("/", " ÷ ")
        .replace("÷", " ÷ ")
        .replace("*", " × ")
        .replace("×", " × ")
        .replace("+", " + ")
        .replace("-", " - ")
        .replace("%", " % ")
    )
    # Collapse multiple spaces
    display_expr = re.sub(r"\s+", " ", display_expr).strip()

    try:
        tokens = tokenize(clean_text)
        parser = ExpressionParser(tokens)
        num_result = parser.parse()

        # Check for integer equivalent
        if isinstance(num_result, float) and num_result.is_integer():
            final_result: Union[int, float] = int(num_result)
        else:
            final_result = num_result

        formatted_res = format_number(final_result)
        full_eq = f"{display_expr} = {formatted_res}"

        return EvaluationResult(
            status="success",
            expression=display_expr,
            result=final_result,
            formatted_result=formatted_res,
            full_equation=full_eq,
            error_message=None,
        )

    except MathEvaluationError as e:
        return EvaluationResult(
            status=e.error_type,
            expression=display_expr or clean_text,
            result=None,
            formatted_result=None,
            full_equation=None,
            error_message=str(e),
        )
    except Exception as e:
        return EvaluationResult(
            status="error",
            expression=display_expr or clean_text,
            result=None,
            formatted_result=None,
            full_equation=None,
            error_message=f"Evaluation error: {str(e)}",
        )
