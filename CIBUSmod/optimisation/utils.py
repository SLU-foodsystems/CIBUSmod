from typing import Callable, Literal, TypedDict
import cvxpy

class Constraint(TypedDict):
    left: Callable
    right: Callable
    rel: Literal["==", ">=", "<="]
    pars: dict


def make_cvxpy_constraint(cons: Constraint, x: cvxpy.Variable) -> cvxpy.Constraint:
    """
    Convert a Constraint-dict to a cvxpy.Constraint instant
    """
    operators = {
        "==": lambda left, right: left == right,
        ">=": lambda left, right: left >= right,
        "<=": lambda left, right: left <= right,
    }
    left = cons["left"]
    right = cons["right"]
    rel = cons["rel"]
    pars = cons["pars"]

    return operators[rel](left(x, **pars), right(**pars))
