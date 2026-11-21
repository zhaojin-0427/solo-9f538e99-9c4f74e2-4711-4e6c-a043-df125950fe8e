"""Pure domain logic: handoff-package building and population conversion.

The math uses exact :class:`fractions.Fraction` arithmetic throughout.

Model
-----
Let ``x_b`` be the (unknown, non-negative integer) population of street
block ``b``. With 0/1 incidence matrices

* ``A`` (blocks x old districts):  old population   ``p = A x``
* ``B`` (blocks x new districts):  new population   ``q = B x``

the exact linear relations between populations are derived without ever
knowing the block populations themselves.
"""
from fractions import Fraction
from functools import reduce
from math import lcm
from typing import Any, Dict, List, Optional, Tuple

from .expressions import format_expression, format_fraction, parse_fraction
from .rationals import Vec, nullspace, primitive_integer, solve


class ConversionError(ValueError):
    """Raised when a whole conversion request must be rejected."""


# --------------------------------------------------------------------------
# Package builder
# --------------------------------------------------------------------------
def build_package(
    old_districts: List[Dict[str, Any]],
    new_districts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a block-free JSON handoff package.

    Each district is ``{"code": str, "blocks": [block_id, ...]}``.
    A block may be covered by any number of districts.
    """
    old_codes = [d["code"] for d in old_districts]
    new_codes = [d["code"] for d in new_districts]
    n_old = len(old_codes)

    blocks = sorted(
        {b for d in old_districts + new_districts for b in d["blocks"]}
    )

    def incidence(districts: List[Dict[str, Any]]):
        return [
            [
                Fraction(1 if block in district["blocks"] else 0)
                for district in districts
            ]
            for block in blocks
        ]

    a = incidence(old_districts)  # blocks x old
    b = incidence(new_districts)  # blocks x new

    # ker(A): every vector k with A k = 0 is an exact equation
    # sum_j k_j * p_j = 0 between old-district populations.
    kernel = nullspace(a, n_old)
    old_relations: List[Dict[str, int]] = []
    for k in kernel:
        ints = primitive_integer(k)
        relation = {code: val for code, val in zip(old_codes, ints) if val != 0}
        old_relations.append(relation)

    # A c_j = b_j  =>  q_j = c_j . p  whenever the system is consistent.
    new_relations = []
    for j, code in enumerate(new_codes):
        col = [row[j] for row in b]
        particular = solve(a, col, n_old)
        if particular is None:
            new_relations.append(
                {"code": code, "derivable": False, "coefficients": {}}
            )
        else:
            coefficients = {
                old_codes[i]: format_fraction(particular[i])
                for i in range(n_old)
                if particular[i] != 0
            }
            new_relations.append(
                {"code": code, "derivable": True, "coefficients": coefficients}
            )

    return {
        "schema_version": 1,
        "old_districts": old_codes,
        "new_districts": new_codes,
        "old_relations": old_relations,
        "new_district_relations": new_relations,
    }


# --------------------------------------------------------------------------
# Converter
# --------------------------------------------------------------------------
def _expand_relation(relation: Dict[str, int], old_codes: List[str]) -> Vec:
    return [Fraction(relation.get(code, 0)) for code in old_codes]


def _support(vector: Vec) -> List[int]:
    return [i for i, v in enumerate(vector) if v != 0]


def _vector_on_available(
    particular: Vec,
    kernel_basis: List[Vec],
    n_vars: int,
    available_idx,
) -> Tuple[bool, Vec]:
    """Find a coefficient vector supported on the available districts.

    All coefficient vectors describing the same new district differ from
    ``particular`` by a member of ``ker(A)``, which is spanned by the rows
    of ``kernel_basis`` (the package's old relations). Hence every solution
    has the form ``d = particular + M^T alpha`` for some rational ``alpha``.
    Requiring ``d`` to vanish on every district without data gives a linear
    system for ``alpha``; consistency is exactly the computability test.
    """
    r = len(kernel_basis)
    unavailable = [i for i in range(n_vars) if i not in available_idx]
    constraint = [
        [kernel_basis[j][i] for j in range(r)] for i in unavailable
    ]
    rhs = [-particular[i] for i in unavailable]
    alpha = solve(constraint, rhs, r)
    if alpha is None:
        return False, particular
    chosen = [
        particular[i]
        + sum(alpha[j] * kernel_basis[j][i] for j in range(r))
        for i in range(n_vars)
    ]
    return True, chosen


class RelationRule:
    """One exact equation between old districts, ready for evaluation.

    ``terms`` holds ``(old-district index, integer coefficient)`` pairs, so
    the left-hand side of the equation ``expression = 0`` is exactly
    ``sum(coef * populations[idx] for idx, coef in terms)``.
    """

    __slots__ = ("relation", "expression", "terms")

    def __init__(
        self,
        relation: Dict[str, int],
        expression: str,
        terms: List[Tuple[int, int]],
    ) -> None:
        self.relation = relation
        self.expression = expression
        self.terms = terms


class NewDistrictPlan:
    """How one new district is handled for one reported-district combination.

    For a computable district the exact population is
    ``sum(mult * populations[idx] for idx, mult in terms) / denominator``
    computed with integer arithmetic only; ``coefficients`` and
    ``expression`` are the display forms shared with the JSON API.
    """

    __slots__ = (
        "code",
        "status",
        "missing_reason",
        "coefficients",
        "expression",
        "terms",
        "denominator",
    )

    def __init__(
        self,
        code: str,
        status: str,
        missing_reason: Optional[str],
        coefficients: Dict[str, str],
        expression: Optional[str],
        terms: List[Tuple[int, int]],
        denominator: int,
    ) -> None:
        self.code = code
        self.status = status
        self.missing_reason = missing_reason
        self.coefficients = coefficients
        self.expression = expression
        self.terms = terms
        self.denominator = denominator


class PreparedRules:
    """All conversion rules for one combination of reported old districts.

    A ``PreparedRules`` is derived solely from the handoff package and the
    *set* of old districts that have data, so it is prepared once per
    reported-district combination and then applied to any number of rows,
    each row contributing its own non-negative integer populations.
    """

    __slots__ = ("old_codes", "new_codes", "available", "relations", "plans")

    def __init__(
        self,
        old_codes: List[str],
        new_codes: List[str],
        available: List[str],
        relations: List[RelationRule],
        plans: List[NewDistrictPlan],
    ) -> None:
        self.old_codes = old_codes
        self.new_codes = new_codes
        self.available = available
        self.relations = relations
        self.plans = plans


def prepare_rules(package: Dict[str, Any], available: List[str]) -> PreparedRules:
    """Prepare conversion rules for one set of available old districts.

    Computability is judged solely from the districts having data and the
    algebraic relations in the package.  The result contains every old
    relation that is fully checkable from the available districts and, for
    each new district, either an exact integer-accumulator formula or the
    reason it stays missing.
    """
    old_codes: List[str] = list(package["old_districts"])
    new_codes: List[str] = list(package["new_districts"])
    n_old = len(old_codes)
    available_idx = {old_codes.index(c) for c in available}

    kernel_basis: List[Vec] = [
        _expand_relation(rel, old_codes) for rel in package["old_relations"]
    ]

    relations: List[RelationRule] = []
    for k in kernel_basis:
        support = _support(k)
        if not set(support).issubset(available_idx):
            continue
        relations.append(
            RelationRule(
                relation={old_codes[i]: int(k[i]) for i in support},
                expression=format_expression(
                    {old_codes[i]: k[i] for i in support}
                ),
                terms=[(i, int(k[i])) for i in support],
            )
        )

    plans: List[NewDistrictPlan] = []
    for item in package["new_district_relations"]:
        code = item["code"]
        if not item["derivable"]:
            plans.append(
                NewDistrictPlan(
                    code=code,
                    status="missing",
                    missing_reason="not_expressible_from_old_districts",
                    coefficients={},
                    expression=None,
                    terms=[],
                    denominator=1,
                )
            )
            continue

        particular = [
            parse_fraction(item["coefficients"][old_codes[i]])
            if old_codes[i] in item["coefficients"]
            else Fraction(0)
            for i in range(n_old)
        ]
        ok, chosen = _vector_on_available(
            particular, kernel_basis, n_old, available_idx
        )
        if not ok:
            nonzero = {
                old_codes[i]: particular[i]
                for i in range(n_old)
                if particular[i] != 0
            }
            plans.append(
                NewDistrictPlan(
                    code=code,
                    status="missing",
                    missing_reason="insufficient_known_old_districts",
                    coefficients={
                        c: format_fraction(v) for c, v in nonzero.items()
                    },
                    expression=format_expression(nonzero),
                    terms=[],
                    denominator=1,
                )
            )
            continue

        nonzero = {
            old_codes[i]: chosen[i] for i in range(n_old) if chosen[i] != 0
        }
        # Exact integer accumulator: value == sum(terms) / denominator.
        denominator = reduce(lcm, (v.denominator for v in chosen), 1)
        terms = [
            (i, int(chosen[i] * denominator))
            for i in range(n_old)
            if chosen[i] != 0
        ]
        plans.append(
            NewDistrictPlan(
                code=code,
                status="computed",
                missing_reason=None,
                coefficients={c: format_fraction(v) for c, v in nonzero.items()},
                expression=format_expression(nonzero),
                terms=terms,
                denominator=denominator,
            )
        )

    return PreparedRules(old_codes, new_codes, list(available), relations, plans)


def evaluate_rules(
    rules: PreparedRules, populations: List[int]
) -> List[Optional[int]]:
    """Apply prepared rules to one row of non-negative integer populations.

    ``populations`` is indexed by old-district position (any value works for
    districts outside the prepared combination — no rule references them).
    Returns one entry per new district (``None`` where not derivable) in
    package order.  Raises :class:`ConversionError` on any violated relation
    or negative/non-integer result, exactly like the JSON ``/convert``.
    """
    for rel in rules.relations:
        value = 0
        for idx, coef in rel.terms:
            value += coef * populations[idx]
        if value != 0:
            raise ConversionError(
                "old-district populations violate the coverage relation "
                f"{rel.expression} = 0 (evaluates to {value})"
            )

    results: List[Optional[int]] = []
    for plan in rules.plans:
        if plan.status == "missing":
            results.append(None)
            continue
        numerator = 0
        for idx, mult in plan.terms:
            numerator += mult * populations[idx]
        if numerator < 0:
            raise ConversionError(
                f"new district {plan.code} converts to a negative population "
                f"({format_fraction(Fraction(numerator, plan.denominator))})"
            )
        if numerator % plan.denominator != 0:
            raise ConversionError(
                f"new district {plan.code} converts to a non-integer population "
                f"({format_fraction(Fraction(numerator, plan.denominator))})"
            )
        results.append(numerator // plan.denominator)
    return results


def convert(package: Dict[str, Any], old_populations: Dict[str, int]) -> Dict[str, Any]:
    """Convert known old-district populations into new-district populations.

    Computability is judged solely from the districts having data *in this
    request* and the algebraic relations in the package. On any violation
    (contradicted coverage relation, negative or fractional result) raises
    :class:`ConversionError` and nothing else is returned.
    """
    old_codes: List[str] = package["old_districts"]
    available = [code for code in old_codes if code in old_populations]
    populations = [int(old_populations.get(code, 0)) for code in old_codes]

    rules = prepare_rules(package, available)
    values = evaluate_rules(rules, populations)

    checked_relations = [
        {
            "relation": rel.relation,
            "expression": rel.expression,
            "evaluated_value": sum(
                coef * populations[idx] for idx, coef in rel.terms
            ),
            "satisfied": True,
        }
        for rel in rules.relations
    ]

    results: List[Dict[str, Any]] = []
    for plan, value in zip(rules.plans, values):
        results.append(
            {
                "code": plan.code,
                "status": plan.status,
                "population": value,
                "coefficients": plan.coefficients,
                "expression": plan.expression,
                "missing_reason": plan.missing_reason,
            }
        )

    computed = [r for r in results if r["status"] == "computed"]
    missing = [r for r in results if r["status"] == "missing"]
    return {
        "available_old_districts": available,
        "checked_relations": checked_relations,
        "computed": computed,
        "missing": missing,
        "results": results,
    }
