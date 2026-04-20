"""Helpers for request-scoped metadata retrieval filters."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Any, Mapping, Sequence, TypeAlias

from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from loguru import logger
from qdrant_client import models

_SCALAR_OPERATORS: dict[str, FilterOperator] = {
    "eq": FilterOperator.EQ,
    "neq": FilterOperator.NE,
    "gt": FilterOperator.GT,
    "gte": FilterOperator.GTE,
    "lt": FilterOperator.LT,
    "lte": FilterOperator.LTE,
    "contains": FilterOperator.CONTAINS,
}

_QdrantCondition: TypeAlias = (
    models.FieldCondition
    | models.IsEmptyCondition
    | models.IsNullCondition
    | models.HasIdCondition
    | models.HasVectorCondition
    | models.NestedCondition
    | models.Filter
)


def build_metadata_filters(
    raw_rules: Sequence[Any] | None,
) -> MetadataFilters | None:
    """Build LlamaIndex metadata filters from API/UI payloads.

    Args:
        raw_rules (Sequence[Any] | None): Sequence of filter-like mappings or objects exposing ``field``,
            ``operator``, ``value``, and optional ``values`` attributes.

    Returns:
        MetadataFilters | None: A combined ``MetadataFilters`` instance, or ``None`` when no valid rules
        were supplied.
    """
    compiled: list[MetadataFilter | MetadataFilters] = []
    for raw_rule in raw_rules or []:
        rule = _coerce_rule(raw_rule)
        if rule is None:
            continue
        compiled_rule = _compile_rule(rule)
        if compiled_rule is not None:
            compiled.append(compiled_rule)

    if not compiled:
        return None
    return MetadataFilters(filters=compiled, condition=FilterCondition.AND)


def build_qdrant_filter(raw_rules: Sequence[Any] | None) -> models.Filter | None:
    """Build a native Qdrant filter for request-scoped retrieval.

    Args:
        raw_rules (Sequence[Any] | None): Sequence of filter-like mappings or objects exposing ``field``,
            ``operator``, ``value``, and optional ``values`` attributes.

    Returns:
        models.Filter | None: A Qdrant ``models.Filter`` instance, or ``None`` when no valid rules were supplied.
    """
    must: list[_QdrantCondition] = []
    must_not: list[_QdrantCondition] = []

    for raw_rule in raw_rules or []:
        rule = _coerce_rule(raw_rule)
        if rule is None:
            continue
        compiled_rule, negate = _compile_qdrant_rule(rule)
        if compiled_rule is None:
            continue
        if negate:
            must_not.append(compiled_rule)
        else:
            must.append(compiled_rule)

    if not must and not must_not:
        return None
    return models.Filter(must=must or None, must_not=must_not or None)


def matches_metadata_filters(
    metadata: Mapping[str, Any],
    raw_rules: Sequence[Any] | None,
) -> bool:
    """Return whether ``metadata`` satisfies all supplied request filter rules.

    This helper is used for best-effort in-memory filtering of sources that are
    retrieved outside the main vector-store query path, such as auxiliary image
    matches from the dedicated image collection.

    Args:
        metadata (Mapping[str, Any]): Metadata payload to test.
        raw_rules (Sequence[Any] | None): Sequence of filter-like mappings or objects exposing
            ``field``, ``operator``, ``value``, and optional ``values``.

    Returns:
        bool: ``True`` when every valid rule matches the metadata payload, otherwise
        ``False``.
    """
    for raw_rule in raw_rules or []:
        rule = _coerce_rule(raw_rule)
        if rule is None:
            continue
        if not _matches_rule(metadata, rule):
            return False
    return True


def _coerce_rule(raw_rule: Any) -> dict[str, Any] | None:
    """Normalize a raw filter payload into a plain dictionary.

    Args:
    raw_rule: A filter-like mapping or object exposing ``field``, ``operator``,
        ``value``, and optional ``values`` attributes.

    Returns:
        dict[str, Any] | None: A normalized dictionary with string keys and scalar or list values, or
            ``None`` if the input cannot be coerced into a valid rule.
    """
    if isinstance(raw_rule, Mapping):
        data = dict(raw_rule)
    else:
        data = {
            "field": getattr(raw_rule, "field", None),
            "operator": getattr(raw_rule, "operator", None),
            "value": getattr(raw_rule, "value", None),
            "values": getattr(raw_rule, "values", None),
        }

    field = str(data.get("field") or "").strip()
    operator = str(data.get("operator") or "").strip().lower()
    if not field or not operator:
        return None

    values = data.get("values")
    if values is not None and not isinstance(values, list):
        values = list(values) if isinstance(values, tuple) else None

    return {
        "field": field,
        "operator": operator,
        "value": data.get("value"),
        "values": values,
    }


def _matches_rule(metadata: Mapping[str, Any], rule: dict[str, Any]) -> bool:
    """Return whether a normalized rule matches an in-memory metadata payload.

    Args:
        metadata (Mapping[str, Any]): The metadata payload to test.
        rule (dict[str, Any]): A normalized rule dictionary.

    Returns:
        bool: True if the rule matches the metadata, False otherwise.
    """
    operator = rule["operator"]
    values = _extract_field_values(metadata, rule["field"])
    if not values:
        return False

    if operator in {"eq", "neq"}:
        expected = _normalize_scalar(rule.get("value"))
        if expected is None:
            return False
        matched = any(_coerce_comparable(value) == expected for value in values)
        return not matched if operator == "neq" else matched

    if operator in {"gt", "gte", "lt", "lte"}:
        expected = _normalize_scalar(rule.get("value"))
        if not isinstance(expected, (int, float)):
            return False
        numeric_values = [
            value
            for value in (_coerce_numeric(value) for value in values)
            if value is not None
        ]
        if not numeric_values:
            return False
        op_map = {
            "gt": lambda value: value > expected,
            "gte": lambda value: value >= expected,
            "lt": lambda value: value < expected,
            "lte": lambda value: value <= expected,
        }
        return any(op_map[operator](value) for value in numeric_values)

    if operator == "in":
        expected_values = _normalize_values(rule.get("values"))
        if not expected_values:
            scalar = _normalize_scalar(rule.get("value"))
            expected_values = [scalar] if scalar is not None else []
        if not expected_values:
            return False
        normalized_actual = {_coerce_comparable(value) for value in values}
        return any(value in normalized_actual for value in expected_values)

    if operator == "contains":
        expected = _normalize_scalar(rule.get("value"))
        if not isinstance(expected, str):
            return False
        needle = expected.lower()
        return any(needle in str(value).lower() for value in values)

    if operator == "mime_match":
        pattern = str(rule.get("value") or "").strip().lower()
        if not pattern:
            return False
        if pattern.endswith("/*"):
            prefix = pattern[:-1]
            return any(str(value).lower().startswith(prefix) for value in values)
        return any(str(value).lower() == pattern for value in values)

    if operator in {
        "date_after",
        "date_on_or_after",
        "date_before",
        "date_on_or_before",
    }:
        boundary = _parse_date_value(
            rule.get("value"),
            upper_bound=operator.endswith("before"),
        )
        if boundary is None:
            return False
        parsed_values = [
            value
            for value in (
                _parse_date_value(value, upper_bound=False) for value in values
            )
            if value is not None
        ]
        if not parsed_values:
            return False
        op_map = {
            "date_after": lambda value: value > boundary,
            "date_on_or_after": lambda value: value >= boundary,
            "date_before": lambda value: value < boundary,
            "date_on_or_before": lambda value: value <= boundary,
        }
        return any(op_map[operator](value) for value in parsed_values)

    logger.warning("Ignoring unsupported in-memory metadata operator '{}'.", operator)
    return False


def _extract_field_values(metadata: Mapping[str, Any], field: str) -> list[Any]:
    """Return one or more values for a dotted metadata field path.

    Args:
        metadata (Mapping[str, Any]): The metadata payload to extract values from.
        field (str): The dotted path to the desired field, e.g. ``author.name``.

    Returns:
        list[Any]: A list of values found at the specified field path. If the path does not exist,
        an empty list is returned.
    """
    parts = [part for part in field.split(".") if part]
    if not parts:
        return []

    current_values: list[Any] = [metadata]
    for part in parts:
        next_values: list[Any] = []
        for current in current_values:
            if isinstance(current, Mapping):
                candidate = current.get(part)
                if candidate is not None:
                    next_values.append(candidate)
            elif isinstance(current, list):
                for item in current:
                    if isinstance(item, Mapping):
                        candidate = item.get(part)
                        if candidate is not None:
                            next_values.append(candidate)
        if not next_values:
            return []
        current_values = next_values

    flattened: list[Any] = []
    for value in current_values:
        if isinstance(value, list):
            flattened.extend(value)
        else:
            flattened.append(value)
    return flattened


def _coerce_comparable(value: Any) -> str | int | float | bool | None:
    """Normalize an in-memory value for scalar comparisons."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if value is None:
        return None
    return str(value).strip()


def _coerce_numeric(value: Any) -> int | float | None:
    """Return a numeric representation for scalar comparisons when possible."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _compile_rule(rule: dict[str, Any]) -> MetadataFilter | MetadataFilters | None:
    """Translate a normalized rule dictionary into a vector-store filter.

    Args:
        rule (dict[str, Any]): A dictionary with keys ``field``, ``operator``, ``value``, and optional
            ``values``, where ``field`` is the metadata key to filter on, and
            ``operator`` is one of the supported filter operations.

    Returns:
        MetadataFilter | MetadataFilters | None: A ``MetadataFilter`` or ``MetadataFilters`` instance, or ``None`` if the rule
        cannot be compiled.
    """
    field = rule["field"]
    operator = rule["operator"]

    if operator in _SCALAR_OPERATORS:
        value = _normalize_scalar(rule.get("value"))
        if value is None:
            return None
        if isinstance(value, bool):
            logger.debug(
                "Skipping boolean MetadataFilter for '{}' and relying on native Qdrant filters.",
                field,
            )
            return None
        return MetadataFilter(
            key=field,
            value=value,
            operator=_SCALAR_OPERATORS[operator],
        )

    if operator == "in":
        values = _normalize_values(rule.get("values"))
        if not values:
            scalar = _normalize_scalar(rule.get("value"))
            values = [scalar] if scalar is not None else []
        if not values:
            return None
        return MetadataFilter(key=field, value=values, operator=FilterOperator.IN)

    if operator == "mime_match":
        return _compile_mime_rule(field=field, raw_value=rule.get("value"))

    if operator in {
        "date_after",
        "date_on_or_after",
        "date_before",
        "date_on_or_before",
    }:
        return _compile_date_rule(
            field=field, operator=operator, raw_value=rule.get("value")
        )

    logger.warning("Ignoring unsupported metadata filter operator '{}'.", operator)
    return None


def _compile_mime_rule(
    *,
    field: str,
    raw_value: Any,
) -> MetadataFilter | None:
    """Compile MIME filters, including simple ``type/*`` wildcard patterns.

    Args:
        field (str): The metadata field to filter on, typically ``mimetype``.
        raw_value (Any): The raw MIME pattern value, which may include a ``/*`` suffix for wildcard matching.

    Returns:
        MetadataFilter | None: A ``MetadataFilter`` instance with the appropriate operator, or ``None`` if the value is invalid.
    """
    value = str(raw_value or "").strip().lower()
    if not value:
        return None
    if value.endswith("/*"):
        return MetadataFilter(
            key=field,
            value=value[:-1],
            operator=FilterOperator.TEXT_MATCH_INSENSITIVE,
        )
    return MetadataFilter(key=field, value=value, operator=FilterOperator.EQ)


def _compile_date_rule(
    *,
    field: str,
    operator: str,
    raw_value: Any,
) -> MetadataFilter | None:
    """Compile date operators into ISO-8601 string comparisons.

    Args:
        field (str): The metadata field to filter on, e.g. ``reference_metadata.timestamp``.
        operator (str): The date comparison operator, one of ``date_after``, ``date_on_or_after``,
            ``date_before``, or ``date_on_or_before``.
        raw_value (Any): The raw date or datetime value, which may be a string, date/datetime object,
            or other type coercible to a date.

    Returns:
        MetadataFilter | None: A ``MetadataFilter`` instance with the appropriate operator and ISO-8601 value, or ``None`` if the value is invalid.
    """
    boundary = _normalize_date_value(raw_value, upper_bound=operator.endswith("before"))
    if boundary is None:
        logger.warning(
            "Ignoring metadata date filter for '{}' due to invalid value '{}'.",
            field,
            raw_value,
        )
        return None

    op_map = {
        "date_after": FilterOperator.GT,
        "date_on_or_after": FilterOperator.GTE,
        "date_before": FilterOperator.LT,
        "date_on_or_before": FilterOperator.LTE,
    }
    return MetadataFilter(key=field, value=boundary, operator=op_map[operator])


def _compile_qdrant_rule(
    rule: dict[str, Any],
) -> tuple[_QdrantCondition | None, bool]:
    """Translate a normalized rule into a native Qdrant condition.

    Args:
        rule (dict[str, Any]): A dictionary with keys ``field``, ``operator``, ``value``, and optional ``values``,
            where ``field`` is the metadata key to filter on, and ``operator`` is one of the
            supported filter operations.

    Returns:
        tuple[_QdrantCondition | None, bool]: A tuple of (compiled_condition, negate), where ``compiled_condition`` is a Qdrant-compatible
            condition or filter, and ``negate`` is a boolean indicating whether the condition should be
            negated (i.e., added to the ``must_not`` list instead of ``must``).
    """
    field = rule["field"]
    operator = rule["operator"]

    if operator in {"eq", "neq"}:
        value = _normalize_scalar(rule.get("value"))
        if value is None:
            return None, False
        return (
            models.FieldCondition(key=field, match=models.MatchValue(value=value)),
            operator == "neq",
        )

    if operator in {"gt", "gte", "lt", "lte"}:
        value = _normalize_scalar(rule.get("value"))
        if not isinstance(value, (int, float)):
            return None, False
        range_kwargs = {operator: value}
        return (
            models.FieldCondition(key=field, range=models.Range(**range_kwargs)),
            False,
        )

    if operator == "in":
        values = _normalize_values(rule.get("values"))
        if not values:
            scalar = _normalize_scalar(rule.get("value"))
            values = [scalar] if scalar is not None else []
        if not values:
            return None, False
        return (
            models.FieldCondition(key=field, match=models.MatchAny(any=values)),
            False,
        )

    if operator == "contains":
        value = _normalize_scalar(rule.get("value"))
        if not isinstance(value, str):
            return None, False
        return (
            models.FieldCondition(key=field, match=models.MatchText(text=value)),
            False,
        )

    if operator == "mime_match":
        value = str(rule.get("value") or "").strip().lower()
        if not value:
            return None, False
        if value.endswith("/*"):
            return (
                models.FieldCondition(
                    key=field,
                    match=models.MatchText(text=value[:-1]),
                ),
                False,
            )
        return (
            models.FieldCondition(key=field, match=models.MatchValue(value=value)),
            False,
        )

    if operator in {
        "date_after",
        "date_on_or_after",
        "date_before",
        "date_on_or_before",
    }:
        boundary = _parse_date_value(
            rule.get("value"),
            upper_bound=operator.endswith("before"),
        )
        if boundary is None:
            logger.warning(
                "Ignoring Qdrant date filter for '{}' due to invalid value '{}'.",
                field,
                rule.get("value"),
            )
            return None, False
        range_key = {
            "date_after": "gt",
            "date_on_or_after": "gte",
            "date_before": "lt",
            "date_on_or_before": "lte",
        }[operator]
        return (
            models.FieldCondition(
                key=field,
                range=models.DatetimeRange(**{range_key: boundary}),
            ),
            False,
        )

    logger.warning("Ignoring unsupported Qdrant filter operator '{}'.", operator)
    return None, False


def _normalize_scalar(value: Any) -> str | int | float | bool | None:
    """Return a vector-store-safe scalar, skipping empty values.

    Args:
        value (Any): The raw input value to normalize, which may be of any type.

    Returns:
        str | int | float | bool | None: A normalized scalar value (string, number, or boolean), or ``None`` if the
            value is empty or cannot be coerced into a valid scalar.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _normalize_values(values: Any) -> list[str | int | float | bool]:
    """Normalize a sequence of scalar values for ``IN`` filters.

    Args:
        values (Any): The raw input value to normalize, which may be a list, tuple, or
            other iterable of scalar values.

    Returns:
        list[str | int | float | bool]: A list of normalized scalar values. Empty or invalid entries will be skipped,
            and non-list inputs will return an empty list.
    """
    if not isinstance(values, list):
        return []
    normalized: list[str | int | float | bool] = []
    for value in values:
        scalar = _normalize_scalar(value)
        if scalar is not None:
            normalized.append(scalar)
    return normalized


def _normalize_date_value(value: Any, *, upper_bound: bool) -> str | None:
    """Normalize a date or datetime input into a UTC ISO-8601 string.

    Args:
        value (Any): The raw date or datetime value, which may be a string, date/datetime object,
            or other type coercible to a date.
        upper_bound (bool): Whether the value represents an upper bound (i.e., "before" operator)
            that should be normalized.

    Returns:
        str | None: A normalized UTC ISO-8601 string, or ``None`` if the value is invalid.
    """
    dt = _parse_date_value(value, upper_bound=upper_bound)
    if dt is None:
        return None
    return dt.isoformat().replace("+00:00", "Z")


def _parse_date_value(value: Any, *, upper_bound: bool) -> datetime | None:
    """Parse a date or datetime input into a UTC datetime.

    Args:
        value (Any): The raw date or datetime value, which may be a string, date/datetime
            object, or other type coercible to a date.
        upper_bound (bool): Whether the value represents an upper bound (i.e., "before" operator)
            that should be normalized with time.max.

    Returns:
        datetime | None: A UTC datetime object, or ``None`` if the value is invalid.
    """
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime.combine(
            value,
            time.max if upper_bound else time.min,
            tzinfo=timezone.utc,
        )
    else:
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            if len(raw) == 10:
                parsed_date = date.fromisoformat(raw)
                dt = datetime.combine(
                    parsed_date,
                    time.max if upper_bound else time.min,
                    tzinfo=timezone.utc,
                )
            else:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt
