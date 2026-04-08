"""Project-specific exceptions."""


class IvsurfError(Exception):
    """Base error for the project."""


class DataValidationError(IvsurfError):
    """Raised when input data violates explicit validation rules."""


class SchemaDriftError(IvsurfError):
    """Raised when an upstream file no longer matches the expected schema."""


class TemporalIntegrityError(IvsurfError):
    """Raised when timing assumptions are violated."""


class JoinCardinalityError(IvsurfError):
    """Raised when a join violates the expected cardinality."""


class InterpolationError(IvsurfError):
    """Raised when surface completion cannot deterministically finish."""

