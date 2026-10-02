"""Project-specific exceptions."""


class IvcastError(Exception):
    """Base error for the project."""


class DataValidationError(IvcastError):
    """Raised when input data violates explicit validation rules."""


class SchemaDriftError(IvcastError):
    """Raised when an upstream file no longer matches the expected schema."""


class TemporalIntegrityError(IvcastError):
    """Raised when timing assumptions are violated."""


class JoinCardinalityError(IvcastError):
    """Raised when a join violates the expected cardinality."""


class InterpolationError(IvcastError):
    """Raised when surface completion cannot deterministically finish."""


class ModelTrainingError(IvcastError):
    """Raised when model fitting fails under explicit training rules."""


class ModelConvergenceError(ModelTrainingError):
    """Raised when an optimizer emits an explicit convergence failure."""
