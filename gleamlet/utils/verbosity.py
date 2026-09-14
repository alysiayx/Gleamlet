"""Shared output controls for Gleamlet's public API."""

from __future__ import annotations

import inspect
import io
import logging
from contextlib import contextmanager, redirect_stdout
from contextvars import ContextVar
from functools import wraps


_VERBOSE_OVERRIDE: ContextVar[bool | int | None] = ContextVar(
    "gleamlet_verbose_override",
    default=None,
)


def routine_output_enabled() -> bool:
    """Return whether routine output is enabled in the current call context."""
    requested = _VERBOSE_OVERRIDE.get()
    return True if requested is None else bool(requested)


@contextmanager
def _user_output(logger: logging.Logger | None, verbose: bool | int):
    """Hide routine console output while preserving warnings and errors."""
    if verbose:
        yield
        return

    console_handlers = [] if logger is None else [
        handler
        for handler in logger.handlers
        if not isinstance(handler, logging.FileHandler)
    ]
    previous_levels = [handler.level for handler in console_handlers]
    try:
        for handler in console_handlers:
            handler.setLevel(logging.WARNING)
        with redirect_stdout(io.StringIO()):
            yield
    finally:
        for handler, level in zip(console_handlers, previous_levels):
            handler.setLevel(level)


def control_user_output(logger: logging.Logger | None = None):
    """Decorate a public callable that exposes a ``verbose`` argument."""
    def decorator(function):
        signature = inspect.signature(function)
        parameters = tuple(signature.parameters)
        is_instance_method = bool(parameters) and parameters[0] == "self"

        @wraps(function)
        def wrapper(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            requested = bound.arguments.get("verbose")
            instance = args[0] if is_instance_method and args else None
            inherited = _VERBOSE_OVERRIDE.get()
            if requested is not None:
                effective = requested
            elif inherited is not None:
                effective = inherited
            else:
                effective = getattr(instance, "verbose", True)

            previous = getattr(instance, "verbose", None) if instance is not None else None
            has_instance_default = instance is not None and hasattr(instance, "verbose")
            if has_instance_default and requested is not None:
                instance.verbose = requested
            token = _VERBOSE_OVERRIDE.set(effective)
            try:
                with _user_output(logger, effective):
                    return function(*args, **kwargs)
            finally:
                _VERBOSE_OVERRIDE.reset(token)
                if has_instance_default and requested is not None:
                    instance.verbose = previous

        return wrapper

    return decorator


def control_class_verbosity(logger: logging.Logger | None = None):
    """Apply output control to public methods and the class constructor."""
    def decorator(cls):
        for name, attribute in list(vars(cls).items()):
            if name != "__init__" and name.startswith("_"):
                continue
            if isinstance(attribute, property):
                continue
            if isinstance(attribute, staticmethod):
                wrapped = staticmethod(control_user_output(logger)(attribute.__func__))
            elif isinstance(attribute, classmethod):
                wrapped = classmethod(control_user_output(logger)(attribute.__func__))
            elif callable(attribute):
                wrapped = control_user_output(logger)(attribute)
            else:
                continue
            setattr(cls, name, wrapped)
        return cls

    return decorator
