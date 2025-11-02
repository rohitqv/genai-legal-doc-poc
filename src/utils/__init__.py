"""
Utils package for Legal Document Analysis PoC
"""
from .config import *
from .logger import logger
from .delta_helpers import (
    get_spark_session,
    initialize_all_tables,
    log_audit_event
)

__all__ = [
    "logger",
    "get_spark_session",
    "initialize_all_tables",
    "log_audit_event"
]

