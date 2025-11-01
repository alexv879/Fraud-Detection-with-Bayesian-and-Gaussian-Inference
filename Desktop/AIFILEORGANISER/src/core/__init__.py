"""Core modules for file organization."""

from .db_manager import DatabaseManager
from .classifier import FileClassifier, classify_file
from .watcher import FolderWatcher, create_watcher
from .actions import ActionManager
from .duplicates import DuplicateFinder, find_duplicates

__all__ = [
    'DatabaseManager',
    'FileClassifier',
    'classify_file',
    'FolderWatcher',
    'create_watcher',
    'ActionManager',
    'DuplicateFinder',
    'find_duplicates'
]
