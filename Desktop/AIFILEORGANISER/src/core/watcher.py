"""
File Watcher Module

Copyright (c) 2025 Alexandru Emanuel Vasile. All rights reserved.
Proprietary Software - 200-Key Limited Release License

This module monitors specified directories for new or modified files using
the watchdog library. Detected files are queued for classification and processing.

NOTICE: This software is proprietary and confidential.
See LICENSE.txt for full terms and conditions.

Author: Alexandru Emanuel Vasile
License: Proprietary (200-key limited release)
"""

import time
import threading
import os
from pathlib import Path
from typing import List, Callable, Optional
from queue import Queue
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent


class FileEventHandler(FileSystemEventHandler):
    """
    Custom event handler for file system changes.

    This handler filters and processes file creation and modification events,
    adding relevant files to a processing queue.

    Attributes:
        callback (Callable): Function to call when a file event occurs
        file_queue (Queue): Queue for detected files
        ignored_extensions (set): File extensions to ignore
        ignored_patterns (set): Filename patterns to ignore
    """

    def __init__(self, callback: Callable = None, file_queue: Queue = None, blacklist: Optional[List[str]] = None, max_queue_size: int = 1000):
        """
        Initialize file event handler.

        Args:
            callback (Callable, optional): Function to call with file path when event occurs
            file_queue (Queue, optional): Queue to add detected files to
            blacklist (List[str], optional): List of paths to ignore
            max_queue_size (int): Maximum queue size to prevent memory leak (HIGH #4 FIX)
        """
        super().__init__()
        self.callback = callback
        self.file_queue = file_queue or Queue(maxsize=max_queue_size)  # Add maxsize (HIGH #4 FIX)
        # Optional list of path prefixes to ignore
        self.blacklist = [str(Path(p).expanduser().resolve()) for p in (blacklist or [])]

        # Ignore temporary and system files
        self.ignored_extensions = {
            '.tmp', '.temp', '.crdownload', '.part', '.partial',
            '.download', '.cache', '.lock', '.swp', '.~'
        }

        self.ignored_patterns = {
            '.DS_Store', 'Thumbs.db', 'desktop.ini', '.gitkeep',
            '.git', '.svn', '__pycache__', 'node_modules'
        }

    def _should_process(self, path: str) -> bool:
        """
        Determine if a file should be processed.

        Args:
            path (str): File path

        Returns:
            bool: True if file should be processed, False if should be ignored
        """
        file_path = Path(path)

        # Ignore directories
        if file_path.is_dir():
            return False

        # Ignore files in ignored patterns
        if file_path.name in self.ignored_patterns:
            return False

        # Ignore hidden files (starting with .)
        if file_path.name.startswith('.'):
            return False

        # Ignore files with ignored extensions
        if file_path.suffix.lower() in self.ignored_extensions:
            return False

        # Ignore very small files (likely incomplete or empty)
        try:
            if file_path.stat().st_size < 100:  # Less than 100 bytes
                return False
        except OSError:
            return False

        # Ignore files under any blacklisted path
        try:
            resolved = str(file_path.resolve())
            for b in self.blacklist:
                try:
                    # Use commonpath to determine ancestry; handle different drives
                    if os.path.commonpath([resolved, b]) == b:
                        return False
                except Exception:
                    # If commonpath fails (different drives), try simple prefix compare
                    if resolved.lower().startswith(b.lower()):
                        return False
        except Exception:
            pass

        return True

    def on_created(self, event: FileSystemEvent):
        """
        Handle file creation events.

        Args:
            event (FileSystemEvent): File system event
        """
        if not event.is_directory and self._should_process(event.src_path):
            # Small delay to ensure file is fully written
            time.sleep(0.5)

            # Re-check file still exists (might have been temp file)
            if Path(event.src_path).exists():
                self._process_file(event.src_path)

    def on_modified(self, event: FileSystemEvent):
        """
        Handle file modification events.

        Args:
            event (FileSystemEvent): File system event
        """
        # For modified files, we're more conservative to avoid processing
        # files that are being actively written to
        if not event.is_directory and self._should_process(event.src_path):
            # Only process if file hasn't been modified in last 2 seconds
            # This prevents processing files that are still being written
            try:
                path = Path(event.src_path)
                if time.time() - path.stat().st_mtime > 2:
                    self._process_file(event.src_path)
            except OSError:
                pass

    def _process_file(self, file_path: str):
        """
        Process a detected file.

        Args:
            file_path (str): Path to the file
        """
        if self.callback:
            self.callback(file_path)

        if self.file_queue:
            self.file_queue.put(file_path)


class FolderWatcher:
    """
    Main folder watching orchestrator.

    This class manages observers for multiple directories and coordinates
    file detection with classification and processing.

    Attributes:
        folders (List[str]): Directories to watch
        callback (Callable): Function to call when files are detected
        observer (Observer): Watchdog observer instance
        file_queue (Queue): Queue of detected files
        processing_thread (threading.Thread): Background processing thread
    """

    def __init__(self, folders: List[str], callback: Callable = None, config: object = None):
        """
        Initialize folder watcher.

        Args:
            folders (List[str]): List of directory paths to watch
            callback (Callable, optional): Function to call with file path when detected
        """
        self.folders = [Path(f).expanduser().resolve() for f in folders]
        self.callback = callback
        self.config = config
        self.observer: Optional[Observer] = None
        self.file_queue = Queue()
        self.processing_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self, background: bool = True):
        """
        Start watching folders.

        Args:
            background (bool): If True, run processing in background thread
        """
        if self._running:
            print("[Watcher] Already running")
            return

        # Create event handler
        blacklist = []
        try:
            if self.config and hasattr(self.config, 'path_blacklist'):
                blacklist = self.config.path_blacklist
        except Exception:
            blacklist = []

        event_handler = FileEventHandler(
            callback=self.callback,
            file_queue=self.file_queue,
            blacklist=blacklist
        )

        # Create observer
        self.observer = Observer()

        # Schedule observers for all folders
        for folder in self.folders:
            if folder.exists() and folder.is_dir():
                self.observer.schedule(event_handler, str(folder), recursive=True)
                print(f"[Watcher] Monitoring: {folder}")
            else:
                print(f"[Watcher] Warning: Folder not found: {folder}")

        # Start observer
        self.observer.start()
        self._running = True
        print("[Watcher] Started successfully")

        # Start processing thread if background mode
        if background and self.callback:
            self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
            self.processing_thread.start()

    def stop(self):
        """Stop watching folders."""
        if not self._running:
            return

        self._running = False

        if self.observer:
            self.observer.stop()
            self.observer.join()

        print("[Watcher] Stopped")

    def _process_queue(self):
        """
        Process files from queue in background.
        This runs in a separate thread.
        """
        while self._running:
            try:
                # Get file from queue with timeout
                if not self.file_queue.empty():
                    file_path = self.file_queue.get(timeout=1)

                    # Call callback if provided
                    if self.callback and Path(file_path).exists():
                        try:
                            self.callback(file_path)
                        except Exception as e:
                            print(f"[Watcher] Error processing {file_path}: {e}")

                    self.file_queue.task_done()
                else:
                    time.sleep(0.5)
            except Exception as e:
                print(f"[Watcher] Queue processing error: {e}")
                time.sleep(1)

    def get_pending_count(self) -> int:
        """
        Get number of files waiting to be processed.

        Returns:
            int: Queue size
        """
        return self.file_queue.qsize()

    def scan_existing_files(self, callback: Callable = None) -> List[str]:
        """
        Scan watched folders for existing files (one-time scan).

        Args:
            callback (Callable, optional): Function to call for each found file

        Returns:
            List[str]: List of found file paths
        """
        found_files = []

        for folder in self.folders:
            if not folder.exists():
                continue

            print(f"[Watcher] Scanning existing files in: {folder}")

            # Walk through directory
            for item in folder.rglob('*'):
                if item.is_file():
                    # Use same filtering logic as event handler
                    event_handler = FileEventHandler()
                    if event_handler._should_process(str(item)):
                        found_files.append(str(item))

                        if callback:
                            try:
                                callback(str(item))
                            except Exception as e:
                                print(f"[Watcher] Error processing {item}: {e}")

        print(f"[Watcher] Found {len(found_files)} existing files")
        return found_files


def create_watcher(folders: List[str], callback: Callable = None) -> FolderWatcher:
    """
    Create and configure a folder watcher.

    Args:
        folders (List[str]): Directories to watch
        callback (Callable, optional): Function to call when files detected

    Returns:
        FolderWatcher: Configured watcher instance
    """
    return FolderWatcher(folders, callback)


if __name__ == "__main__":
    # Test watcher
    def test_callback(file_path: str):
        print(f"File detected: {file_path}")

    # Watch current directory for testing
    test_folders = ["."]

    watcher = create_watcher(test_folders, callback=test_callback)
    watcher.start()

    try:
        print("Watcher running... Press Ctrl+C to stop")
        while True:
            time.sleep(1)
            pending = watcher.get_pending_count()
            if pending > 0:
                print(f"Pending files in queue: {pending}")
    except KeyboardInterrupt:
        print("\nStopping watcher...")
        watcher.stop()
