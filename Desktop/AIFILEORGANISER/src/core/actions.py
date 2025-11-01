"""
Action Manager Module

Copyright (c) 2025 Alexandru Emanuel Vasile. All rights reserved.
Proprietary Software - 200-Key Limited Release License

This module handles file operations (move, rename, delete, archive) based on
classification results. It supports dry-run mode, undo functionality, and
comprehensive logging of all actions.

NOTICE: This software is proprietary and confidential. Unauthorized copying,
modification, distribution, or use is strictly prohibited.
See LICENSE.txt for full terms and conditions.

Version: 1.0.0
Author: Alexandru Emanuel Vasile
License: Proprietary (200-key limited release)
"""

import os
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import json

# Initialize logger for audit trail (MEDIUM #2 FIX)
logger = logging.getLogger(__name__)


class ActionManager:
    """
    Manages file operations with safety features and logging.

    Attributes:
        config: Configuration object
        db_manager: Database manager for logging
        dry_run (bool): If True, simulate actions without actually performing them
        undo_history (List): Stack of recent actions for undo functionality
    """

    def __init__(self, config, db_manager, dry_run: bool = None):
        """
        Initialize action manager.

        Args:
            config: Configuration object
            db_manager: Database manager instance
            dry_run (bool, optional): Override config dry_run setting
        """
        self.config = config
        self.db_manager = db_manager
        self.dry_run = dry_run if dry_run is not None else config.dry_run
        self.undo_history: List[Dict[str, Any]] = []
        self.max_undo_history = 50  # Keep last 50 actions

    def execute(self, file_path: str, classification: Dict[str, Any],
                user_approved: bool = False, folder_policy: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute file organization action based on classification result.

        Args:
            file_path (str): Current file path
            classification (Dict): Classification result from classifier
            user_approved (bool): Whether user explicitly approved this action
            folder_policy (Dict, optional): Folder policy dict to override config lookup

        Returns:
            Dict: Action result with keys:
                - success (bool): Whether action succeeded
                - action (str): Action type performed
                - old_path (str): Original file path
                - new_path (str): New file path (if moved/renamed)
                - time_saved (float): Estimated time saved in minutes
                - message (str): Human-readable result message
        """
        # Log start of operation (MEDIUM #2 FIX - Audit trail)
        logger.info(f"Starting organization of {file_path} (user_approved={user_approved})")

        try:
            path = Path(file_path)

            # Validate file exists
            if not path.exists():
                logger.warning(f"File not found: {file_path}")
                return {
                    'success': False,
                    'action': 'none',
                    'old_path': file_path,
                    'new_path': None,
                    'time_saved': 0.0,
                    'message': 'File not found'
                }

            # Check folder policy allow_move (CRITICAL FIX #1)
            if folder_policy is None:
                folder_policy = self.config.get_folder_policy(file_path)

            if folder_policy and folder_policy.get('allow_move') is False:
                logger.info(f"Operation blocked by folder policy: {file_path}")
                return {
                    'success': False,
                    'action': 'blocked',
                    'old_path': file_path,
                    'new_path': None,
                    'time_saved': 0.0,
                    'message': 'Operation blocked: folder policy disallows moves'
                }

            # Check against configured blacklist paths
            try:
                blacklist = getattr(self.config, 'path_blacklist', []) or []
                resolved = str(path.resolve())
                for b in blacklist:
                    try:
                        b_res = str(Path(b).expanduser().resolve())
                        if os.path.commonpath([resolved, b_res]) == b_res:
                            return {
                                'success': False,
                                'action': 'blocked',
                                'old_path': file_path,
                                'new_path': None,
                                'time_saved': 0.0,
                                'message': f'Operation blocked: path is blacklisted ({b_res})'
                            }
                    except Exception:
                        if resolved.lower().startswith(str(Path(b).expanduser().resolve()).lower()):
                            return {
                                'success': False,
                                'action': 'blocked',
                                'old_path': file_path,
                                'new_path': None,
                                'time_saved': 0.0,
                                'message': f'Operation blocked: path is blacklisted ({b})'
                            }
            except Exception:
                pass

            # Determine action based on classification
            suggested_path = classification.get('suggested_path')
            suggested_rename = classification.get('rename')

            # Build new path with path traversal validation
            if suggested_path:
                try:
                    new_path = self._build_destination_path(path, suggested_path, suggested_rename)
                    action_type = 'move'
                except ValueError as e:
                    # Path validation failed (MEDIUM #3 - Security)
                    return {
                        'success': False,
                        'action': 'blocked',
                        'old_path': file_path,
                        'new_path': None,
                        'time_saved': 0.0,
                        'message': f'Security: {str(e)}'
                    }
            elif suggested_rename:
                new_path = path.parent / suggested_rename
                action_type = 'rename'
            else:
                return {
                    'success': False,
                    'action': 'none',
                    'old_path': file_path,
                    'new_path': None,
                    'time_saved': 0.0,
                    'message': 'No action suggested'
                }

            # Perform the action
            if self.dry_run:
                result = self._dry_run_action(path, new_path, action_type)
            else:
                result = self._perform_action(path, new_path, action_type)

            # Log action to database and file system (MEDIUM #2 FIX)
            if result['success']:
                time_saved = self.config.time_estimates.get(action_type, 0.3)

                logger.info(f"Successfully {action_type}d: {path} -> {new_path}")

                self.db_manager.log_action(
                    filename=path.name,
                    old_path=str(path),
                    new_path=str(new_path) if new_path else None,
                    operation=action_type,
                    time_saved=time_saved,
                    category=classification.get('category'),
                    ai_suggested=classification.get('method') == 'ai',
                    user_approved=user_approved
                )

                result['time_saved'] = time_saved

                # Add to undo history
                self._add_to_undo_history({
                    'action': action_type,
                    'old_path': str(path),
                    'new_path': str(new_path),
                    'timestamp': datetime.now().isoformat()
                })
            else:
                logger.warning(f"Action failed for {file_path}: {result.get('message', 'Unknown reason')}")

            return result

        except Exception as e:
            logger.error(f"Failed to organize {file_path}: {str(e)}", exc_info=True)
            return {
                'success': False,
                'action': 'error',
                'old_path': file_path,
                'new_path': None,
                'time_saved': 0.0,
                'message': f'Error: {str(e)}'
            }

    def _validate_path_safety(self, suggested_path: str, base_dir: Path) -> tuple:
        """
        Ensure path doesn't escape base_destination (MEDIUM #3 FIX - Security).

        Args:
            suggested_path (str): Path to validate
            base_dir (Path): Base directory that path should stay within

        Returns:
            tuple: (is_safe: bool, error_message: str)
        """
        if not suggested_path:
            return True, ""

        # Check for path traversal patterns
        if ".." in suggested_path:
            return False, "Path contains '..' (path traversal attempt)"

        # Absolute paths could bypass base_destination
        if os.path.isabs(suggested_path):
            return False, "Absolute paths not allowed for security"

        try:
            # Verify resolved path stays within base_destination
            test_path = (base_dir / suggested_path).resolve()
            test_path.relative_to(base_dir)
            return True, ""
        except (ValueError, OSError) as e:
            return False, f"Path escapes base directory: {str(e)}"

    def _build_destination_path(self, source_path: Path, suggested_path: str,
                                suggested_rename: Optional[str] = None) -> Path:
        """
        Build complete destination path for file with path traversal protection.

        Args:
            source_path (Path): Current file path
            suggested_path (str): Suggested destination directory
            suggested_rename (str, optional): Suggested new filename

        Returns:
            Path: Complete destination path

        Raises:
            ValueError: If path validation fails (path traversal attempt)
        """
        # Use configured base destination (CRITICAL FIX #2)
        try:
            base_dir = Path(self.config.base_destination).expanduser().resolve()
        except (AttributeError, OSError):
            base_dir = Path.home()  # Fallback only on error

        # Validate path safety (MEDIUM #3 FIX - Security)
        is_safe, error_msg = self._validate_path_safety(suggested_path, base_dir)
        if not is_safe:
            raise ValueError(f"Path validation failed: {error_msg}")

        # Handle absolute vs relative suggested paths
        suggested_obj = Path(suggested_path)
        if suggested_obj.is_absolute():
            # This should already be blocked by _validate_path_safety, but double-check
            raise ValueError("Absolute paths not allowed")
        else:
            # Remove any leading slashes to avoid accidental absolute joining
            dest_dir = base_dir / Path(suggested_path.lstrip('/'))

        # Determine filename
        filename = suggested_rename if suggested_rename else source_path.name

        # Handle filename conflicts
        dest_path = dest_dir / filename

        if dest_path.exists() and dest_path != source_path:
            # Add counter to filename
            stem = dest_path.stem
            suffix = dest_path.suffix
            counter = 1

            while dest_path.exists():
                dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        return dest_path

    def _perform_action(self, source: Path, destination: Path, action: str) -> Dict[str, Any]:
        """
        Actually perform file operation with race condition protection.

        Args:
            source (Path): Source file path
            destination (Path): Destination file path
            action (str): Action type ('move', 'rename', etc.)

        Returns:
            Dict: Result information
        """
        try:
            # Re-check file exists just before operation (CRITICAL FIX #3)
            if not source.exists():
                return {
                    'success': False,
                    'action': action,
                    'old_path': str(source),
                    'new_path': str(destination),
                    'message': f'File no longer exists at {source}'
                }

            # Check if file is locked/in use (CRITICAL FIX #3)
            try:
                with open(source, 'rb+') as f:
                    pass
            except (IOError, PermissionError) as e:
                return {
                    'success': False,
                    'action': action,
                    'old_path': str(source),
                    'new_path': str(destination),
                    'message': f'File is locked or in use: {str(e)}'
                }

            # Ensure destination directory exists
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Perform move/rename
            shutil.move(str(source), str(destination))

            return {
                'success': True,
                'action': action,
                'old_path': str(source),
                'new_path': str(destination),
                'message': f'Successfully {action}d file to {destination}'
            }

        except Exception as e:
            return {
                'success': False,
                'action': action,
                'old_path': str(source),
                'new_path': str(destination),
                'message': f'Failed to {action} file: {str(e)}'
            }

    def _dry_run_action(self, source: Path, destination: Path, action: str) -> Dict[str, Any]:
        """
        Simulate file operation without actually performing it.

        Args:
            source (Path): Source file path
            destination (Path): Destination file path
            action (str): Action type

        Returns:
            Dict: Simulated result
        """
        return {
            'success': True,
            'action': f'{action}_dry_run',
            'old_path': str(source),
            'new_path': str(destination),
            'message': f'[DRY RUN] Would {action} file to {destination}'
        }

    def delete_file(self, file_path: str, reason: str = "User requested") -> Dict[str, Any]:
        """
        Delete a file.

        Args:
            file_path (str): Path to file to delete
            reason (str): Reason for deletion

        Returns:
            Dict: Result information
        """
        try:
            path = Path(file_path)

            if not path.exists():
                return {
                    'success': False,
                    'action': 'delete',
                    'message': 'File not found'
                }

            if self.dry_run:
                message = f'[DRY RUN] Would delete {path}'
            else:
                path.unlink()
                message = f'Deleted {path}'

                # Log deletion
                time_saved = self.config.time_estimates.get('delete', 0.2)
                self.db_manager.log_action(
                    filename=path.name,
                    old_path=str(path),
                    new_path=None,
                    operation='delete',
                    time_saved=time_saved,
                    user_approved=True
                )

            return {
                'success': True,
                'action': 'delete',
                'old_path': str(path),
                'message': message
            }

        except Exception as e:
            return {
                'success': False,
                'action': 'delete',
                'message': f'Error deleting file: {str(e)}'
            }

    def archive_file(self, file_path: str, archive_dir: str = None) -> Dict[str, Any]:
        """
        Archive a file to archive directory.

        Args:
            file_path (str): Path to file to archive
            archive_dir (str, optional): Archive directory path

        Returns:
            Dict: Result information
        """
        try:
            path = Path(file_path)

            if not path.exists():
                return {
                    'success': False,
                    'action': 'archive',
                    'message': 'File not found'
                }

            # Default archive location
            if archive_dir is None:
                archive_dir = Path.home() / "Archive" / datetime.now().strftime("%Y/%m")
            else:
                archive_dir = Path(archive_dir)

            # Build destination
            dest_path = archive_dir / path.name

            # Handle conflicts
            if dest_path.exists():
                counter = 1
                while dest_path.exists():
                    dest_path = archive_dir / f"{path.stem}_{counter}{path.suffix}"
                    counter += 1

            if self.dry_run:
                message = f'[DRY RUN] Would archive to {dest_path}'
            else:
                # Create archive directory
                archive_dir.mkdir(parents=True, exist_ok=True)

                # Move to archive
                shutil.move(str(path), str(dest_path))
                message = f'Archived to {dest_path}'

                # Log action
                time_saved = self.config.time_estimates.get('archive', 0.4)
                self.db_manager.log_action(
                    filename=path.name,
                    old_path=str(path),
                    new_path=str(dest_path),
                    operation='archive',
                    time_saved=time_saved,
                    user_approved=True
                )

            return {
                'success': True,
                'action': 'archive',
                'old_path': str(path),
                'new_path': str(dest_path),
                'message': message
            }

        except Exception as e:
            return {
                'success': False,
                'action': 'archive',
                'message': f'Error archiving file: {str(e)}'
            }

    def undo_last_action(self) -> Dict[str, Any]:
        """
        Undo the last file operation.

        Returns:
            Dict: Result of undo operation
        """
        if not self.undo_history:
            return {
                'success': False,
                'message': 'No actions to undo'
            }

        # Get last action from database
        last_action = self.db_manager.undo_last_action()

        if not last_action:
            return {
                'success': False,
                'message': 'No undoable actions in database'
            }

        try:
            old_path = Path(last_action['old_path'])
            new_path = Path(last_action['new_path']) if last_action['new_path'] else None

            if not new_path or not new_path.exists():
                return {
                    'success': False,
                    'message': 'Cannot undo: destination file not found'
                }

            if self.dry_run:
                message = f'[DRY RUN] Would undo: move {new_path} back to {old_path}'
            else:
                # Ensure original directory exists
                old_path.parent.mkdir(parents=True, exist_ok=True)

                # Move back
                shutil.move(str(new_path), str(old_path))
                message = f'Undone: restored {old_path}'

            return {
                'success': True,
                'action': 'undo',
                'message': message
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Error undoing action: {str(e)}'
            }

    def _add_to_undo_history(self, action: Dict[str, Any]):
        """
        Add action to undo history.

        Args:
            action (Dict): Action details
        """
        self.undo_history.append(action)

        # Keep only recent history
        if len(self.undo_history) > self.max_undo_history:
            self.undo_history = self.undo_history[-self.max_undo_history:]

    def set_dry_run(self, enabled: bool):
        """
        Enable or disable dry run mode.

        Args:
            enabled (bool): True to enable dry run, False to disable
        """
        self.dry_run = enabled

    def get_stats(self) -> Dict[str, Any]:
        """
        Get action statistics from database.

        Returns:
            Dict: Statistics including total actions, time saved, etc.
        """
        return self.db_manager.get_stats('all')


if __name__ == "__main__":
    # Test action manager
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config import get_config
    from core.db_manager import DatabaseManager

    config = get_config()
    db = DatabaseManager()

    # Enable dry run for testing
    action_mgr = ActionManager(config, db, dry_run=True)

    # Test classification result
    test_classification = {
        'category': 'Documents',
        'suggested_path': 'Documents/Test/',
        'rename': None,
        'method': 'rule-based'
    }

    # Test action
    result = action_mgr.execute(
        file_path=__file__,  # Use this file as test
        classification=test_classification,
        user_approved=False
    )

    print(f"Action result: {json.dumps(result, indent=2)}")
