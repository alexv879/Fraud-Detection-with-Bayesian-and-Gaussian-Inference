"""
Duplicate File Finder Module

Copyright (c) 2025 Alexandru Emanuel Vasile. All rights reserved.
Proprietary Software - 200-Key Limited Release License

This module identifies duplicate files using content-based hashing.
It can detect exact duplicates and provide cleanup recommendations.

NOTICE: This software is proprietary and confidential.
See LICENSE.txt for full terms and conditions.

Author: Alexandru Emanuel Vasile
License: Proprietary (200-key limited release)
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import os


class DuplicateFinder:
    """
    Finds duplicate files based on content hashing.

    Attributes:
        config: Configuration object
        db_manager: Database manager for storing duplicate information
        hash_algorithm (str): Hash algorithm to use ('sha1', 'md5', 'sha256')
        min_file_size (int): Minimum file size to consider (bytes)
        file_hashes (Dict): Cache of file hash calculations
    """

    def __init__(self, config, db_manager, hash_algorithm: str = None, min_file_size: int = 1024):
        """
        Initialize duplicate finder.

        Args:
            config: Configuration object
            db_manager: Database manager instance
            hash_algorithm (str, optional): Hash algorithm ('sha1', 'md5', 'sha256')
            min_file_size (int): Minimum file size to check in bytes (default 1KB)
        """
        self.config = config
        self.db_manager = db_manager
        self.hash_algorithm = hash_algorithm or config.hash_algorithm or 'sha1'
        self.min_file_size = min_file_size
        self.file_hashes: Dict[str, str] = {}  # path -> hash cache

    def calculate_hash(self, file_path: str, chunk_size: int = 8192) -> Optional[str]:
        """
        Calculate hash of file content.

        Args:
            file_path (str): Path to file
            chunk_size (int): Size of chunks to read (bytes)

        Returns:
            str or None: Hex digest of file hash, or None if error
        """
        # Check cache first
        if file_path in self.file_hashes:
            return self.file_hashes[file_path]

        try:
            path = Path(file_path)

            # Check file size
            file_size = path.stat().st_size
            if file_size < self.min_file_size:
                return None

            # Choose hash algorithm
            if self.hash_algorithm == 'md5':
                hasher = hashlib.md5()
            elif self.hash_algorithm == 'sha256':
                hasher = hashlib.sha256()
            else:  # default to sha1
                hasher = hashlib.sha1()

            # Read file in chunks and update hash
            with open(path, 'rb') as f:
                while chunk := f.read(chunk_size):
                    hasher.update(chunk)

            file_hash = hasher.hexdigest()

            # Cache result
            self.file_hashes[file_path] = file_hash

            return file_hash

        except Exception as e:
            print(f"Error hashing {file_path}: {e}")
            return None

    def find_duplicates_in_directory(self, directory: str, recursive: bool = True) -> List[Dict[str, any]]:
        """
        Find all duplicate files in a directory.

        Args:
            directory (str): Directory path to scan
            recursive (bool): If True, scan subdirectories

        Returns:
            List[Dict]: List of duplicate groups, each containing:
                - hash (str): File content hash
                - paths (List[str]): List of duplicate file paths
                - size (int): File size in bytes
                - total_wasted_space (int): Space that could be freed
        """
        dir_path = Path(directory)

        if not dir_path.exists() or not dir_path.is_dir():
            print(f"Directory not found: {directory}")
            return []

        # Build hash -> paths mapping
        hash_map: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

        # Scan directory
        if recursive:
            files = dir_path.rglob('*')
        else:
            files = dir_path.glob('*')

        print(f"Scanning for duplicates in: {directory}")
        scanned_count = 0

        for file_path in files:
            if not file_path.is_file():
                continue

            # Calculate hash
            file_hash = self.calculate_hash(str(file_path))

            if file_hash:
                file_size = file_path.stat().st_size
                hash_map[file_hash].append((str(file_path), file_size))
                scanned_count += 1

                if scanned_count % 100 == 0:
                    print(f"Scanned {scanned_count} files...")

        print(f"Scan complete: {scanned_count} files processed")

        # Filter to only duplicates (hash appears more than once)
        duplicates = []

        for file_hash, file_list in hash_map.items():
            if len(file_list) > 1:
                paths = [path for path, size in file_list]
                size = file_list[0][1]  # All duplicates have same size
                wasted_space = size * (len(file_list) - 1)  # Space occupied by duplicates

                duplicate_group = {
                    'hash': file_hash,
                    'paths': paths,
                    'size': size,
                    'total_wasted_space': wasted_space,
                    'count': len(paths)
                }

                duplicates.append(duplicate_group)

                # Store in database
                for path in paths:
                    self.db_manager.add_duplicate(file_hash, path, size)

        # Sort by wasted space (descending)
        duplicates.sort(key=lambda x: x['total_wasted_space'], reverse=True)

        return duplicates

    def find_duplicates_in_multiple_directories(self, directories: List[str]) -> List[Dict[str, any]]:
        """
        Find duplicates across multiple directories.

        Args:
            directories (List[str]): List of directory paths

        Returns:
            List[Dict]: Duplicate groups
        """
        all_duplicates = []

        for directory in directories:
            duplicates = self.find_duplicates_in_directory(directory)
            all_duplicates.extend(duplicates)

        return all_duplicates

    def get_duplicate_summary(self, duplicates: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Generate summary statistics for duplicates.

        Args:
            duplicates (List[Dict]): List of duplicate groups

        Returns:
            Dict: Summary statistics
        """
        total_files = sum(group['count'] for group in duplicates)
        total_duplicate_files = sum(group['count'] - 1 for group in duplicates)
        total_wasted_space = sum(group['total_wasted_space'] for group in duplicates)

        return {
            'total_duplicate_groups': len(duplicates),
            'total_duplicate_files': total_duplicate_files,
            'total_files_including_originals': total_files,
            'total_wasted_space_bytes': total_wasted_space,
            'total_wasted_space_mb': round(total_wasted_space / (1024 * 1024), 2),
            'total_wasted_space_gb': round(total_wasted_space / (1024 * 1024 * 1024), 2)
        }

    def suggest_duplicates_to_keep(self, duplicate_group: Dict[str, any]) -> Dict[str, any]:
        """
        Suggest which file to keep and which to delete from duplicate group.

        Strategy:
        1. Keep file with shortest path (likely more organized)
        2. Keep file with most descriptive name
        3. Keep oldest file (likely the original)

        Args:
            duplicate_group (Dict): Single duplicate group

        Returns:
            Dict: Suggestions with keys:
                - keep (str): Path to keep
                - delete (List[str]): Paths to delete
                - reason (str): Explanation
        """
        paths = duplicate_group['paths']

        if len(paths) < 2:
            return {
                'keep': paths[0] if paths else None,
                'delete': [],
                'reason': 'No duplicates to remove'
            }

        # Strategy 1: Prefer shorter paths (better organized)
        sorted_by_length = sorted(paths, key=lambda p: len(p))

        # Strategy 2: Prefer paths with meaningful names (not temp/download folders)
        temp_keywords = ['temp', 'tmp', 'download', 'cache', 'trash']

        def is_temp_location(path: str) -> bool:
            path_lower = path.lower()
            return any(keyword in path_lower for keyword in temp_keywords)

        non_temp_paths = [p for p in paths if not is_temp_location(p)]

        if non_temp_paths:
            # Keep the first non-temp path with shortest length
            keep_path = sorted(non_temp_paths, key=lambda p: len(p))[0]
            reason = "Keeping file in more permanent location"
        else:
            # All are in temp locations, just keep shortest path
            keep_path = sorted_by_length[0]
            reason = "Keeping file with shortest path"

        delete_paths = [p for p in paths if p != keep_path]

        return {
            'keep': keep_path,
            'delete': delete_paths,
            'reason': reason
        }

    def find_temp_and_junk_files(self, directory: str) -> List[str]:
        """
        Find temporary and junk files that can be safely deleted.

        Args:
            directory (str): Directory to scan

        Returns:
            List[str]: List of temp/junk file paths
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            return []

        junk_patterns = [
            '*.tmp', '*.temp', '*.cache',
            '*.crdownload', '*.part', '*.partial',
            '*.download', '.DS_Store', 'Thumbs.db',
            'desktop.ini', '*.bak', '*~'
        ]

        junk_files = []

        for pattern in junk_patterns:
            junk_files.extend([str(p) for p in dir_path.rglob(pattern)])

        return junk_files

    def cleanup_duplicates(self, duplicate_group: Dict[str, any], dry_run: bool = True) -> Dict[str, any]:
        """
        Clean up duplicate files (delete all but one).

        Args:
            duplicate_group (Dict): Duplicate group to clean
            dry_run (bool): If True, don't actually delete files

        Returns:
            Dict: Cleanup result
        """
        suggestion = self.suggest_duplicates_to_keep(duplicate_group)

        deleted_count = 0
        errors = []

        for file_path in suggestion['delete']:
            try:
                if not dry_run:
                    Path(file_path).unlink()
                    self.db_manager.remove_duplicate_entry(file_path)

                deleted_count += 1
            except Exception as e:
                errors.append(f"Error deleting {file_path}: {e}")

        return {
            'success': len(errors) == 0,
            'kept': suggestion['keep'],
            'deleted_count': deleted_count,
            'errors': errors,
            'dry_run': dry_run,
            'space_freed': duplicate_group['size'] * deleted_count
        }


def find_duplicates(directory: str, config, db_manager) -> List[Dict[str, any]]:
    """
    Convenience function to find duplicates in a directory.

    Args:
        directory (str): Directory to scan
        config: Configuration object
        db_manager: Database manager

    Returns:
        List[Dict]: Duplicate groups
    """
    finder = DuplicateFinder(config, db_manager)
    return finder.find_duplicates_in_directory(directory)


if __name__ == "__main__":
    # Test duplicate finder
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config import get_config
    from core.db_manager import DatabaseManager

    config = get_config()
    db = DatabaseManager()

    finder = DuplicateFinder(config, db)

    # Test with current directory
    test_dir = Path.home() / "Downloads"

    if test_dir.exists():
        print(f"Scanning for duplicates in: {test_dir}")
        duplicates = finder.find_duplicates_in_directory(str(test_dir), recursive=False)

        if duplicates:
            print(f"\nFound {len(duplicates)} duplicate groups:")

            summary = finder.get_duplicate_summary(duplicates)
            print(f"\nSummary:")
            print(f"  Total duplicate files: {summary['total_duplicate_files']}")
            print(f"  Wasted space: {summary['total_wasted_space_mb']} MB")

            # Show first few duplicates
            for i, group in enumerate(duplicates[:3]):
                print(f"\nGroup {i+1}:")
                print(f"  Size: {group['size']} bytes")
                print(f"  Count: {group['count']} files")
                for path in group['paths']:
                    print(f"    - {path}")
        else:
            print("No duplicates found!")
    else:
        print(f"Test directory not found: {test_dir}")
