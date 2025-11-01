"""
Database Management Module

Copyright (c) 2025 Alexandru Emanuel Vasile. All rights reserved.
Proprietary Software - 200-Key Limited Release License

This module handles all database operations for the Private AI File Organiser.
It uses SQLite for local storage of file logs, statistics, license info, and duplicate tracking.

Tables:
    - files_log: Records all file operations
    - duplicates: Tracks duplicate file hashes
    - license: Stores license validation status
    - stats: Aggregated statistics (daily, weekly, monthly)

NOTICE: This software is proprietary and confidential.
See LICENSE.txt for full terms and conditions.

Author: Alexandru Emanuel Vasile
License: Proprietary (200-key limited release)
"""

import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager


class DatabaseManager:
    """
    Manages SQLite database operations for file tracking, statistics, and license management.

    Attributes:
        db_path (Path): Path to SQLite database file
        connection (sqlite3.Connection): Active database connection
    """

    def __init__(self, db_path: str = None):
        """
        Initialize database manager and create tables if they don't exist.

        Args:
            db_path (str, optional): Path to database file. Defaults to data/database/organiser.db
        """
        if db_path is None:
            # Default to data/database/organiser.db in project root
            self.db_path = Path(__file__).parent.parent.parent / "data" / "database" / "organiser.db"
        else:
            self.db_path = Path(db_path)

        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.connection: Optional[sqlite3.Connection] = None
        self._initialize_database()

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.

        Yields:
            sqlite3.Connection: Database connection

        Example:
            >>> with db.get_connection() as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT * FROM files_log")
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _initialize_database(self) -> None:
        """
        Create database tables if they don't exist.
        Called automatically during initialization.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Files log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS files_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    old_path TEXT NOT NULL,
                    new_path TEXT,
                    operation TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    time_saved REAL DEFAULT 0.0,
                    category TEXT,
                    ai_suggested BOOLEAN DEFAULT 0,
                    user_approved BOOLEAN DEFAULT 0,
                    raw_response TEXT,
                    model_name TEXT,
                    prompt_hash TEXT
                )
            """)

            # Migration: Add new columns if they don't exist
            try:
                cursor.execute("ALTER TABLE files_log ADD COLUMN raw_response TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

            try:
                cursor.execute("ALTER TABLE files_log ADD COLUMN model_name TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

            try:
                cursor.execute("ALTER TABLE files_log ADD COLUMN prompt_hash TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Duplicates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS duplicates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    discovered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(file_hash, file_path)
                )
            """)

            # License table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS license (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    license_key TEXT UNIQUE NOT NULL,
                    activation_date DATETIME,
                    expiry_date DATETIME,
                    status TEXT DEFAULT 'unused',
                    last_verified DATETIME
                )
            """)

            # Statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stat_date DATE UNIQUE NOT NULL,
                    files_organised INTEGER DEFAULT 0,
                    time_saved_minutes REAL DEFAULT 0.0,
                    duplicates_removed INTEGER DEFAULT 0,
                    ai_classifications INTEGER DEFAULT 0
                )
            """)

            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_timestamp ON files_log(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_duplicates_hash ON duplicates(file_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_date ON stats(stat_date)")

            conn.commit()

    # ==================== File Log Operations ====================

    def log_action(self, filename: str, old_path: str, new_path: Optional[str],
                   operation: str, time_saved: float = 0.0, category: Optional[str] = None,
                   ai_suggested: bool = False, user_approved: bool = False,
                   raw_response: Optional[str] = None, model_name: Optional[str] = None,
                   prompt_hash: Optional[str] = None) -> int:
        """
        Log a file operation to the database with atomic transaction support.

        Args:
            filename (str): Name of the file
            old_path (str): Original file path
            new_path (str, optional): New file path (if moved/renamed)
            operation (str): Type of operation (move, rename, delete, archive)
            time_saved (float): Estimated time saved in minutes
            category (str, optional): File category
            ai_suggested (bool): Whether AI suggested this action
            user_approved (bool): Whether user approved this action
            raw_response (str, optional): Raw LLM response for debugging
            model_name (str, optional): Name of the model used
            prompt_hash (str, optional): Hash of the prompt for traceability

        Returns:
            int: ID of the inserted log entry
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                # Start explicit transaction (HIGH #5 FIX)
                cursor.execute("BEGIN IMMEDIATE")

                # Insert log entry
                cursor.execute("""
                    INSERT INTO files_log
                    (filename, old_path, new_path, operation, time_saved, category, ai_suggested, user_approved,
                     raw_response, model_name, prompt_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (filename, old_path, new_path, operation, time_saved, category, ai_suggested, user_approved,
                      raw_response, model_name, prompt_hash))

                log_id = cursor.lastrowid

                # Update daily stats
                today = datetime.now().date()
                cursor.execute("""
                    INSERT INTO stats (stat_date, files_organised, time_saved_minutes, ai_classifications)
                    VALUES (?, 1, ?, ?)
                    ON CONFLICT(stat_date) DO UPDATE SET
                        files_organised = files_organised + 1,
                        time_saved_minutes = time_saved_minutes + ?,
                        ai_classifications = ai_classifications + ?
                """, (today, time_saved, 1 if ai_suggested else 0, time_saved, 1 if ai_suggested else 0))

                # Commit transaction (HIGH #5 FIX)
                conn.commit()
                return log_id

            except Exception as e:
                # Rollback on any error (HIGH #5 FIX)
                conn.rollback()
                raise e

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve recent file operation logs.

        Args:
            limit (int): Maximum number of logs to retrieve

        Returns:
            List[Dict]: List of log entries as dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM files_log
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def search_logs(self, query: str = None, category: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search the files_log table by filename, old_path, new_path or category using simple LIKE queries.

        Args:
            query (str, optional): Substring to search for in filename/paths
            category (str, optional): Category to filter by
            limit (int): Max results

        Returns:
            List[Dict]: Matching log entries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            where_clauses = []
            params: List[Any] = []

            if query:
                # Escape SQL LIKE wildcards (HIGH #3 FIX)
                escaped_query = query.replace('%', '\\%').replace('_', '\\_')
                like = f"%{escaped_query}%"
                where_clauses.append(
                    "(filename LIKE ? ESCAPE '\\' OR old_path LIKE ? ESCAPE '\\' OR new_path LIKE ? ESCAPE '\\')"
                )
                params.extend([like, like, like])

            if category:
                where_clauses.append("category = ?")
                params.append(category)

            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

            sql = f"""
                SELECT * FROM files_log
                WHERE {where_sql}
                ORDER BY timestamp DESC
                LIMIT ?
            """

            params.append(limit)

            cursor.execute(sql, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

    def undo_last_action(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the last action for undo functionality.

        Returns:
            Dict or None: Last action details or None if no actions exist
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM files_log
                WHERE operation IN ('move', 'rename')
                ORDER BY timestamp DESC
                LIMIT 1
            """)

            row = cursor.fetchone()
            return dict(row) if row else None

    # ==================== Duplicate Operations ====================

    def add_duplicate(self, file_hash: str, file_path: str, file_size: int) -> bool:
        """
        Add a duplicate file entry.

        Args:
            file_hash (str): Hash of the file content
            file_path (str): Path to the file
            file_size (int): File size in bytes

        Returns:
            bool: True if added successfully, False if already exists
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO duplicates (file_hash, file_path, file_size)
                    VALUES (?, ?, ?)
                """, (file_hash, file_path, file_size))
                return True
        except sqlite3.IntegrityError:
            return False  # Duplicate entry already exists

    def get_duplicates(self) -> List[Dict[str, List[str]]]:
        """
        Get all sets of duplicate files grouped by hash.

        Returns:
            List[Dict]: List of duplicate groups, each containing file paths
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT file_hash, GROUP_CONCAT(file_path) as paths, file_size
                FROM duplicates
                GROUP BY file_hash
                HAVING COUNT(*) > 1
            """)

            duplicates = []
            for row in cursor.fetchall():
                duplicates.append({
                    'hash': row['file_hash'],
                    'paths': row['paths'].split(','),
                    'size': row['file_size']
                })

            return duplicates

    def remove_duplicate_entry(self, file_path: str) -> bool:
        """
        Remove a duplicate entry from tracking.

        Args:
            file_path (str): Path to the file to remove

        Returns:
            bool: True if removed successfully
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM duplicates WHERE file_path = ?", (file_path,))
            return cursor.rowcount > 0

    # ==================== License Operations ====================

    def store_license(self, license_key: str, expiry_date: datetime, status: str = 'active') -> bool:
        """
        Store or update license information.

        Args:
            license_key (str): The license key
            expiry_date (datetime): License expiration date
            status (str): License status (unused, active, expired)

        Returns:
            bool: True if stored successfully
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO license (license_key, activation_date, expiry_date, status, last_verified)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(license_key) DO UPDATE SET
                    expiry_date = ?,
                    status = ?,
                    last_verified = ?
            """, (license_key, datetime.now(), expiry_date, status, datetime.now(),
                  expiry_date, status, datetime.now()))

            return True

    def get_license_status(self) -> Optional[Dict[str, Any]]:
        """
        Get current license status.

        Returns:
            Dict or None: License information or None if no license exists
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM license
                ORDER BY activation_date DESC
                LIMIT 1
            """)

            row = cursor.fetchone()
            if row:
                license_info = dict(row)
                # Check if expired
                if license_info['expiry_date']:
                    expiry = datetime.fromisoformat(license_info['expiry_date'])
                    if expiry < datetime.now():
                        license_info['status'] = 'expired'
                return license_info

            return None

    def is_license_valid(self) -> bool:
        """
        Check if current license is valid and not expired.

        Returns:
            bool: True if license is valid and active
        """
        license_info = self.get_license_status()
        if not license_info:
            return False

        if license_info['status'] != 'active':
            return False

        if license_info['expiry_date']:
            expiry = datetime.fromisoformat(license_info['expiry_date'])
            return expiry > datetime.now()

        return False

    # ==================== Statistics Operations ====================

    def get_stats(self, period: str = 'all') -> Dict[str, Any]:
        """
        Get aggregated statistics for a given period.

        Args:
            period (str): Time period ('today', 'week', 'month', 'all')

        Returns:
            Dict: Statistics including files organised, time saved, etc.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if period == 'today':
                date_filter = "stat_date = date('now')"
            elif period == 'week':
                date_filter = "stat_date >= date('now', '-7 days')"
            elif period == 'month':
                date_filter = "stat_date >= date('now', '-30 days')"
            else:  # 'all'
                date_filter = "1=1"

            cursor.execute(f"""
                SELECT
                    SUM(files_organised) as total_files,
                    SUM(time_saved_minutes) as total_time_saved,
                    SUM(duplicates_removed) as total_duplicates,
                    SUM(ai_classifications) as total_ai_classifications
                FROM stats
                WHERE {date_filter}
            """)

            row = cursor.fetchone()
            return {
                'files_organised': row['total_files'] or 0,
                'time_saved_minutes': row['total_time_saved'] or 0.0,
                'time_saved_hours': round((row['total_time_saved'] or 0.0) / 60, 2),
                'duplicates_removed': row['total_duplicates'] or 0,
                'ai_classifications': row['total_ai_classifications'] or 0,
                'period': period
            }

    def update_duplicate_stats(self, count: int) -> None:
        """
        Update duplicate removal statistics for today.

        Args:
            count (int): Number of duplicates removed
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            today = datetime.now().date()
            cursor.execute("""
                INSERT INTO stats (stat_date, duplicates_removed)
                VALUES (?, ?)
                ON CONFLICT(stat_date) DO UPDATE SET
                    duplicates_removed = duplicates_removed + ?
            """, (today, count, count))


if __name__ == "__main__":
    # Test database operations
    db = DatabaseManager()
    print("Database initialized successfully!")

    # Test logging an action
    log_id = db.log_action(
        filename="test_file.pdf",
        old_path="/home/user/Downloads/test_file.pdf",
        new_path="/home/user/Documents/PDFs/test_file.pdf",
        operation="move",
        time_saved=0.5,
        category="Documents",
        ai_suggested=True,
        user_approved=True
    )
    print(f"Logged action with ID: {log_id}")

    # Test retrieving stats
    stats = db.get_stats('today')
    print(f"Today's stats: {stats}")

    # Test license check
    is_valid = db.is_license_valid()
    print(f"License valid: {is_valid}")
