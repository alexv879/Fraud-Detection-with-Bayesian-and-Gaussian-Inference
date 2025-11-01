"""
Private AI File Organiser - Main Entry Point

Copyright (c) 2025 Alexandru Emanuel Vasile. All rights reserved.
Proprietary Software - 200-Key Limited Release License

This is the main application entry point that orchestrates all components.
It provides CLI interface for running the organiser in different modes.

NOTICE: This software is proprietary and confidential. Unauthorized copying,
modification, distribution, or use is strictly prohibited.
See LICENSE.txt for full terms and conditions.

Version: 1.0.0
Author: Alexandru Emanuel Vasile
License: Proprietary (200-key limited release)
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from core.db_manager import DatabaseManager
from core.classifier import FileClassifier
from core.actions import ActionManager
from core.watcher import FolderWatcher
from core.duplicates import DuplicateFinder
from ai.ollama_client import OllamaClient
from license.validator import LicenseValidator
from ui.dashboard import run_dashboard


class FileOrganiser:
    """
    Main application orchestrator.

    This class coordinates all components of the file organiser.

    Attributes:
        config: Configuration object
        db: Database manager
        ollama: Ollama AI client
        classifier: File classifier
        action_manager: Action manager
        watcher: Folder watcher
        duplicate_finder: Duplicate finder
        license_validator: License validator
    """

    def __init__(self):
        """Initialize the file organiser."""
        print("🤖 AI File Organiser - Initializing...")

        # Load configuration
        self.config = get_config()

        # Initialize database
        self.db = DatabaseManager()

        # Initialize license validator
        self.license_validator = LicenseValidator(self.config, self.db)

        # Check license
        if not self._check_license():
            print("\n❌ License validation failed. Please activate a license.")
            self._prompt_license_activation()
            return

        print("✅ License valid")

        # Initialize Ollama client
        print("Connecting to Ollama...")
        self.ollama = OllamaClient(
            base_url=self.config.ollama_base_url,
            model=self.config.ollama_model,
            timeout=self.config.get('ollama_timeout', 30)
        )

        if self.ollama.is_available():
            print(f"✅ Ollama connected (model: {self.config.ollama_model})")
        else:
            print(f"⚠️  Ollama not available - AI classification disabled")
            print(f"   Make sure Ollama is running on {self.config.ollama_base_url}")

        # Initialize classifier
        ollama_client = self.ollama if self.ollama.is_available() else None
        self.classifier = FileClassifier(self.config, ollama_client)

        # Initialize action manager
        self.action_manager = ActionManager(self.config, self.db)

        # Initialize duplicate finder
        self.duplicate_finder = DuplicateFinder(self.config, self.db)

        # Initialize watcher (don't start yet)
        self.watcher = FolderWatcher(
            folders=self.config.watched_folders,
            callback=self._on_file_detected,
            config=self.config
        )

        print("✅ Initialization complete\n")

    def _check_license(self) -> bool:
        """
        Check if license is valid.

        Returns:
            bool: True if license is valid
        """
        status = self.license_validator.check_license_status()
        return status['is_valid']

    def _prompt_license_activation(self):
        """Prompt user to activate license."""
        print("\n" + "="*50)
        print("LICENSE ACTIVATION REQUIRED")
        print("="*50)

        license_key = input("\nEnter your license key (XXXX-XXXX-XXXX-XXXX): ").strip()

        if license_key:
            result = self.license_validator.activate_license(license_key)
            print(f"\n{result['message']}")

            if result['success']:
                print("\n✅ You can now use the AI File Organiser!")
                # Re-initialize after successful activation
                self.__init__()
            else:
                print("\n❌ Activation failed. Exiting.")
                sys.exit(1)
        else:
            print("\n❌ No license key provided. Exiting.")
            sys.exit(1)

    def _on_file_detected(self, file_path: str):
        """
        Callback when watcher detects a new file.

        Args:
            file_path (str): Path to detected file
        """
        print(f"\n📁 New file detected: {Path(file_path).name}")

        # Classify file
        classification = self.classifier.classify(file_path)

        print(f"   Category: {classification['category']}")
        print(f"   Suggested: {classification.get('suggested_path', 'N/A')}")
        print(f"   Reason: {classification['reason']}")
        print(f"   Method: {classification['method']} ({classification['confidence']} confidence)")

        # Auto-execute if in auto mode
        if self.config.auto_mode:
            print(f"   🤖 Auto mode: executing action...")

            result = self.action_manager.execute(
                file_path=file_path,
                classification=classification,
                user_approved=False
            )

            if result['success']:
                print(f"   ✅ {result['message']}")
                print(f"   ⏱️  Time saved: {result.get('time_saved', 0):.1f} minutes")
            else:
                print(f"   ❌ {result['message']}")
        else:
            print(f"   ⏸️  Waiting for approval (use dashboard or CLI)")

    def start_watch_mode(self):
        """Start watching folders for new files."""
        print("👀 Starting watch mode...")
        print(f"Monitoring folders:")
        for folder in self.config.watched_folders:
            print(f"  - {folder}")

        print(f"\nAuto mode: {'ENABLED' if self.config.auto_mode else 'DISABLED'}")
        print(f"Dry run: {'ENABLED' if self.config.dry_run else 'DISABLED'}")
        print(f"AI classification: {'ENABLED' if self.config.enable_ai else 'DISABLED'}")

        print("\nPress Ctrl+C to stop\n")

        self.watcher.start(background=False)

        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping watcher...")
            self.watcher.stop()
            print("Goodbye! 👋")

    def scan_existing_files(self):
        """Scan existing files in watched folders."""
        print("🔍 Scanning existing files...")

        files = self.watcher.scan_existing_files(callback=self._on_file_detected)

        print(f"\n✅ Scan complete: {len(files)} files found")

    def find_duplicates(self):
        """Find and report duplicate files."""
        print("🔍 Scanning for duplicate files...")

        all_duplicates = []

        for folder in self.config.watched_folders:
            print(f"\nScanning: {folder}")
            duplicates = self.duplicate_finder.find_duplicates_in_directory(folder, recursive=True)
            all_duplicates.extend(duplicates)

        if not all_duplicates:
            print("\n✅ No duplicates found!")
            return

        # Show summary
        summary = self.duplicate_finder.get_duplicate_summary(all_duplicates)

        print(f"\n" + "="*50)
        print("DUPLICATE FILES SUMMARY")
        print("="*50)
        print(f"Duplicate groups: {summary['total_duplicate_groups']}")
        print(f"Duplicate files: {summary['total_duplicate_files']}")
        print(f"Wasted space: {summary['total_wasted_space_mb']:.2f} MB")
        print(f"              ({summary['total_wasted_space_gb']:.2f} GB)")

        print(f"\n" + "-"*50)
        print("TOP 10 DUPLICATE GROUPS")
        print("-"*50)

        for i, group in enumerate(all_duplicates[:10], 1):
            print(f"\n{i}. {group['count']} files ({group['size']} bytes each)")
            for path in group['paths']:
                print(f"   - {path}")

    def run_dashboard(self, host: str = "127.0.0.1", port: int = 5000):
        """
        Run the web dashboard.

        Args:
            host (str): Host to bind to
            port (int): Port to listen on
        """
        run_dashboard(host, port)

    def show_stats(self):
        """Show statistics."""
        stats = self.db.get_stats('all')

        print("\n" + "="*50)
        print("AI FILE ORGANISER - STATISTICS")
        print("="*50)
        print(f"Files organised: {stats['files_organised']}")
        print(f"Time saved: {stats['time_saved_hours']:.2f} hours")
        print(f"AI classifications: {stats['ai_classifications']}")
        print(f"Duplicates removed: {stats['duplicates_removed']}")

        # License info
        license_status = self.license_validator.check_license_status()
        print(f"\nLicense status: {license_status['status'].upper()}")
        if license_status.get('days_remaining'):
            print(f"Days remaining: {license_status['days_remaining']}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Private AI File Organiser - Your local-first file organization assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s dashboard           # Run web dashboard
  %(prog)s watch              # Watch folders for new files
  %(prog)s scan               # Scan existing files
  %(prog)s duplicates         # Find duplicate files
  %(prog)s stats              # Show statistics

For more information, visit: https://github.com/yourproject
        """
    )

    parser.add_argument(
        'command',
        choices=['dashboard', 'watch', 'scan', 'duplicates', 'stats', 'license'],
        help='Command to execute'
    )

    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Dashboard host (default: 127.0.0.1)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Dashboard port (default: 5000)'
    )

    parser.add_argument(
        '--activate',
        metavar='LICENSE_KEY',
        help='Activate license with key'
    )

    args = parser.parse_args()

    # Handle license activation
    if args.activate:
        organiser = FileOrganiser()
        result = organiser.license_validator.activate_license(args.activate)
        print(result['message'])
        return

    # Handle commands
    if args.command == 'dashboard':
        organiser = FileOrganiser()
        organiser.run_dashboard(args.host, args.port)

    elif args.command == 'watch':
        organiser = FileOrganiser()
        organiser.start_watch_mode()

    elif args.command == 'scan':
        organiser = FileOrganiser()
        organiser.scan_existing_files()

    elif args.command == 'duplicates':
        organiser = FileOrganiser()
        organiser.find_duplicates()

    elif args.command == 'stats':
        organiser = FileOrganiser()
        organiser.show_stats()

    elif args.command == 'license':
        organiser = FileOrganiser()
        status = organiser.license_validator.check_license_status()
        print(f"\n{status['message']}")


if __name__ == "__main__":
    main()
