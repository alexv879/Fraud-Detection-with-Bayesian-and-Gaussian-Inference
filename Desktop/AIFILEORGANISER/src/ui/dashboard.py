"""
FastAPI Dashboard Module

Copyright (c) 2025 Alexandru Emanuel Vasile. All rights reserved.
Proprietary Software - 200-Key Limited Release License

This module provides a web-based dashboard for the Private AI File Organiser.
The dashboard includes:
- File inbox (pending files)
- Statistics and time saved
- Settings and configuration
- License management

NOTICE: This software is proprietary and confidential.
See LICENSE.txt for full terms and conditions.

Author: Alexandru Emanuel Vasile
License: Proprietary (200-key limited release)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import defaultdict
from time import time
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from core.db_manager import DatabaseManager
from core.classifier import FileClassifier
from core.actions import ActionManager
from core.duplicates import DuplicateFinder
from core.watcher import FolderWatcher
from ai.ollama_client import OllamaClient
from license.validator import LicenseValidator


# Rate limiting (HIGH-5 FIX)
_rate_limit_cache = defaultdict(list)
_RATE_LIMIT_WINDOW = 60  # seconds
_MAX_REQUESTS_PER_WINDOW = 10  # max deep analyze requests per window


def _check_rate_limit(ip: str) -> bool:
    """
    Check if IP is within rate limit for deep analyze endpoint.

    Args:
        ip: Client IP address

    Returns:
        bool: True if within limit, False if rate limit exceeded
    """
    now = time()

    # Clean old entries outside the window
    _rate_limit_cache[ip] = [t for t in _rate_limit_cache[ip] if now - t < _RATE_LIMIT_WINDOW]

    # Check limit
    if len(_rate_limit_cache[ip]) >= _MAX_REQUESTS_PER_WINDOW:
        return False

    # Record this request
    _rate_limit_cache[ip].append(now)
    return True


# Pydantic models for API requests/responses
class FileActionRequest(BaseModel):
    file_path: str
    action: str  # 'approve', 'reject', 'custom'
    custom_path: Optional[str] = None


class LicenseActivationRequest(BaseModel):
    license_key: str


class SettingsUpdateRequest(BaseModel):
    auto_mode: Optional[bool] = None
    dry_run: Optional[bool] = None
    enable_ai: Optional[bool] = None


class DeepAnalyzeRequest(BaseModel):
    file_path: str


# Initialize FastAPI app
app = FastAPI(
    title="AI File Organiser Dashboard",
    description="Web dashboard for Private AI File Organiser",
    version="1.0.0"
)

# Global application state
class AppState:
    """Application state container."""
    def __init__(self):
        self.config = get_config()
        self.db = DatabaseManager()
        self.ollama = None
        self.classifier = None
        self.action_manager = None
        self.duplicate_finder = None
        self.watcher = None
        self.license_validator = None
        self.pending_files: List[Dict[str, Any]] = []

        self._initialize()

    def _initialize(self):
        """Initialize all components."""
        # Initialize Ollama client
        self.ollama = OllamaClient(
            base_url=self.config.ollama_base_url,
            model=self.config.ollama_model,
            timeout=self.config.get('ollama_timeout', 30)
        )

        # Initialize classifier
        ollama_client = self.ollama if self.ollama.is_available() else None
        self.classifier = FileClassifier(self.config, ollama_client)

        # Initialize action manager
        self.action_manager = ActionManager(self.config, self.db)

        # Initialize duplicate finder
        self.duplicate_finder = DuplicateFinder(self.config, self.db)

        # Initialize license validator
        self.license_validator = LicenseValidator(self.config, self.db)

        # Initialize watcher (but don't start yet)
        self.watcher = FolderWatcher(
            folders=self.config.watched_folders,
            callback=self.on_file_detected,
            config=self.config
        )

    def on_file_detected(self, file_path: str):
        """
        Callback when watcher detects a new file.

        Args:
            file_path (str): Path to detected file
        """
        # Classify the file
        classification = self.classifier.classify(file_path)

        # Add to pending files
        self.pending_files.append({
            'file_path': file_path,
            'classification': classification,
            'detected_at': Path(file_path).stat().st_mtime
        })

    def start_watcher(self):
        """Start the folder watcher."""
        if self.watcher and not self.watcher._running:
            self.watcher.start(background=True)

    def stop_watcher(self):
        """Stop the folder watcher."""
        if self.watcher and self.watcher._running:
            self.watcher.stop()


# Create global app state
state = AppState()


# ==================== HTML Templates ====================

def get_dashboard_html() -> str:
    """Generate dashboard HTML."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI File Organiser Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            color: #333;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        header h1 {
            font-size: 32px;
            margin-bottom: 10px;
        }

        header p {
            opacity: 0.9;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .stat-card h3 {
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .stat-card .value {
            font-size: 36px;
            font-weight: bold;
            color: #667eea;
        }

        .stat-card .label {
            font-size: 12px;
            color: #999;
            margin-top: 5px;
        }

        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }

        .tab {
            padding: 12px 24px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 16px;
            color: #666;
            transition: all 0.3s;
        }

        .tab.active {
            color: #667eea;
            border-bottom: 3px solid #667eea;
            margin-bottom: -2px;
        }

        .tab-content {
            display: none;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .tab-content.active {
            display: block;
        }

        .file-item {
            padding: 15px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin-bottom: 10px;
            transition: all 0.3s;
        }

        .file-item:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .file-name {
            font-weight: 600;
            margin-bottom: 5px;
        }

        .file-meta {
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
        }

        .action-buttons {
            display: flex;
            gap: 10px;
        }

        button {
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }

        .btn-primary {
            background: #667eea;
            color: white;
        }

        .btn-primary:hover {
            background: #5568d3;
        }

        .btn-secondary {
            background: #e0e0e0;
            color: #333;
        }

        .btn-secondary:hover {
            background: #d0d0d0;
        }

        .btn-danger {
            background: #f44336;
            color: white;
        }

        .btn-danger:hover {
            background: #d32f2f;
        }

        .license-status {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        .license-valid {
            background: #e8f5e9;
            border: 1px solid #4caf50;
            color: #2e7d32;
        }

        .license-invalid {
            background: #ffebee;
            border: 1px solid #f44336;
            color: #c62828;
        }

        input[type="text"] {
            padding: 10px;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            font-size: 14px;
            width: 100%;
            max-width: 400px;
        }

        .setting-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 15px 0;
            border-bottom: 1px solid #e0e0e0;
        }

        .setting-item:last-child {
            border-bottom: none;
        }

        .toggle-switch {
            position: relative;
            width: 50px;
            height: 24px;
        }

        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }

        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 24px;
        }

        .slider:before {
            position: absolute;
            content: "";
            height: 18px;
            width: 18px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }

        input:checked + .slider {
            background-color: #667eea;
        }

        input:checked + .slider:before {
            transform: translateX(26px);
        }

        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }

        .empty-state svg {
            width: 80px;
            height: 80px;
            margin-bottom: 20px;
            opacity: 0.3;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 AI File Organiser</h1>
            <p>Your private, local-first file organization assistant</p>
        </header>

        <div class="stats-grid" id="stats-grid">
            <!-- Stats will be loaded dynamically -->
        </div>

        <div class="tabs">
            <button class="tab active" onclick="switchTab('inbox')">📥 Inbox</button>
            <button class="tab" onclick="switchTab('history')">📊 History</button>
            <button class="tab" onclick="switchTab('duplicates')">🔍 Duplicates</button>
            <button class="tab" onclick="switchTab('search')">🔎 Search</button>
            <button class="tab" onclick="switchTab('settings')">⚙️ Settings</button>
            <button class="tab" onclick="switchTab('license')">🔐 License</button>
        </div>

        <div id="inbox" class="tab-content active">
            <h2>Pending Files</h2>
            <div id="pending-files">
                <!-- Files will be loaded dynamically -->
            </div>
        </div>

        <div id="history" class="tab-content">
            <h2>Recent Actions</h2>
            <div id="history-list">
                <!-- History will be loaded dynamically -->
            </div>
        </div>

        <div id="duplicates" class="tab-content">
            <h2>Duplicate Files</h2>
            <button class="btn-primary" onclick="scanDuplicates()">Scan for Duplicates</button>
            <div id="duplicates-list" style="margin-top: 20px;">
                <!-- Duplicates will be loaded dynamically -->
            </div>
        </div>

        <div id="search" class="tab-content">
            <h2>Find moved files</h2>
            <div style="margin-bottom: 12px; display:flex; gap:8px; align-items:center;">
                <input type="text" id="search-query" placeholder="Search by filename, folder, or text" style="flex:1;" />
                <input type="text" id="search-category" placeholder="Category (optional)" style="width:200px;" />
                <button class="btn-primary" onclick="searchFiles()">Search</button>
            </div>
            <div id="search-results">
                <!-- Search results will appear here -->
            </div>
        </div>

        <div id="settings" class="tab-content">
            <h2>Settings</h2>
            <div class="settings-container">
                <div class="setting-item">
                    <div>
                        <strong>Auto Mode</strong>
                        <p style="font-size: 14px; color: #666; margin-top: 5px;">Automatically organize files without approval</p>
                    </div>
                    <label class="toggle-switch">
                        <input type="checkbox" id="auto-mode-toggle" onchange="updateSetting('auto_mode', this.checked)">
                        <span class="slider"></span>
                    </label>
                </div>
                <div class="setting-item">
                    <div>
                        <strong>Dry Run Mode</strong>
                        <p style="font-size: 14px; color: #666; margin-top: 5px;">Simulate actions without actually moving files</p>
                    </div>
                    <label class="toggle-switch">
                        <input type="checkbox" id="dry-run-toggle" checked onchange="updateSetting('dry_run', this.checked)">
                        <span class="slider"></span>
                    </label>
                </div>
                <div class="setting-item">
                    <div>
                        <strong>AI Classification</strong>
                        <p style="font-size: 14px; color: #666; margin-top: 5px;">Use local AI for intelligent file classification</p>
                    </div>
                    <label class="toggle-switch">
                        <input type="checkbox" id="ai-toggle" checked onchange="updateSetting('enable_ai', this.checked)">
                        <span class="slider"></span>
                    </label>
                </div>
            </div>
        </div>

        <div id="license" class="tab-content">
            <h2>License Management</h2>
            <div id="license-status">
                <!-- License status will be loaded dynamically -->
            </div>
            <div style="margin-top: 20px;">
                <h3>Activate License</h3>
                <p style="color: #666; margin: 10px 0;">Enter your license key (format: XXXX-XXXX-XXXX-XXXX)</p>
                <input type="text" id="license-key-input" placeholder="ABCD-1234-EFGH-5678">
                <button class="btn-primary" onclick="activateLicense()" style="margin-left: 10px;">Activate</button>
            </div>
        </div>
    </div>

    <script>
        // Tab switching
        function switchTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });

            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');

            // Load data for specific tabs
            if (tabName === 'history') {
                loadHistory();
            } else if (tabName === 'license') {
                loadLicenseStatus();
            } else if (tabName === 'inbox') {
                loadPendingFiles();
            }
        }

        // Load stats
        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();

                document.getElementById('stats-grid').innerHTML = `
                    <div class="stat-card">
                        <h3>Files Organised</h3>
                        <div class="value">${stats.files_organised}</div>
                        <div class="label">Total files</div>
                    </div>
                    <div class="stat-card">
                        <h3>Time Saved</h3>
                        <div class="value">${stats.time_saved_hours}</div>
                        <div class="label">Hours</div>
                    </div>
                    <div class="stat-card">
                        <h3>AI Classifications</h3>
                        <div class="value">${stats.ai_classifications}</div>
                        <div class="label">Using AI</div>
                    </div>
                    <div class="stat-card">
                        <h3>Duplicates Removed</h3>
                        <div class="value">${stats.duplicates_removed}</div>
                        <div class="label">Files</div>
                    </div>
                `;
            } catch (error) {
                console.error('Error loading stats:', error);
            }
        }

        // Load pending files
        async function loadPendingFiles() {
            try {
                const response = await fetch('/api/pending-files');
                const files = await response.json();

                const container = document.getElementById('pending-files');

                if (files.length === 0) {
                    container.innerHTML = `
                        <div class="empty-state">
                            <p>No pending files</p>
                        </div>
                    `;
                    return;
                }

                container.innerHTML = files.map(file => `
                    <div class="file-item">
                        <div class="file-name">${file.filename}</div>
                        <div class="file-meta">
                            Category: ${file.classification.category} |
                            Suggested: ${file.classification.suggested_path || 'N/A'}
                        </div>
                        <div class="file-meta" style="font-size: 12px; font-style: italic;">
                            ${file.classification.reason}
                        </div>
                        <div class="action-buttons">
                            <button class="btn-primary" onclick="approveFile('${file.file_path}')">✓ Approve</button>
                            <button class="btn-secondary" onclick="rejectFile('${file.file_path}')">✗ Reject</button>
                            <button class="btn-secondary" onclick="deepAnalyze('${file.file_path}')">🔍 Deep Analyze</button>
                        </div>
                        <div id="deep-analysis-${btoa(file.file_path).replace(/=/g,'')}" style="display:none; margin-top:15px; padding:15px; background:#f9f9f9; border-radius:6px;"></div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error loading pending files:', error);
            }
        }

        // Load history
        async function loadHistory() {
            try {
                const response = await fetch('/api/history');
                const history = await response.json();

                const container = document.getElementById('history-list');

                if (history.length === 0) {
                    container.innerHTML = `
                        <div class="empty-state">
                            <p>No history yet</p>
                        </div>
                    `;
                    return;
                }

                container.innerHTML = history.map(item => `
                    <div class="file-item">
                        <div class="file-name">${item.filename}</div>
                        <div class="file-meta">
                            ${item.operation} | ${new Date(item.timestamp).toLocaleString()}
                        </div>
                        <div class="file-meta" style="font-size: 12px;">
                            ${item.old_path} → ${item.new_path || 'Deleted'}
                        </div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error loading history:', error);
            }
        }

        // Load license status
        async function loadLicenseStatus() {
            try {
                const response = await fetch('/api/license/status');
                const status = await response.json();

                const container = document.getElementById('license-status');
                const isValid = status.is_valid;

                container.innerHTML = `
                    <div class="license-status ${isValid ? 'license-valid' : 'license-invalid'}">
                        <strong>Status:</strong> ${status.status.toUpperCase()}<br>
                        ${status.message}
                        ${status.days_remaining ? `<br><strong>${status.days_remaining}</strong> days remaining` : ''}
                    </div>
                `;
            } catch (error) {
                console.error('Error loading license status:', error);
            }
        }

        // Approve file action
        async function approveFile(filePath) {
            try {
                const response = await fetch('/api/files/approve', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file_path: filePath })
                });

                const result = await response.json();
                alert(result.message);
                loadPendingFiles();
                loadStats();
            } catch (error) {
                console.error('Error approving file:', error);
            }
        }

        // Reject file action
        async function rejectFile(filePath) {
            try {
                const response = await fetch('/api/files/reject', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file_path: filePath })
                });

                const result = await response.json();
                loadPendingFiles();
            } catch (error) {
                console.error('Error rejecting file:', error);
            }
        }

        // Activate license
        async function activateLicense() {
            const key = document.getElementById('license-key-input').value;

            if (!key) {
                alert('Please enter a license key');
                return;
            }

            try {
                const response = await fetch('/api/license/activate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ license_key: key })
                });

                const result = await response.json();
                alert(result.message);
                loadLicenseStatus();
            } catch (error) {
                console.error('Error activating license:', error);
            }
        }

        // Update settings
        async function updateSetting(setting, value) {
            try {
                const response = await fetch('/api/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ [setting]: value })
                });

                const result = await response.json();
                console.log('Setting updated:', result);
            } catch (error) {
                console.error('Error updating setting:', error);
            }
        }

        // Scan for duplicates
        async function scanDuplicates() {
            const container = document.getElementById('duplicates-list');
            container.innerHTML = '<p>Scanning for duplicates...</p>';

            try {
                const response = await fetch('/api/duplicates/scan');
                const duplicates = await response.json();

                if (duplicates.groups.length === 0) {
                    container.innerHTML = `
                        <div class="empty-state">
                            <p>No duplicates found!</p>
                        </div>
                    `;
                    return;
                }

                container.innerHTML = `
                    <p><strong>Found ${duplicates.summary.total_duplicate_groups} duplicate groups</strong></p>
                    <p>Total wasted space: ${duplicates.summary.total_wasted_space_mb} MB</p>
                    <div style="margin-top: 20px;">
                        ${duplicates.groups.slice(0, 10).map((group, idx) => `
                            <div class="file-item">
                                <div class="file-name">Duplicate Group ${idx + 1}</div>
                                <div class="file-meta">${group.count} files, ${group.size} bytes each</div>
                                ${group.paths.map(path => `<div style="font-size: 12px; color: #666;">${path}</div>`).join('')}
                            </div>
                        `).join('')}
                    </div>
                `;
            } catch (error) {
                console.error('Error scanning duplicates:', error);
                container.innerHTML = '<p style="color: red;">Error scanning for duplicates</p>';
            }
        }

        // Search files
        async function searchFiles() {
            const q = document.getElementById('search-query').value;
            const category = document.getElementById('search-category').value;
            const params = new URLSearchParams();
            if (q) params.append('q', q);
            if (category) params.append('category', category);

            const container = document.getElementById('search-results');
            container.innerHTML = '<p>Searching...</p>';

            try {
                const response = await fetch('/api/search?' + params.toString());
                const results = await response.json();

                if (!results || results.length === 0) {
                    container.innerHTML = '<div class="empty-state"><p>No results</p></div>';
                    return;
                }

                container.innerHTML = results.map(item => `
                    <div class="file-item">
                        <div class="file-name">${item.filename}</div>
                        <div class="file-meta">${item.operation} | ${new Date(item.timestamp).toLocaleString()}</div>
                        <div class="file-meta" style="font-size:12px;">${item.old_path} → ${item.new_path || 'Deleted'}</div>
                        <div class="file-meta" style="font-size:12px; font-style:italic;">Category: ${item.category || 'N/A'}</div>
                    </div>
                `).join('');

            } catch (error) {
                console.error('Error searching files:', error);
                container.innerHTML = '<p style="color: red;">Error searching files</p>';
            }
        }

        // Deep analyze file
        async function deepAnalyze(filePath) {
            const containerId = 'deep-analysis-' + btoa(filePath).replace(/=/g, '');
            const container = document.getElementById(containerId);

            // Show container and display loading state
            container.style.display = 'block';
            container.innerHTML = '<p style="color: #667eea;">🤖 Running deep agent analysis...</p>';

            try {
                const response = await fetch('/api/files/deep-analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file_path: filePath })
                });

                const result = await response.json();

                // Format the analysis result
                const evidence = result.evidence || [];
                const evidenceHtml = evidence.length > 0
                    ? `<div style="margin-top: 10px;">
                         <strong>Evidence:</strong>
                         <ul style="margin: 5px 0 0 20px; font-size: 13px;">
                           ${evidence.map(e => `<li>${e}</li>`).join('')}
                         </ul>
                       </div>`
                    : '';

                const blockReason = result.block_reason
                    ? `<div style="margin-top: 10px; color: #f44336;"><strong>⚠️ Blocked:</strong> ${result.block_reason}</div>`
                    : '';

                container.innerHTML = `
                    <div style="border-left: 3px solid #667eea; padding-left: 12px;">
                        <h4 style="margin: 0 0 10px 0; color: #667eea;">🧠 Agent Analysis Result</h4>
                        <div><strong>Category:</strong> ${result.category || 'N/A'}</div>
                        <div><strong>Suggested Path:</strong> ${result.suggested_path || 'N/A'}</div>
                        <div><strong>Rename:</strong> ${result.rename || 'No rename suggested'}</div>
                        <div><strong>Confidence:</strong> <span style="text-transform: uppercase; font-weight: bold; color: ${
                            result.confidence === 'high' ? '#4caf50' :
                            result.confidence === 'medium' ? '#ff9800' : '#f44336'
                        }">${result.confidence || 'N/A'}</span></div>
                        <div><strong>Suggested Action:</strong> ${result.action || 'none'}</div>
                        <div style="margin-top: 8px;"><strong>Reason:</strong> ${result.reason || 'N/A'}</div>
                        ${evidenceHtml}
                        ${blockReason}
                    </div>
                    <button class="btn-secondary" onclick="document.getElementById('${containerId}').style.display='none'" style="margin-top: 10px;">
                        Close
                    </button>
                `;

            } catch (error) {
                console.error('Error during deep analysis:', error);
                container.innerHTML = `
                    <p style="color: red;">❌ Deep analysis failed: ${error.message}</p>
                    <button class="btn-secondary" onclick="document.getElementById('${containerId}').style.display='none'" style="margin-top: 10px;">
                        Close
                    </button>
                `;
            }
        }

        // Initial load
        loadStats();
        loadPendingFiles();
        setInterval(loadStats, 30000); // Refresh stats every 30 seconds
        setInterval(loadPendingFiles, 10000); // Refresh pending files every 10 seconds
    </script>
</body>
</html>
"""


# ==================== API Endpoints ====================

@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve dashboard HTML."""
    return get_dashboard_html()


@app.get("/api/stats")
def get_stats():
    """Get statistics."""
    return state.db.get_stats('all')


@app.get("/api/pending-files")
def get_pending_files():
    """Get pending files for review."""
    return [
        {
            'file_path': item['file_path'],
            'filename': Path(item['file_path']).name,
            'classification': item['classification']
        }
        for item in state.pending_files
    ]


@app.post("/api/files/approve")
def approve_file(request: FileActionRequest):
    """Approve and execute file action."""
    # Find file in pending
    file_item = None
    for item in state.pending_files:
        if item['file_path'] == request.file_path:
            file_item = item
            break

    if not file_item:
        raise HTTPException(status_code=404, detail="File not found in pending list")

    # Execute action
    result = state.action_manager.execute(
        file_path=file_item['file_path'],
        classification=file_item['classification'],
        user_approved=True
    )

    # Remove from pending
    state.pending_files.remove(file_item)

    return result


@app.post("/api/files/reject")
def reject_file(request: FileActionRequest):
    """Reject file action."""
    # Remove from pending
    state.pending_files = [
        item for item in state.pending_files
        if item['file_path'] != request.file_path
    ]

    return {'success': True, 'message': 'File rejected'}


@app.get("/api/history")
def get_history():
    """Get recent file operation history."""
    return state.db.get_recent_logs(50)


@app.get("/api/search")
def search_files(q: str = None, category: str = None, limit: int = 100):
    """Search moved/renamed files in the history log.

    Params:
        q: substring to search filename/old_path/new_path
        category: optional category filter
        limit: max results
    """
    results = state.db.search_logs(query=q, category=category, limit=limit)
    return results


@app.get("/api/duplicates/scan")
def scan_duplicates():
    """Scan for duplicate files."""
    duplicates = []

    for folder in state.config.watched_folders:
        folder_dups = state.duplicate_finder.find_duplicates_in_directory(folder, recursive=False)
        duplicates.extend(folder_dups)

    summary = state.duplicate_finder.get_duplicate_summary(duplicates)

    return {
        'groups': duplicates,
        'summary': summary
    }


@app.get("/api/license/status")
def get_license_status():
    """Get license status."""
    return state.license_validator.check_license_status()


@app.post("/api/license/activate")
def activate_license(request: LicenseActivationRequest):
    """Activate license."""
    result = state.license_validator.activate_license(request.license_key)
    return result


@app.post("/api/files/deep-analyze")
def deep_analyze_file(request: DeepAnalyzeRequest, req: Request):
    """
    Perform deep agent analysis on a file.

    This endpoint uses the AgentAnalyzer to perform multi-step reasoning
    and return a detailed classification plan with evidence. The analysis
    is non-destructive and respects all folder policies and blacklists.

    Security: Validates file is in watched folders or pending list to prevent
    path traversal attacks and unauthorized file access. Rate limited to prevent
    DOS attacks.

    Args:
        request: Contains file_path to analyze
        req: FastAPI Request object for rate limiting

    Returns:
        Dict: Agent analysis result with category, suggested_path, evidence, etc.
    """
    # Rate limit check (HIGH-5 FIX)
    client_ip = req.client.host
    if not _check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {_MAX_REQUESTS_PER_WINDOW} requests per {_RATE_LIMIT_WINDOW} seconds."
        )

    file_path = request.file_path

    # Validate file exists
    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Validate file is in watched folders or pending list (HIGH-7 FIX)
    file_path_obj = Path(file_path).resolve()

    # Check if in pending files
    is_pending = any(
        Path(item['file_path']).resolve() == file_path_obj
        for item in state.pending_files
    )

    # Check if in watched folders
    is_watched = False
    for watched in state.config.watched_folders:
        watched_path = Path(watched).expanduser().resolve()
        try:
            if os.path.commonpath([str(file_path_obj), str(watched_path)]) == str(watched_path):
                is_watched = True
                break
        except ValueError:
            # Different drives on Windows - skip
            continue

    if not is_pending and not is_watched:
        raise HTTPException(
            status_code=403,
            detail="File not in watched folders or pending list"
        )

    # Check blacklist (defense in depth)
    blacklist = getattr(state.config, 'path_blacklist', []) or []
    for blacklisted in blacklist:
        try:
            blacklisted_resolved = Path(blacklisted).expanduser().resolve()
            if os.path.commonpath([str(file_path_obj), str(blacklisted_resolved)]) == str(blacklisted_resolved):
                raise HTTPException(
                    status_code=403,
                    detail="File is in blacklisted location"
                )
        except (ValueError, OSError):
            # Different drives on Windows - skip
            continue

    # Perform deep analysis using classifier with deep_analysis=True
    try:
        result = state.classifier.classify(str(file_path_obj), deep_analysis=True)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deep analysis failed: {str(e)}")


@app.post("/api/settings")
def update_settings(request: SettingsUpdateRequest):
    """Update settings."""
    if request.auto_mode is not None:
        state.config.update('auto_mode', request.auto_mode)

    if request.dry_run is not None:
        state.config.update('dry_run', request.dry_run)
        state.action_manager.set_dry_run(request.dry_run)

    if request.enable_ai is not None:
        state.config.update('classification.enable_ai', request.enable_ai)

    state.config.save()

    return {'success': True, 'message': 'Settings updated'}


@app.post("/api/watcher/start")
def start_watcher():
    """Start folder watcher."""
    state.start_watcher()
    return {'success': True, 'message': 'Watcher started'}


@app.post("/api/watcher/stop")
def stop_watcher():
    """Stop folder watcher."""
    state.stop_watcher()
    return {'success': True, 'message': 'Watcher stopped'}


def run_dashboard(host: str = "127.0.0.1", port: int = 5000):
    """
    Run the dashboard server.

    Args:
        host (str): Host to bind to
        port (int): Port to listen on
    """
    import uvicorn

    print(f"""
    ============================================
    AI File Organiser - Dashboard
    ============================================

    Dashboard: http://{host}:{port}

    Press Ctrl+C to stop
    ============================================
    """)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_dashboard()
