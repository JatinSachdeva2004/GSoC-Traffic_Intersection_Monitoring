"""
User Guide Tab - Help documentation viewer
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextBrowser, 
    QComboBox, QLabel, QPushButton, QSplitter, QTreeWidget, 
    QTreeWidgetItem, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtGui import QFont, QPixmap

import os
import markdown
from pathlib import Path

class UserGuideTab(QWidget):
    """User Guide tab for displaying help documentation"""
    
    def __init__(self):
        super().__init__()
        self.docs_path = Path(__file__).parent.parent / "docs" / "user-guide"
        self.setup_ui()
        self.load_documentation()
    
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        
        # Title
        title_label = QLabel("📚 User Guide & Documentation")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        
        # Document selector
        self.doc_selector = QComboBox()
        self.doc_selector.setMinimumWidth(250)
        self.doc_selector.currentTextChanged.connect(self.load_selected_document)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.load_documentation)
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(QLabel("Select Document:"))
        header_layout.addWidget(self.doc_selector)
        header_layout.addWidget(refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Main content area
        splitter = QSplitter(Qt.Horizontal)
        
        # Navigation tree (left panel)
        self.nav_tree = QTreeWidget()
        self.nav_tree.setHeaderLabel("Documentation")
        self.nav_tree.setMaximumWidth(250)
        self.nav_tree.itemClicked.connect(self.on_nav_item_clicked)
        
        # Content viewer (right panel)
        self.content_viewer = QTextBrowser()
        self.content_viewer.setOpenExternalLinks(True)
        
        splitter.addWidget(self.nav_tree)
        splitter.addWidget(self.content_viewer)
        splitter.setSizes([250, 800])
        
        layout.addWidget(splitter)
        
        # Status bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        
        layout.addLayout(status_layout)
    
    def load_documentation(self):
        """Load available documentation files"""
        try:
            self.doc_selector.clear()
            self.nav_tree.clear()
            
            if not self.docs_path.exists():
                self.content_viewer.setHtml(
                    "<h2>📚 Documentation</h2>"
                    "<p>No documentation directory found. Creating default documentation...</p>"
                )
                self.create_default_docs()
                return
            
            # Find all markdown files
            md_files = list(self.docs_path.glob("*.md"))
            
            if not md_files:
                self.content_viewer.setHtml(
                    "<h2>📚 Documentation</h2>"
                    "<p>No documentation files found in the user-guide directory.</p>"
                )
                return
            
            # Populate document selector
            for md_file in sorted(md_files):
                display_name = self.format_filename(md_file.stem)
                self.doc_selector.addItem(display_name, md_file)
            
            # Populate navigation tree
            self.populate_nav_tree(md_files)
            
            # Load first document
            if md_files:
                self.load_document(md_files[0])
                
            self.status_label.setText(f"Loaded {len(md_files)} documentation files")
            
        except Exception as e:
            self.show_error(f"Error loading documentation: {e}")
    
    def populate_nav_tree(self, md_files):
        """Populate the navigation tree"""
        categories = {
            "Getting Started": [],
            "Smart Intersection": [],
            "Configuration": [],
            "Troubleshooting": [],
            "Other": []
        }
        
        # Categorize files
        for md_file in md_files:
            filename = md_file.stem.lower()
            
            if any(word in filename for word in ['start', 'getting', 'intro', 'overview']):
                categories["Getting Started"].append(md_file)
            elif any(word in filename for word in ['smart', 'intersection', 'scene']):
                categories["Smart Intersection"].append(md_file)
            elif any(word in filename for word in ['config', 'setup', 'settings']):
                categories["Configuration"].append(md_file)
            elif any(word in filename for word in ['trouble', 'support', 'help', 'error']):
                categories["Troubleshooting"].append(md_file)
            else:
                categories["Other"].append(md_file)
        
        # Add to tree
        for category, files in categories.items():
            if files:
                category_item = QTreeWidgetItem(self.nav_tree, [category])
                category_item.setExpanded(True)
                
                for md_file in sorted(files):
                    file_item = QTreeWidgetItem(category_item, [self.format_filename(md_file.stem)])
                    file_item.setData(0, Qt.UserRole, md_file)
    
    def format_filename(self, filename):
        """Format filename for display"""
        # Convert kebab-case to title case
        return filename.replace('-', ' ').replace('_', ' ').title()
    
    def on_nav_item_clicked(self, item, column):
        """Handle navigation tree item click"""
        file_path = item.data(0, Qt.UserRole)
        if file_path and isinstance(file_path, Path):
            self.load_document(file_path)
            # Update selector
            for i in range(self.doc_selector.count()):
                if self.doc_selector.itemData(i) == file_path:
                    self.doc_selector.setCurrentIndex(i)
                    break
    
    def load_selected_document(self, display_name):
        """Load document selected from dropdown"""
        current_index = self.doc_selector.currentIndex()
        if current_index >= 0:
            file_path = self.doc_selector.itemData(current_index)
            if file_path:
                self.load_document(file_path)
    
    def load_document(self, file_path):
        """Load and display a markdown document"""
        try:
            if not file_path.exists():
                self.content_viewer.setHtml(f"<p>File not found: {file_path}</p>")
                return
            
            # Read markdown content
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            # Convert to HTML
            html_content = markdown.markdown(
                md_content,
                extensions=['toc', 'codehilite', 'fenced_code', 'tables']
            )
            
            # Add CSS styling
            styled_html = self.add_css_styling(html_content)
            
            # Display in browser
            self.content_viewer.setHtml(styled_html)
            
            self.status_label.setText(f"Loaded: {file_path.name}")
            
        except Exception as e:
            self.show_error(f"Error loading document {file_path.name}: {e}")
    
    def add_css_styling(self, html_content):
        """Add CSS styling to HTML content"""
        css = """
        <style>
        body {
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 20px;
            color: #333;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }
        h1 { border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        h2 { border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }
        
        code {
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: "Consolas", "Monaco", monospace;
        }
        
        pre {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
            overflow-x: auto;
        }
        
        blockquote {
            border-left: 4px solid #3498db;
            margin-left: 0;
            padding-left: 20px;
            color: #666;
            font-style: italic;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }
        
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        
        ul, ol {
            margin-left: 20px;
        }
        
        li {
            margin-bottom: 5px;
        }
        
        .highlight {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }
        </style>
        """
        
        return f"<!DOCTYPE html><html><head>{css}</head><body>{html_content}</body></html>"
    
    def create_default_docs(self):
        """Create default documentation if none exists"""
        try:
            self.docs_path.mkdir(parents=True, exist_ok=True)
            
            # Create a basic getting started document
            getting_started = """# Getting Started with Traffic Monitoring Desktop App

## Overview
Welcome to the Traffic Monitoring Desktop Application with Smart Intersection Analytics.

## Key Features
- **Real-time Detection**: YOLO-based object detection with Intel Arc GPU acceleration
- **Multi-camera Support**: Process multiple camera feeds simultaneously  
- **Scene Analytics**: Smart intersection analytics for traffic monitoring
- **VLM Insights**: Vision Language Model for contextual understanding
- **Local Processing**: All processing happens locally with no cloud dependency

## Quick Start
1. Launch the application: `python main.py`
2. Configure your cameras in the Config tab
3. Start live detection in the Live Detection tab
4. View analytics in the Analytics tab
5. Access insights through the VLM panel

## Support
For help and troubleshooting, check the other documentation sections or contact support.
"""
            
            with open(self.docs_path / "getting-started.md", 'w', encoding='utf-8') as f:
                f.write(getting_started)
            
            self.load_documentation()
            
        except Exception as e:
            self.show_error(f"Error creating default documentation: {e}")
    
    def show_error(self, message):
        """Show error message"""
        self.content_viewer.setHtml(f"<h2>Error</h2><p>{message}</p>")
        self.status_label.setText("Error occurred")
        
        # Also show message box for critical errors
        QMessageBox.warning(self, "Documentation Error", message)
