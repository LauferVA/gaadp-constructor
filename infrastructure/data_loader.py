"""
DATA LOADER
Ingests real datasets into the Graph so Agents can test against reality.
"""
import os
import uuid
import json
import csv
import logging
from typing import Optional, Dict, Any, List
from core.ontology import NodeType

logger = logging.getLogger("DataLoader")


class DataLoader:
    """
    Loads external data files into the knowledge graph as DOC nodes.
    Supports CSV, JSON, and plain text files.
    """

    def __init__(self, graph_db):
        self.db = graph_db
        self.loaded_files: List[str] = []

    def ingest_file(self, file_path: str, description: str = "") -> Optional[str]:
        """
        Reads a local file, creates a DATA/DOC node in the graph.

        Args:
            file_path: Path to the file to ingest
            description: Human-readable description of the data

        Returns:
            node_id if successful, None otherwise
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            print(f"   ❌ File not found: {file_path}")
            return None

        abs_path = os.path.abspath(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()

        try:
            # Parse based on file type
            if file_ext == '.json':
                content, preview = self._load_json(abs_path)
            elif file_ext == '.csv':
                content, preview = self._load_csv(abs_path)
            else:
                content, preview = self._load_text(abs_path)

            # Create node
            node_id = f"data_{uuid.uuid4().hex[:8]}"

            node_content = f"""FILE: {abs_path}
TYPE: {file_ext}
DESC: {description}
ROWS/LINES: {self._count_records(content, file_ext)}

PREVIEW:
{preview}
"""

            self.db.add_node(
                node_id,
                NodeType.DOC,
                content=node_content,
                metadata={
                    "source": "filesystem",
                    "path": abs_path,
                    "file_type": file_ext,
                    "description": description,
                    "raw_content_hash": hash(content)
                }
            )

            self.loaded_files.append(abs_path)
            print(f"   📄 Ingested Data Node: {node_id} ({file_path})")
            logger.info(f"Ingested {file_path} as node {node_id}")

            return node_id

        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            print(f"   ❌ Failed to ingest {file_path}: {e}")
            return None

    def _load_json(self, path: str) -> tuple:
        """Load JSON file and return content + preview."""
        with open(path, 'r') as f:
            data = json.load(f)

        content = json.dumps(data)
        preview = json.dumps(data, indent=2)[:500]
        if len(content) > 500:
            preview += "\n... (truncated)"

        return content, preview

    def _load_csv(self, path: str) -> tuple:
        """Load CSV file and return content + preview."""
        with open(path, 'r') as f:
            content = f.read()

        # Parse for preview
        lines = content.split('\n')
        preview_lines = lines[:10]  # First 10 rows
        preview = '\n'.join(preview_lines)
        if len(lines) > 10:
            preview += f"\n... ({len(lines) - 10} more rows)"

        return content, preview

    def _load_text(self, path: str) -> tuple:
        """Load text file and return content + preview."""
        with open(path, 'r') as f:
            content = f.read()

        preview = content[:500]
        if len(content) > 500:
            preview += "\n... (truncated)"

        return content, preview

    def _count_records(self, content: str, file_ext: str) -> int:
        """Count records in the content."""
        if file_ext == '.json':
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    return len(data)
                elif isinstance(data, dict):
                    return len(data.keys())
            except:
                pass
            return 1
        elif file_ext == '.csv':
            return len(content.split('\n')) - 1  # Exclude header
        else:
            return len(content.split('\n'))

    def ingest_directory(self, dir_path: str, pattern: str = "*") -> List[str]:
        """
        Ingest all matching files from a directory.

        Args:
            dir_path: Directory to scan
            pattern: Glob pattern (e.g., "*.csv")

        Returns:
            List of created node_ids
        """
        import glob

        node_ids = []
        search_path = os.path.join(dir_path, pattern)

        for file_path in glob.glob(search_path):
            if os.path.isfile(file_path):
                node_id = self.ingest_file(file_path, f"Auto-ingested from {dir_path}")
                if node_id:
                    node_ids.append(node_id)

        return node_ids

    def interactive_load(self) -> List[str]:
        """
        Interactive mode for loading files.
        Returns list of created node_ids.
        """
        node_ids = []

        print("\n📂 DATA INGESTION MODE")
        print("   Enter file paths one at a time. Press Enter with empty path to finish.")

        while True:
            path = input("   📁 File path (or Enter to skip): ").strip()
            if not path:
                break

            desc = input("   📝 Description: ").strip()
            node_id = self.ingest_file(path, desc)
            if node_id:
                node_ids.append(node_id)

        return node_ids

    def get_loaded_summary(self) -> Dict[str, Any]:
        """Get summary of all loaded files."""
        return {
            "total_files": len(self.loaded_files),
            "files": self.loaded_files
        }
