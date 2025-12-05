"""
GRAPH VISUALIZER
Exports knowledge graph to various visualization formats.
"""
import json
import logging
from typing import Dict, Optional
from pathlib import Path

from infrastructure.graph_db import GraphDB
from core.ontology import NodeType, NodeStatus, EdgeType

logger = logging.getLogger("Visualizer")

# Color schemes for node types and statuses
NODE_COLORS = {
    NodeType.REQ.value: "#FF6B6B",           # Red
    NodeType.CLARIFICATION.value: "#FFB347", # Orange
    NodeType.SPEC.value: "#4ECDC4",          # Teal
    NodeType.PLAN.value: "#45B7D1",          # Blue
    NodeType.CODE.value: "#96CEB4",          # Green
    NodeType.TEST.value: "#FFEAA7",          # Yellow
    NodeType.DOC.value: "#DDA0DD",           # Plum
    NodeType.ESCALATION.value: "#DC143C",    # Crimson
}

STATUS_SHAPES = {
    NodeStatus.PENDING.value: "ellipse",
    NodeStatus.PROCESSING.value: "box",
    NodeStatus.BLOCKED.value: "octagon",
    NodeStatus.TESTING.value: "hexagon",      # Gen-2 TDD: being tested
    NodeStatus.TESTED.value: "diamond",       # Gen-2 TDD: tests passed
    NodeStatus.VERIFIED.value: "doublecircle",
    NodeStatus.FAILED.value: "triangle",
}

EDGE_STYLES = {
    # High-level relationships
    EdgeType.TRACES_TO.value: {"color": "#888888", "style": "dashed"},
    EdgeType.DEPENDS_ON.value: {"color": "#E74C3C", "style": "solid"},
    EdgeType.IMPLEMENTS.value: {"color": "#27AE60", "style": "solid"},
    EdgeType.VERIFIES.value: {"color": "#3498DB", "style": "bold"},
    EdgeType.DEFINES.value: {"color": "#9B59B6", "style": "dotted"},
    EdgeType.BLOCKS.value: {"color": "#DC143C", "style": "bold"},
    EdgeType.FEEDBACK.value: {"color": "#F39C12", "style": "dashed"},
    EdgeType.RESOLVED_BY.value: {"color": "#2ECC71", "style": "solid"},
    # Extended relationships (AST/CPG)
    EdgeType.CONTAINS.value: {"color": "#2ECC71", "style": "solid"},
    EdgeType.REFERENCES.value: {"color": "#E74C3C", "style": "dashed"},
    EdgeType.INHERITS.value: {"color": "#9B59B6", "style": "bold"},
}


class GraphVisualizer:
    """
    Exports graph to visualization formats.

    Supported formats:
    - DOT (Graphviz)
    - JSON (D3.js / Cytoscape)
    - HTML (Standalone D3 visualization)
    - Mermaid (Markdown diagrams)

    Can work with either a GraphDB instance or a raw NetworkX graph.
    """

    def __init__(self, db_or_graph):
        """
        Initialize visualizer with either GraphDB or NetworkX graph.

        Args:
            db_or_graph: Either a GraphDB instance or a networkx.DiGraph
        """
        import networkx as nx
        if isinstance(db_or_graph, GraphDB):
            self.graph = db_or_graph.graph
        elif isinstance(db_or_graph, nx.DiGraph):
            self.graph = db_or_graph
        else:
            raise TypeError(f"Expected GraphDB or nx.DiGraph, got {type(db_or_graph)}")

    def to_dot(self, include_content: bool = False) -> str:
        """
        Export to DOT format for Graphviz.

        Args:
            include_content: Include node content in labels

        Returns:
            DOT string
        """
        lines = ['digraph GAADP {']
        lines.append('  rankdir=TB;')
        lines.append('  node [fontname="Arial"];')
        lines.append('  edge [fontname="Arial"];')
        lines.append('')

        # Nodes
        for node_id, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'UNKNOWN')
            status = data.get('status', 'PENDING')
            color = NODE_COLORS.get(node_type, '#CCCCCC')
            shape = STATUS_SHAPES.get(status, 'ellipse')

            label = f"{node_id[:15]}\\n[{node_type}]"
            if include_content:
                content = str(data.get('content', ''))[:50].replace('"', '\\"')
                label += f"\\n{content}..."

            lines.append(f'  "{node_id}" [')
            lines.append(f'    label="{label}",')
            lines.append(f'    shape={shape},')
            lines.append(f'    fillcolor="{color}",')
            lines.append(f'    style=filled')
            lines.append('  ];')

        lines.append('')

        # Edges
        for source, target, data in self.graph.edges(data=True):
            edge_type = data.get('type', 'UNKNOWN')
            style_info = EDGE_STYLES.get(edge_type, {"color": "#000000", "style": "solid"})

            lines.append(f'  "{source}" -> "{target}" [')
            lines.append(f'    label="{edge_type}",')
            lines.append(f'    color="{style_info["color"]}",')
            lines.append(f'    style={style_info["style"]}')
            lines.append('  ];')

        lines.append('}')
        return '\n'.join(lines)

    def to_json(self) -> Dict:
        """
        Export to JSON format for D3.js/Cytoscape.

        Returns:
            Dict with nodes and links arrays
        """
        nodes = []
        links = []

        for node_id, data in self.graph.nodes(data=True):
            nodes.append({
                "id": node_id,
                "type": data.get('type', 'UNKNOWN'),
                "status": data.get('status', 'PENDING'),
                "content_preview": str(data.get('content', ''))[:100],
                "color": NODE_COLORS.get(data.get('type'), '#CCCCCC'),
                "metadata": data.get('metadata', {})
            })

        for source, target, data in self.graph.edges(data=True):
            edge_type = data.get('type', 'UNKNOWN')
            style = EDGE_STYLES.get(edge_type, {"color": "#000000"})
            links.append({
                "source": source,
                "target": target,
                "type": edge_type,
                "color": style["color"],
                "signed_by": data.get('signed_by'),
                "created_at": data.get('created_at')
            })

        return {"nodes": nodes, "links": links}

    def to_mermaid(self) -> str:
        """
        Export to Mermaid format for Markdown diagrams.

        Returns:
            Mermaid diagram string
        """
        lines = ['graph TD']

        # Define node styles
        for node_id, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'UNKNOWN')
            status = data.get('status', 'PENDING')
            short_id = node_id[:12]

            # Mermaid node shapes based on status - explicit handling for all NodeStatus values
            if status == NodeStatus.VERIFIED.value:
                lines.append(f'    {short_id}(({node_type}))')     # Circle - success
            elif status == NodeStatus.FAILED.value:
                lines.append(f'    {short_id}>{node_type}]')       # Asymmetric - failure
            elif status == NodeStatus.TESTED.value:
                lines.append(f'    {short_id}{{{{{node_type}}}}}') # Hexagon - tests passed
            elif status == NodeStatus.TESTING.value:
                lines.append(f'    {short_id}[/{node_type}/]')     # Parallelogram - in test
            elif status == NodeStatus.BLOCKED.value:
                lines.append(f'    {short_id}{{{{node_type}}}}')   # Rhombus - blocked
            else:
                # PENDING, PROCESSING - default box
                lines.append(f'    {short_id}[{node_type}]')

        lines.append('')

        # Edges
        for source, target, data in self.graph.edges(data=True):
            edge_type = data.get('type', 'UNKNOWN')
            src_short = source[:12]
            tgt_short = target[:12]

            if edge_type == EdgeType.DEPENDS_ON.value:
                lines.append(f'    {src_short} --> |depends| {tgt_short}')
            elif edge_type == EdgeType.IMPLEMENTS.value:
                lines.append(f'    {src_short} -.-> |implements| {tgt_short}')
            elif edge_type == EdgeType.VERIFIES.value:
                lines.append(f'    {src_short} ==> |verifies| {tgt_short}')
            else:
                lines.append(f'    {src_short} --> {tgt_short}')

        return '\n'.join(lines)

    def to_html(self, title: str = "GAADP Knowledge Graph") -> str:
        """
        Export to standalone HTML with D3.js visualization.

        Returns:
            Complete HTML string
        """
        graph_data = self.to_json()

        return f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; }}
        #graph {{ width: 100vw; height: 100vh; }}
        .node {{ cursor: pointer; }}
        .node text {{ font-size: 10px; }}
        .link {{ stroke-opacity: 0.6; }}
        .tooltip {{
            position: absolute;
            background: white;
            border: 1px solid #ccc;
            padding: 10px;
            border-radius: 4px;
            max-width: 300px;
        }}
    </style>
</head>
<body>
    <div id="graph"></div>
    <script>
        const data = {json.dumps(graph_data)};

        const width = window.innerWidth;
        const height = window.innerHeight;

        const svg = d3.select("#graph")
            .append("svg")
            .attr("width", width)
            .attr("height", height);

        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2));

        const link = svg.append("g")
            .selectAll("line")
            .data(data.links)
            .join("line")
            .attr("class", "link")
            .attr("stroke", d => d.color)
            .attr("stroke-width", 2);

        const node = svg.append("g")
            .selectAll("g")
            .data(data.nodes)
            .join("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        node.append("circle")
            .attr("r", 20)
            .attr("fill", d => d.color);

        node.append("text")
            .attr("dy", 4)
            .attr("text-anchor", "middle")
            .text(d => d.type);

        node.append("title")
            .text(d => d.id + "\\n" + d.content_preview);

        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        }});

        function dragstarted(event) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }}

        function dragged(event) {{
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }}

        function dragended(event) {{
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }}
    </script>
</body>
</html>'''

    def export(
        self,
        output_path: str,
        format: str = "html",
        include_content: bool = False
    ) -> str:
        """
        Export graph to file.

        Args:
            output_path: Output file path
            format: One of 'dot', 'json', 'html', 'mermaid'
            include_content: Include content in DOT labels

        Returns:
            Output path
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "dot":
            content = self.to_dot(include_content)
        elif format == "json":
            content = json.dumps(self.to_json(), indent=2)
        elif format == "html":
            content = self.to_html()
        elif format == "mermaid":
            content = self.to_mermaid()
        else:
            raise ValueError(f"Unknown format: {format}")

        with open(path, 'w') as f:
            f.write(content)

        logger.info(f"Exported graph to {path} ({format})")
        return str(path)
