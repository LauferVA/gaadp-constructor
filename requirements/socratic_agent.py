"""
SOCRATIC RESEARCH AGENT
Implements the research phase that precedes Architect planning.

Two modes:
1. INTERACTIVE - Asks user clarifying questions (production)
2. AUTOMATED - Extracts specs from existing codebase (development/testing)

The Socratic phase ensures requirements are complete and unambiguous
before expensive LLM calls to Architect/Builder/Verifier.
"""
import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from agents.base_agent import BaseAgent
from core.ontology import NodeType, EdgeType, NodeStatus, AgentRole
from core.protocols import SocratesOutput, ClarifyingQuestion, get_agent_tools
from infrastructure.graph_db import GraphDB

logger = logging.getLogger("Socrates")


class QuestionPriority(str, Enum):
    """Priority levels for clarifying questions."""
    CRITICAL = "CRITICAL"  # Blocks all progress
    HIGH = "HIGH"          # Significantly impacts architecture
    MEDIUM = "MEDIUM"      # Affects implementation details
    LOW = "LOW"            # Nice to have clarification


@dataclass
class AmbiguityAnalysis:
    """Result of analyzing a requirement for ambiguity."""
    ambiguity_score: float  # 0.0 (clear) to 1.0 (very ambiguous)
    missing_details: List[str]
    conflicting_aspects: List[str]
    undefined_terms: List[str]
    suggested_questions: List[Dict]
    research_hints: List[str]  # Things to look up automatically


@dataclass
class ResearchResult:
    """Result of automated research."""
    source: str  # "codebase", "web", "docs", "user"
    content: str
    relevance_score: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class SocraticConfig:
    """Configuration for the Socratic phase."""
    ambiguity_threshold: float = 0.3  # Below this, proceed to Architect
    max_questions_per_round: int = 5
    max_research_rounds: int = 3
    user_timeout_seconds: float = 300.0
    auto_answer_low_priority: bool = True  # Use defaults for LOW questions


class SocraticPhase(ABC):
    """
    Abstract base for the Socratic research phase.

    Subclasses implement different strategies:
    - InteractiveSocraticPhase: Asks user questions
    - DevelopmentSocraticPhase: Extracts from existing code
    """

    def __init__(self, db: GraphDB, config: Optional[SocraticConfig] = None):
        self.db = db
        self.config = config or SocraticConfig()
        self.logger = logging.getLogger(f"Socrates.{self.__class__.__name__}")

    @abstractmethod
    async def research(self, requirement: str, req_id: str) -> Dict:
        """
        Execute the research phase.

        Args:
            requirement: The raw requirement text
            req_id: The REQ node ID in the graph

        Returns:
            Dict with:
                - enriched_requirement: Enhanced requirement text
                - sub_requirements: List of decomposed sub-REQs
                - research_context: Gathered information
                - saturation_reached: Whether we have enough info
        """
        pass

    def _analyze_ambiguity(self, requirement: str) -> AmbiguityAnalysis:
        """
        Analyze a requirement for ambiguity.

        This is a heuristic analysis - looks for:
        - Vague terms ("fast", "secure", "user-friendly")
        - Missing specifics (no file paths, no data types)
        - Implicit assumptions
        """
        # Vague terms that need clarification
        vague_terms = [
            "fast", "quick", "efficient", "secure", "safe", "robust",
            "user-friendly", "easy", "simple", "complex", "advanced",
            "modern", "best practice", "standard", "proper", "good",
            "handle", "process", "manage", "support"
        ]

        # Technical aspects that should be specified
        missing_checks = {
            "data_types": ["int", "str", "float", "list", "dict", "bool", "class"],
            "file_paths": [".py", ".js", ".ts", ".json", ".yaml", "/", "\\"],
            "error_handling": ["error", "exception", "try", "catch", "raise"],
            "testing": ["test", "assert", "verify", "validate"],
            "dependencies": ["import", "require", "from", "library", "package"],
        }

        req_lower = requirement.lower()

        # Find vague terms
        undefined_terms = [term for term in vague_terms if term in req_lower]

        # Check for missing details
        missing_details = []
        for category, indicators in missing_checks.items():
            if not any(ind in req_lower for ind in indicators):
                if category == "file_paths":
                    missing_details.append("No target file path specified")
                elif category == "error_handling":
                    missing_details.append("Error handling approach not specified")
                elif category == "testing":
                    missing_details.append("Testing requirements not specified")
                elif category == "dependencies":
                    missing_details.append("Dependencies/imports not specified")

        # Generate suggested questions
        suggested_questions = []

        if "No target file path specified" in missing_details:
            suggested_questions.append({
                "question": "Where should this code be placed? (file path)",
                "priority": QuestionPriority.HIGH.value,
                "options": None,
                "default": "auto-determine from context"
            })

        for term in undefined_terms[:3]:  # Limit to top 3
            suggested_questions.append({
                "question": f"What does '{term}' mean in this context?",
                "priority": QuestionPriority.MEDIUM.value,
                "options": None,
                "default": f"Use reasonable defaults for '{term}'"
            })

        # Research hints
        research_hints = []
        if any(term in req_lower for term in ["api", "endpoint", "rest", "http"]):
            research_hints.append("Look for existing API patterns in codebase")
        if any(term in req_lower for term in ["database", "db", "sql", "model"]):
            research_hints.append("Check existing database models")
        if any(term in req_lower for term in ["auth", "login", "user", "permission"]):
            research_hints.append("Review authentication patterns")

        # Calculate ambiguity score
        score = min(1.0, (
            len(undefined_terms) * 0.1 +
            len(missing_details) * 0.15 +
            (0.2 if len(requirement) < 50 else 0)  # Short requirements are often vague
        ))

        return AmbiguityAnalysis(
            ambiguity_score=score,
            missing_details=missing_details,
            conflicting_aspects=[],  # Would need LLM to detect
            undefined_terms=undefined_terms,
            suggested_questions=suggested_questions,
            research_hints=research_hints
        )

    def _create_sub_requirement(
        self,
        parent_req_id: str,
        content: str,
        source: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Create a sub-requirement node linked to parent."""
        sub_req_id = f"req_{uuid.uuid4().hex[:8]}"

        self.db.add_node(
            sub_req_id,
            NodeType.REQ,
            content,
            metadata={
                "source": source,
                "parent_id": parent_req_id,
                **(metadata or {})
            }
        )

        # Link to parent
        self.db.add_edge(
            sub_req_id, parent_req_id,
            EdgeType.TRACES_TO,
            signed_by="socrates",
            signature=f"socratic_{uuid.uuid4().hex[:8]}"
        )

        self.logger.info(f"Created sub-REQ: {sub_req_id} → {parent_req_id}")
        return sub_req_id


class InteractiveSocraticPhase(SocraticPhase):
    """
    Interactive Socratic phase that asks the user clarifying questions.

    Flow:
    1. Analyze requirement for ambiguity
    2. Research automatically (codebase, docs)
    3. Generate prioritized questions
    4. Ask user (with timeout and defaults)
    5. Incorporate answers into requirement
    6. Loop until saturation or max rounds
    """

    def __init__(
        self,
        db: GraphDB,
        gateway=None,  # LLM Gateway for intelligent analysis
        config: Optional[SocraticConfig] = None
    ):
        super().__init__(db, config)
        self.gateway = gateway
        self._pending_questions: List[Dict] = []
        self._user_answers: Dict[str, str] = {}

    async def research(self, requirement: str, req_id: str) -> Dict:
        """Execute interactive research phase."""
        self.logger.info(f"Starting interactive research for {req_id}")

        enriched_requirement = requirement
        sub_requirements = []
        research_context = []

        for round_num in range(self.config.max_research_rounds):
            self.logger.info(f"Research round {round_num + 1}/{self.config.max_research_rounds}")

            # 1. Analyze current state
            analysis = self._analyze_ambiguity(enriched_requirement)
            self.logger.info(f"Ambiguity score: {analysis.ambiguity_score:.2f}")

            # 2. Check saturation
            if analysis.ambiguity_score < self.config.ambiguity_threshold:
                self.logger.info("Saturation reached - ambiguity below threshold")
                return {
                    "enriched_requirement": enriched_requirement,
                    "sub_requirements": sub_requirements,
                    "research_context": research_context,
                    "saturation_reached": True,
                    "rounds_completed": round_num + 1
                }

            # 3. Automated research
            for hint in analysis.research_hints:
                result = await self._automated_research(hint)
                if result:
                    research_context.append(result)

            # 4. Ask user questions
            questions = analysis.suggested_questions[:self.config.max_questions_per_round]

            if questions:
                answers = await self._ask_user_questions(questions)

                # Incorporate answers
                for q, a in answers.items():
                    if a and a != "skip":
                        # Create sub-requirement for each substantial answer
                        sub_req_id = self._create_sub_requirement(
                            req_id,
                            f"Clarification: {q}\nAnswer: {a}",
                            source="user"
                        )
                        sub_requirements.append(sub_req_id)

                        # Enrich the requirement
                        enriched_requirement += f"\n\n[User Clarification]\nQ: {q}\nA: {a}"

        # Max rounds reached
        self.logger.warning("Max research rounds reached without saturation")
        return {
            "enriched_requirement": enriched_requirement,
            "sub_requirements": sub_requirements,
            "research_context": research_context,
            "saturation_reached": False,
            "rounds_completed": self.config.max_research_rounds
        }

    async def _automated_research(self, hint: str) -> Optional[ResearchResult]:
        """Perform automated research based on hint."""
        # This would use tools like read_file, list_directory, etc.
        # For now, return None - to be implemented with MCP tools
        self.logger.debug(f"Automated research hint: {hint}")
        return None

    async def _ask_user_questions(self, questions: List[Dict]) -> Dict[str, str]:
        """
        Ask user clarifying questions via console.

        Returns dict mapping question -> answer
        """
        answers = {}

        print("\n" + "=" * 60)
        print("🤔 SOCRATIC QUESTIONS - Please clarify:")
        print("=" * 60)

        for i, q in enumerate(questions, 1):
            priority = q.get("priority", "MEDIUM")
            question_text = q.get("question", "")
            options = q.get("options")
            default = q.get("default")

            print(f"\n[{priority}] Question {i}:")
            print(f"  {question_text}")

            if options:
                for j, opt in enumerate(options, 1):
                    print(f"    {j}. {opt}")
                print(f"    (or type your own answer)")

            if default:
                print(f"  [Default: {default}]")

            # Auto-answer LOW priority if configured
            if priority == "LOW" and self.config.auto_answer_low_priority and default:
                print(f"  → Using default (LOW priority)")
                answers[question_text] = default
                continue

            try:
                # Get user input with timeout
                print("  Your answer (or press Enter for default, 'skip' to skip): ", end="", flush=True)

                # Use asyncio for timeout
                try:
                    loop = asyncio.get_event_loop()
                    answer = await asyncio.wait_for(
                        loop.run_in_executor(None, input),
                        timeout=self.config.user_timeout_seconds
                    )
                except asyncio.TimeoutError:
                    print(f"\n  → Timeout, using default")
                    answer = ""

                if not answer.strip():
                    answer = default or "skip"

                answers[question_text] = answer.strip()

            except Exception as e:
                self.logger.error(f"Error getting user input: {e}")
                answers[question_text] = default or "skip"

        print("\n" + "=" * 60)
        return answers


class DevelopmentSocraticPhase(SocraticPhase):
    """
    Automated Socratic phase for development and testing.

    Instead of asking users, this extracts specifications from:
    - Existing codebase (the code IS the spec)
    - Docstrings and comments
    - Test files (expected behavior)
    - Config files

    Perfect for:
    - Benchmarking (can GAADP reconstruct this code?)
    - Regression testing
    - Automated CI/CD pipelines
    """

    def __init__(
        self,
        db: GraphDB,
        source_path: str = ".",  # Path to existing codebase
        config: Optional[SocraticConfig] = None
    ):
        super().__init__(db, config)
        self.source_path = source_path

    async def research(self, requirement: str, req_id: str) -> Dict:
        """
        Execute automated research phase.

        For dev/test mode, the "requirement" is often a pointer to existing code.
        We extract the specification from that code.
        """
        self.logger.info(f"Starting development research for {req_id}")

        sub_requirements = []
        research_context = []

        # Parse the requirement to understand what we're analyzing
        analysis_target = self._parse_analysis_target(requirement)

        if analysis_target.get("type") == "file":
            # Extract spec from a specific file
            file_path = analysis_target.get("path")
            spec = await self._extract_spec_from_file(file_path)
            research_context.append(spec)

            # Create sub-requirements for each extracted component
            for component in spec.get("components", []):
                sub_req_id = self._create_sub_requirement(
                    req_id,
                    component["description"],
                    source="code_analysis",
                    metadata={"component_type": component["type"], "source_file": file_path}
                )
                sub_requirements.append(sub_req_id)

        elif analysis_target.get("type") == "directory":
            # Analyze entire directory structure
            dir_path = analysis_target.get("path")
            structure = await self._analyze_directory(dir_path)
            research_context.append(structure)

        elif analysis_target.get("type") == "reconstruction":
            # Special mode: hide code and see if we can rebuild it
            target_file = analysis_target.get("path")
            spec = await self._create_reconstruction_spec(target_file)
            research_context.append(spec)

        # Build enriched requirement
        enriched_requirement = requirement
        if research_context:
            enriched_requirement += "\n\n[AUTO-EXTRACTED CONTEXT]\n"
            for ctx in research_context:
                if isinstance(ctx, dict):
                    enriched_requirement += json.dumps(ctx, indent=2)[:1000] + "\n"

        return {
            "enriched_requirement": enriched_requirement,
            "sub_requirements": sub_requirements,
            "research_context": research_context,
            "saturation_reached": True,  # Dev mode always proceeds
            "rounds_completed": 1,
            "mode": "development"
        }

    def _parse_analysis_target(self, requirement: str) -> Dict:
        """Parse requirement to determine what to analyze."""
        req_lower = requirement.lower()

        # Check for file reference
        if any(ext in requirement for ext in [".py", ".js", ".ts", ".yaml", ".json"]):
            # Extract file path
            import re
            match = re.search(r'[\w\/\-\.]+\.(py|js|ts|yaml|json|md)', requirement)
            if match:
                return {"type": "file", "path": match.group(0)}

        # Check for reconstruction mode
        if "reconstruct" in req_lower or "rebuild" in req_lower:
            match = re.search(r'[\w\/\-\.]+\.(py|js|ts)', requirement)
            if match:
                return {"type": "reconstruction", "path": match.group(0)}

        # Check for directory
        if "/" in requirement and "." not in requirement.split("/")[-1]:
            return {"type": "directory", "path": requirement.strip()}

        # Default: treat as natural language requirement
        return {"type": "natural", "content": requirement}

    async def _extract_spec_from_file(self, file_path: str) -> Dict:
        """
        Extract specification from an existing Python file.

        Returns structured spec including:
        - Module docstring (purpose)
        - Class definitions (with docstrings)
        - Function signatures (with docstrings)
        - Dependencies (imports)
        """
        import ast
        import os

        full_path = os.path.join(self.source_path, file_path)

        if not os.path.exists(full_path):
            self.logger.warning(f"File not found: {full_path}")
            return {"error": f"File not found: {file_path}", "components": []}

        with open(full_path, "r") as f:
            source = f.read()

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return {"error": f"Syntax error: {e}", "components": []}

        spec = {
            "file_path": file_path,
            "module_docstring": ast.get_docstring(tree),
            "imports": [],
            "components": []
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    spec["imports"].append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    spec["imports"].append(f"{module}.{alias.name}")

            elif isinstance(node, ast.ClassDef):
                class_spec = {
                    "type": "class",
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                    "methods": [],
                    "description": f"Class '{node.name}'"
                }

                if ast.get_docstring(node):
                    class_spec["description"] += f": {ast.get_docstring(node)[:100]}"

                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_spec = {
                            "name": item.name,
                            "args": [arg.arg for arg in item.args.args],
                            "docstring": ast.get_docstring(item)
                        }
                        class_spec["methods"].append(method_spec)

                spec["components"].append(class_spec)

            elif isinstance(node, ast.FunctionDef) and not isinstance(node, ast.AsyncFunctionDef):
                # Top-level functions only
                if hasattr(node, 'col_offset') and node.col_offset == 0:
                    func_spec = {
                        "type": "function",
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "docstring": ast.get_docstring(node),
                        "description": f"Function '{node.name}'"
                    }

                    if ast.get_docstring(node):
                        func_spec["description"] += f": {ast.get_docstring(node)[:100]}"

                    spec["components"].append(func_spec)

        return spec

    async def _analyze_directory(self, dir_path: str) -> Dict:
        """Analyze directory structure for project understanding."""
        import os

        full_path = os.path.join(self.source_path, dir_path)

        structure = {
            "path": dir_path,
            "files": [],
            "subdirs": [],
            "key_files": []
        }

        if not os.path.exists(full_path):
            return {"error": f"Directory not found: {dir_path}"}

        for item in os.listdir(full_path):
            item_path = os.path.join(full_path, item)

            if os.path.isdir(item_path):
                if not item.startswith(".") and item != "__pycache__":
                    structure["subdirs"].append(item)
            else:
                structure["files"].append(item)

                # Identify key files
                if item in ["__init__.py", "main.py", "app.py", "server.py"]:
                    structure["key_files"].append(item)
                elif item.endswith(".py") and not item.startswith("test_"):
                    structure["key_files"].append(item)

        return structure

    async def _create_reconstruction_spec(self, target_file: str) -> Dict:
        """
        Create a specification for reconstructing a file.

        This is for the "code reconstruction" benchmark:
        - Extract what the code DOES (behavior spec)
        - Hide HOW it does it (implementation)
        - See if GAADP can recreate the logic
        """
        file_spec = await self._extract_spec_from_file(target_file)

        if "error" in file_spec:
            return file_spec

        # Create behavior-focused spec (not implementation)
        reconstruction_spec = {
            "type": "reconstruction_challenge",
            "target_file": target_file,
            "objective": f"Recreate the functionality of {target_file}",
            "constraints": [
                "Must implement all public functions/classes",
                "Must pass the same test cases (if any)",
                "Implementation details may differ"
            ],
            "public_interface": []
        }

        for component in file_spec.get("components", []):
            if component["type"] == "class":
                interface = {
                    "type": "class",
                    "name": component["name"],
                    "docstring": component.get("docstring"),
                    "methods": [
                        {"name": m["name"], "args": m["args"], "docstring": m.get("docstring")}
                        for m in component.get("methods", [])
                        if not m["name"].startswith("_")  # Public methods only
                    ]
                }
                reconstruction_spec["public_interface"].append(interface)

            elif component["type"] == "function":
                if not component["name"].startswith("_"):
                    interface = {
                        "type": "function",
                        "name": component["name"],
                        "args": component.get("args", []),
                        "docstring": component.get("docstring")
                    }
                    reconstruction_spec["public_interface"].append(interface)

        return reconstruction_spec


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_socratic_phase(
    db: GraphDB,
    mode: str = "interactive",
    **kwargs
) -> SocraticPhase:
    """
    Factory function to create appropriate Socratic phase.

    Args:
        db: GraphDB instance
        mode: "interactive" or "development"
        **kwargs: Additional config options

    Returns:
        Configured SocraticPhase instance
    """
    config = SocraticConfig(**{k: v for k, v in kwargs.items() if hasattr(SocraticConfig, k)})

    if mode == "interactive":
        gateway = kwargs.get("gateway")
        return InteractiveSocraticPhase(db, gateway, config)
    elif mode == "development":
        source_path = kwargs.get("source_path", ".")
        return DevelopmentSocraticPhase(db, source_path, config)
    else:
        raise ValueError(f"Unknown mode: {mode}")


async def run_socratic_phase(
    db: GraphDB,
    requirement: str,
    req_id: str,
    mode: str = "interactive",
    **kwargs
) -> Dict:
    """
    Convenience function to run Socratic phase.

    Args:
        db: GraphDB instance
        requirement: The requirement text
        req_id: The REQ node ID
        mode: "interactive" or "development"

    Returns:
        Research results dict
    """
    phase = create_socratic_phase(db, mode, **kwargs)
    return await phase.research(requirement, req_id)
