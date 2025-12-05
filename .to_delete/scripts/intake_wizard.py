#!/usr/bin/env python3
"""
GAADP DETERMINISTIC INTAKE WIZARD (Phase 1)

A CLI tool that collects L1 system-level requirements and generates prompt.md.
This is the first phase of the two-phase specification process:
  Phase 1 (This): Deterministic data collection - no AI needed
  Phase 2 (Socrates): Dialectic specification - AI-driven gap analysis

Usage:
    python scripts/intake_wizard.py                    # Interactive mode
    python scripts/intake_wizard.py --output my.md    # Custom output file
    python scripts/intake_wizard.py --json input.json # Import from JSON
    python scripts/intake_wizard.py --minimal         # Skip optional questions

The output prompt.md file becomes input to the Socrates agent for Phase 2.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


# =============================================================================
# CANONICAL QUESTIONS (from docs/canonical_questions_final.tsv)
# =============================================================================

L1_QUESTIONS = [
    {
        "id": "L1.01",
        "category": "Problem",
        "question": "What is the single most important problem this software solves for you?",
        "technical_name": "Problem Statement",
        "required": True,
        "multiline": True,
    },
    {
        "id": "L1.02",
        "category": "Users",
        "question": "Who exactly will use this? Describe the person at the keyboard.",
        "technical_name": "User Persona",
        "required": True,
        "multiline": True,
        "examples": ["A busy doctor reviewing patient data", "A warehouse worker scanning inventory", "A teenager browsing social media"],
    },
    {
        "id": "L1.03",
        "category": "Integration",
        "question": "Does this need to connect to any existing systems you already use?",
        "technical_name": "System Integration",
        "required": True,
        "multiline": True,
        "examples": ["Salesforce", "SAP", "PostgreSQL database", "REST API at api.example.com"],
    },
    {
        "id": "L1.04",
        "category": "Compliance",
        "question": "Are there legal or industry rules we must follow?",
        "technical_name": "Regulatory Compliance",
        "required": True,
        "options": ["None", "HIPAA", "GDPR", "PCI-DSS", "SOC2", "Other"],
        "allow_multiple": True,
    },
    {
        "id": "L1.05",
        "category": "Timeline",
        "question": "Is there a hard deadline we need to hit?",
        "technical_name": "Timeline Constraint",
        "required": True,
        "multiline": False,
        "examples": ["Launch by Q1 2025", "Demo next Tuesday", "No hard deadline"],
    },
    {
        "id": "L1.06",
        "category": "Budget",
        "question": "Is there a budget limit for building or running this?",
        "technical_name": "Budget Constraint",
        "required": True,
        "multiline": False,
        "examples": ["$5000 total", "$100/month hosting max", "No budget constraint"],
    },
    {
        "id": "L1.07",
        "category": "Scale",
        "question": "How many people will use this at once? Now and in a year?",
        "technical_name": "Concurrency Scale",
        "required": True,
        "multiline": False,
        "examples": ["10 users now, 100 in a year", "Millions of concurrent users", "Just me"],
    },
    {
        "id": "L1.08",
        "category": "Platform",
        "question": "Where will people use this?",
        "technical_name": "Platform Target",
        "required": True,
        "options": ["Web browser", "Mobile (iOS)", "Mobile (Android)", "Desktop", "CLI tool", "API/Backend only", "Other"],
        "allow_multiple": True,
    },
    {
        "id": "L1.09",
        "category": "Trade-off",
        "question": "If you had to choose: get something working fast, or build it to last?",
        "technical_name": "Speed vs Robustness",
        "required": True,
        "options": ["MVP/Prototype (fast)", "Production-grade (robust)", "Balanced"],
        "allow_multiple": False,
    },
]

TERMINAL_QUESTIONS = [
    {
        "id": "T1",
        "category": "Features",
        "question": "What features do you need? (List each on a new line)",
        "technical_name": "Feature List",
        "required": True,
        "multiline": True,
        "list_input": True,
    },
    {
        "id": "T2",
        "category": "Additional",
        "question": "What else do you want the code to do? (Anything we missed?)",
        "technical_name": "Additional Requirements",
        "required": False,
        "multiline": True,
    },
    {
        "id": "T3",
        "category": "Delegation",
        "question": "Want to skip any of these questions? I'll use my best judgment for skipped items.\nWhich categories can the AI decide for you?",
        "technical_name": "Delegation License",
        "required": False,
        "options": ["Error handling details", "Code style/formatting", "Test coverage level", "Performance optimizations", "None - ask me everything"],
        "allow_multiple": True,
    },
]


# =============================================================================
# CLI INTERACTION UTILITIES
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


def print_header(text: str):
    """Print a styled header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.END}\n")


def print_question(q: Dict):
    """Print a formatted question."""
    required = f"{Colors.RED}*{Colors.END}" if q.get("required") else ""
    print(f"{Colors.BOLD}[{q['id']}]{Colors.END} {q['question']} {required}")

    if q.get("examples"):
        print(f"{Colors.DIM}  Examples: {', '.join(q['examples'][:3])}{Colors.END}")

    if q.get("options"):
        for i, opt in enumerate(q["options"], 1):
            print(f"  {Colors.YELLOW}{i}.{Colors.END} {opt}")


def get_input(q: Dict) -> Any:
    """Get user input for a question."""
    if q.get("options"):
        return get_option_input(q)
    elif q.get("multiline"):
        return get_multiline_input(q)
    else:
        return get_single_input(q)


def get_single_input(q: Dict) -> str:
    """Get single-line input."""
    while True:
        response = input(f"{Colors.GREEN}>{Colors.END} ").strip()
        if response or not q.get("required"):
            return response
        print(f"{Colors.RED}This field is required.{Colors.END}")


def get_multiline_input(q: Dict) -> str:
    """Get multi-line input (end with empty line)."""
    print(f"{Colors.DIM}  (Enter text, press Enter twice when done){Colors.END}")
    lines = []
    while True:
        line = input()
        if not line and lines:
            break
        lines.append(line)

    result = "\n".join(lines).strip()
    if not result and q.get("required"):
        print(f"{Colors.RED}This field is required.{Colors.END}")
        return get_multiline_input(q)
    return result


def get_option_input(q: Dict) -> List[str]:
    """Get selection from options."""
    options = q["options"]
    allow_multiple = q.get("allow_multiple", False)

    if allow_multiple:
        print(f"{Colors.DIM}  (Enter numbers separated by commas, e.g., 1,3,4){Colors.END}")

    while True:
        response = input(f"{Colors.GREEN}>{Colors.END} ").strip()

        if not response:
            if q.get("required"):
                print(f"{Colors.RED}This field is required.{Colors.END}")
                continue
            return []

        try:
            indices = [int(x.strip()) for x in response.split(",")]
            if all(1 <= i <= len(options) for i in indices):
                selected = [options[i-1] for i in indices]
                if not allow_multiple and len(selected) > 1:
                    print(f"{Colors.RED}Please select only one option.{Colors.END}")
                    continue
                return selected
        except ValueError:
            pass

        print(f"{Colors.RED}Please enter valid option number(s).{Colors.END}")


def get_list_input(q: Dict) -> List[str]:
    """Get a list of items (features)."""
    print(f"{Colors.DIM}  (Enter each item on a new line, press Enter twice when done){Colors.END}")
    items = []
    while True:
        item = input(f"{Colors.GREEN}  -{Colors.END} ").strip()
        if not item:
            if items or not q.get("required"):
                break
            print(f"{Colors.RED}Please enter at least one item.{Colors.END}")
            continue
        items.append(item)
    return items


# =============================================================================
# PROMPT.MD GENERATOR
# =============================================================================

def generate_prompt_md(answers: Dict[str, Any], output_path: str) -> str:
    """Generate the prompt.md file from collected answers."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build the document
    lines = [
        "# GAADP Project Specification",
        f"# Generated: {timestamp}",
        "# Phase: Deterministic Intake (Phase 1)",
        "",
        "---",
        "",
        "## L1: System-Level Requirements",
        "",
    ]

    # L1 Questions
    section_map = {
        "L1.01": ("Problem Statement", "PROBLEM_STATEMENT"),
        "L1.02": ("User Persona", "USER_PERSONA"),
        "L1.03": ("System Integration", "INTEGRATIONS"),
        "L1.04": ("Regulatory Compliance", "COMPLIANCE"),
        "L1.05": ("Timeline Constraint", "TIMELINE"),
        "L1.06": ("Budget Constraint", "BUDGET"),
        "L1.07": ("Concurrency Scale", "SCALE"),
        "L1.08": ("Platform Target", "PLATFORM"),
        "L1.09": ("Speed vs Robustness Trade-off", "TRADEOFF"),
    }

    for qid, (title, _) in section_map.items():
        answer = answers.get(qid, "Not specified")

        if isinstance(answer, list):
            answer_str = ", ".join(answer) if answer else "None specified"
        else:
            answer_str = answer or "Not specified"

        lines.extend([
            f"### {title}",
            f"**{qid}**",
            "```",
            answer_str,
            "```",
            "",
        ])

    # Feature List
    lines.extend([
        "---",
        "",
        "## Feature List",
        "",
        "### Core Features",
    ])

    features = answers.get("T1", [])
    if features:
        for feat in features:
            lines.append(f"- [ ] {feat}")
    else:
        lines.append("- (No features specified)")
    lines.append("")

    # Additional Requirements
    additional = answers.get("T2", "")
    if additional:
        lines.extend([
            "### Additional Requirements",
            "```",
            additional,
            "```",
            "",
        ])

    # Delegation License
    delegations = answers.get("T3", [])
    lines.extend([
        "---",
        "",
        "## Delegation License",
        "",
        "The user has granted the following delegation permissions:",
        "",
    ])

    if delegations and "None - ask me everything" not in delegations:
        for d in delegations:
            lines.append(f"- **{d}**: Agent may decide")
        lines.append("")
        lines.append("For these items, the Socrates agent may use best judgment without asking.")
    else:
        lines.append("- **None**: User wants to be asked about all decisions")

    lines.extend([
        "",
        "---",
        "",
        "## Phase 2: Dialectic Specification Status",
        "",
        "- [ ] Socrates analysis complete",
        "- [ ] All blocking clarifications resolved",
        "- [ ] L2 module decomposition complete",
        "- [ ] L3 structural validators passed",
        "",
        "---",
        "",
        "*This document was generated by GAADP Deterministic Intake.*",
        "*Phase 2 (Dialectic Specification) will extend this with L2/L3 details.*",
    ])

    content = "\n".join(lines)

    # Write to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(content)

    return content


# =============================================================================
# MAIN WIZARD
# =============================================================================

def run_wizard(minimal: bool = False) -> Dict[str, Any]:
    """Run the interactive intake wizard."""

    print_header("GAADP Deterministic Intake Wizard")
    print("This wizard collects system-level requirements for your project.")
    print("Your answers will be saved to prompt.md for the Socrates agent.\n")
    print(f"{Colors.RED}*{Colors.END} = Required field")
    print(f"{Colors.DIM}Press Ctrl+C at any time to cancel.{Colors.END}\n")

    answers = {}

    # L1 Questions
    print_header("Phase 1: System-Level Questions (L1)")

    for q in L1_QUESTIONS:
        if minimal and not q.get("required"):
            continue

        print_question(q)
        answers[q["id"]] = get_input(q)
        print()

    # Terminal Questions
    print_header("Feature List & Final Questions")

    for q in TERMINAL_QUESTIONS:
        if minimal and not q.get("required"):
            continue

        print_question(q)

        if q.get("list_input"):
            answers[q["id"]] = get_list_input(q)
        else:
            answers[q["id"]] = get_input(q)
        print()

    return answers


def load_from_json(json_path: str) -> Dict[str, Any]:
    """Load answers from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def save_to_json(answers: Dict[str, Any], json_path: str):
    """Save answers to a JSON file."""
    with open(json_path, 'w') as f:
        json.dump(answers, f, indent=2)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GAADP Deterministic Intake Wizard - Collect requirements and generate prompt.md"
    )
    parser.add_argument(
        "--output", "-o",
        default="prompt.md",
        help="Output file path (default: prompt.md)"
    )
    parser.add_argument(
        "--json", "-j",
        help="Import answers from JSON file instead of interactive mode"
    )
    parser.add_argument(
        "--save-json",
        help="Also save answers to JSON file for reuse"
    )
    parser.add_argument(
        "--minimal", "-m",
        action="store_true",
        help="Skip optional questions"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output except errors"
    )

    args = parser.parse_args()

    try:
        if args.json:
            # Load from JSON
            answers = load_from_json(args.json)
            if not args.quiet:
                print(f"Loaded answers from {args.json}")
        else:
            # Interactive mode
            answers = run_wizard(minimal=args.minimal)

        # Generate prompt.md
        content = generate_prompt_md(answers, args.output)

        if not args.quiet:
            print_header("Summary")
            print(f"{Colors.GREEN}✓{Colors.END} Generated: {args.output}")
            print(f"  - {len([a for a in answers.values() if a])} questions answered")
            print(f"  - {len(answers.get('T1', []))} features listed")

        # Optionally save JSON
        if args.save_json:
            save_to_json(answers, args.save_json)
            if not args.quiet:
                print(f"{Colors.GREEN}✓{Colors.END} Saved JSON: {args.save_json}")

        if not args.quiet:
            print(f"\n{Colors.CYAN}Next step:{Colors.END} Run the Socrates agent to analyze gaps:")
            print(f"  python gaadp_main.py --prompt {args.output}")

    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Wizard cancelled.{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.END}")
        sys.exit(1)


if __name__ == "__main__":
    main()
