# GAADP PROJECT STRUCTURE (v2.0)
GAADP/
├── .blueprint/                 # LEGISLATIVE (Config & Rules)
│   ├── global_ontology.md      # The Source of Truth
│   ├── prime_directives.md     # The 17 Laws
│   └── *.yaml                  # Behavioral Configs
├── core/                       # NERVOUS SYSTEM (Types & Logic)
│   ├── ontology.py             # Python Enums
│   └── cpg_builder.py          # Code -> Graph Parser
├── infrastructure/             # INFRASTRUCTURE (Storage & State)
│   ├── graph_db.py             # NetworkX Wrapper with Traversal Logic
│   └── llm_gateway.py          # Production LLM Connection
├── agents/                     # EXECUTIVE (The Workers)
│   ├── base_agent.py           # Runtime Signing & Event Loop
│   ├── concrete_agents.py      # Real Agent Roles
├── orchestration/              # LOGIC
│   └── wavefront.py            # Parallel Execution
├── requirements/               # INPUT
│   └── socratic_engine.py      # Ambiguity Resolution
├── simulation/                 # DRY RUN
│   ├── mock_agents.py
├── validate_blueprint.py       # THE POLICE (Integrity Checker)
├── validate_phase.py           # THE SMOKE TEST
├── simulate.py                 # THE DRY RUN
└── production_main.py          # THE REAL DEAL
