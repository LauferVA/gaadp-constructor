"""
DOMAIN DISCOVERY ENGINE
Dynamically researches and injects industry norms.
"""
import logging
import os
from typing import Optional
from infrastructure.llm_gateway import LLMGateway

logger = logging.getLogger("DomainDiscovery")


class DomainResearcher:
    """
    Identifies the domain of a user request and researches
    industry-specific coding standards to inject as Prime Directives.
    """

    def __init__(self, gateway: Optional[LLMGateway] = None):
        self.gateway = gateway or LLMGateway()

    def run_discovery(self, user_request: str, interactive: bool = True) -> dict:
        """
        Run the full discovery pipeline:
        1. Identify Domain
        2. Research Norms
        3. Confirm with User (if interactive)
        4. Inject into Prime Directives

        Returns dict with domain, norms, and injection status.
        """
        print("\n🌍 PHASE 0: CONTEXTUAL GROUNDING")

        # Step 1: Identify Domain
        domain = self._identify_domain(user_request)
        print(f"   🎯 Target Domain detected: [{domain}]")

        if interactive:
            confirm = input("   > Is this correct? (Y/n): ").strip().lower()
            if confirm not in ['', 'y', 'yes']:
                domain = input("   > Enter Domain: ").strip()

        # Step 2: Research Norms
        print(f"   📡 Researching Coding Standards for {domain}...")
        norms = self._research_norms(domain)

        # Step 3: User Confirmation
        print(f"\n   ⚖️  PROPOSED LAWS for {domain}:")
        print(norms)

        injected = False
        if interactive:
            confirm = input("\n   > Adopt these laws? (Y/n): ").strip().lower()
            if confirm in ['', 'y', 'yes']:
                self._inject_directives(domain, norms)
                print("   ✅ Laws Injected into Prime Directives.")
                injected = True
        else:
            # Non-interactive mode: auto-inject
            self._inject_directives(domain, norms)
            injected = True

        return {
            "domain": domain,
            "norms": norms,
            "injected": injected
        }

    def _identify_domain(self, user_request: str) -> str:
        """Use LLM to identify the domain/industry of the request."""
        try:
            domain_prompt = f"""Identify the specific Industry/Field for this software request:
'{user_request}'

Return ONLY the domain name (e.g., 'FinTech', 'Bioinformatics', 'E-Commerce', 'DevOps').
One or two words maximum."""

            domain = self.gateway.call_model(
                "ARCHITECT",
                "You are a software domain classifier. Be precise and concise.",
                domain_prompt
            ).strip().strip('"\'')

            return domain
        except Exception as e:
            logger.warning(f"Domain detection failed: {e}")
            return "General Software"

    def _research_norms(self, domain: str) -> str:
        """
        Research industry-specific coding standards.
        In production, this would call a search API (Tavily, SerpAPI).
        Currently uses LLM knowledge as a fallback.
        """
        # Simulated search results - in production, call actual search API
        search_context = self._simulate_search(domain)

        synthesis_prompt = f"""Based on industry knowledge for {domain} software development:

{search_context}

Define 3-5 Critical 'Prime Directives' that are MANDATORY for {domain} software.
Focus on:
- Security requirements specific to {domain}
- Data handling regulations (if applicable)
- Required libraries/frameworks
- Testing standards

Format each as:
- Rule: [Specific constraint or requirement]
"""
        try:
            norms = self.gateway.call_model(
                "ARCHITECT",
                "You are a software standards expert. Define strict, enforceable rules.",
                synthesis_prompt
            )
            return norms
        except Exception as e:
            logger.error(f"Norm synthesis failed: {e}")
            return f"- Rule: Follow {domain} industry best practices\n- Rule: Maintain high test coverage"

    def _simulate_search(self, domain: str) -> str:
        """
        Simulate search results. Replace with actual API call in production.
        """
        domain_knowledge = {
            "Bioinformatics": """
                - Use BioPython for sequence analysis
                - FASTA/FASTQ file handling required
                - Reproducibility is critical (version pinning)
                - Large dataset memory management
                - HIPAA compliance if handling patient data
            """,
            "FinTech": """
                - PCI-DSS compliance mandatory for payment data
                - No hardcoded credentials ever
                - Audit logging required for all transactions
                - Use decimal types for currency (never float)
                - SOC2 compliance considerations
            """,
            "Healthcare": """
                - HIPAA compliance mandatory
                - PHI must be encrypted at rest and in transit
                - Audit trails for all data access
                - Role-based access control required
            """,
            "E-Commerce": """
                - PCI-DSS for payment processing
                - GDPR compliance for EU customers
                - Rate limiting on APIs
                - Input validation on all user data
            """,
            "DevOps": """
                - Infrastructure as Code principles
                - Secrets management (no hardcoded credentials)
                - Idempotent operations
                - Comprehensive logging and monitoring
            """
        }

        return domain_knowledge.get(domain, f"""
            - Follow {domain} industry standard practices
            - Implement comprehensive error handling
            - Write unit tests for all business logic
            - Document public APIs
        """)

    def _inject_directives(self, domain: str, norms: str):
        """Append domain-specific directives to prime_directives.md"""
        directives_path = ".blueprint/prime_directives.md"

        # Check if domain section already exists
        if os.path.exists(directives_path):
            with open(directives_path, "r") as f:
                content = f.read()
            if f"## VI. DOMAIN LAWS ({domain})" in content:
                logger.info(f"Domain {domain} already has injected laws, skipping.")
                return

        with open(directives_path, "a") as f:
            f.write(f"\n\n## VI. DOMAIN LAWS ({domain})\n")
            f.write(f"> Auto-discovered for {domain} domain\n\n")
            f.write(norms)
            f.write("\n")

        logger.info(f"Injected {domain} laws into {directives_path}")


def quick_discover(request: str) -> str:
    """Quick helper to get domain without full pipeline."""
    researcher = DomainResearcher()
    return researcher._identify_domain(request)
