"""
BASE AGENT (Hardened)
Integrates Runtime Signing, File Locks, and LLM Gateway.
"""
import json
import logging
import yaml
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Any
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from infrastructure.llm_gateway import LLMGateway
from core.ontology import AgentRole

class BaseAgent(ABC):
    def __init__(self, agent_id: str, role: AgentRole, graph_db):
        self.agent_id = agent_id
        self.role = role
        self.graph_db = graph_db
        self.gateway = LLMGateway()
        self.logger = logging.getLogger(f"Agent.{role}.{agent_id}")
        self._private_key = ed25519.Ed25519PrivateKey.generate()
        self._public_key = self._private_key.public_key()
        self._save_keys()

        with open(".blueprint/prompt_templates.yaml", "r") as f:
            self.templates = yaml.safe_load(f)
        with open(".blueprint/prime_directives.md", "r") as f:
            self.directives = f.read()

    def _save_keys(self):
        key_dir = ".gaadp/keys"
        os.makedirs(key_dir, exist_ok=True)
        pub_key_path = f"{key_dir}/{self.agent_id}.pub"
        pem = self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        with open(pub_key_path, "wb") as f:
            f.write(pem)

    def acquire_lock(self, node_id: str, timeout: int = 10) -> bool:
        lock_dir = ".gaadp/locks"
        os.makedirs(lock_dir, exist_ok=True)
        lock_file = f"{lock_dir}/{node_id}.lock"
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with open(lock_file, "x") as f:
                    f.write(self.agent_id)
                return True
            except FileExistsError:
                if os.path.getmtime(lock_file) < time.time() - 30:
                    os.remove(lock_file)
                    continue
                time.sleep(0.1)
        return False

    def release_lock(self, node_id: str):
        lock_file = f".gaadp/locks/{node_id}.lock"
        if os.path.exists(lock_file):
            os.remove(lock_file)

    def sign_content(self, content: Any, previous_hash: str = "GENESIS") -> str:
        """
        Cryptographically signs data linked to history.
        Creates a Merkle-like chain of custody.
        """
        payload = {
            "content": content,
            "prev_hash": previous_hash,
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
        # Sort keys for deterministic hashing
        data_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
        signature = self._private_key.sign(data_bytes).hex()
        return signature

    def _hydrate_prompt(self, template_id: str, vars: Dict) -> str:
        template = self.templates.get(template_id, {}).get('instruction', "")
        vars['prime_directives_text'] = self.directives
        vars['agent_role'] = self.role.value
        vars['agent_id'] = self.agent_id
        return template.format(**vars)

    def _parse_json_response(self, response_text: str) -> Dict:
        try:
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except json.JSONDecodeError:
            raise ValueError("LLM did not return valid JSON")

    @abstractmethod
    async def process(self, context: Dict) -> Dict:
        pass
