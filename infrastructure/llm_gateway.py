"""
GAADP LLM GATEWAY
"""
import yaml
import logging
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger("LLM_Gateway")

class LLMGateway:
    def __init__(self, config_path: str = ".blueprint/llm_router.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self._cost_session = 0.0

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def call_model(self, role: str, system_prompt: str, user_context: str) -> str:
        role_config = self.config['model_assignments'].get(role)
        if not role_config:
            raise ValueError(f"Role {role} not defined in llm_router.yaml")

        try:
            response = completion(
                model=role_config['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_context}
                ],
                temperature=role_config['temperature'],
                max_tokens=role_config.get('max_tokens', 4000)
            )
        except Exception as e:
            logger.error(f"API Failure: {e}")
            raise e

        cost = response._hidden_params.get("response_cost", 0.0)
        self._cost_session += cost
        logger.info(f"Call Cost: ${cost:.4f} | Total: ${self._cost_session:.4f}")
        return response.choices[0].message.content
