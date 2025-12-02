import os
import sys
import anthropic
from infrastructure.llm_providers import AnthropicAPIProvider

def main():
    print("============================================================")
    print(f"  ANTHROPIC MODEL SCANNER (SDK v{anthropic.__version__})")
    print("============================================================")

    # 1. Check Key
    key = os.getenv("ANTHROPIC_API_KEY", "")
    print(f"1. Key Length: {len(key)}")
    
    # 2. Candidate Models (Including older/lighter ones)
    candidates = [
        "claude-3-5-haiku-20241022",   # Newest Fast Model
        "claude-3-haiku-20240307",     # Previous Fast Model (Most likely to work)
        "claude-3-5-sonnet-latest",    # Re-try Sonnet
        "claude-3-sonnet-20240229",    # Old Sonnet
    ]

    provider = AnthropicAPIProvider()
    working_model = None

    for model in candidates:
        print(f"\n   Trying: {model} ...")
        try:
            response = provider.call(
                system_prompt="Test.",
                user_prompt="Reply 'OK'.",
                model_config={"model": model, "max_tokens": 10}
            )
            print(f"   ✅ SUCCESS! Response: {response.strip()}")
            working_model = model
            break
        except Exception as e:
            if "not_found_error" in str(e):
                print("   ❌ Not Found (404) - Account restricted")
            elif "credit balance" in str(e):
                print("   ❌ Billing Error (400)")
            else:
                print(f"   ❌ Error: {e}")

    if working_model:
        print("\n============================================================")
        print(f"  🎉 WINNER: {working_model}")
        print("============================================================")
        # Generate the patch command dynamically
        print("\nTo use this model, update infrastructure/llm_gateway.py with:")
        print(f'   "anthropic_api": {{ "heavy": "{working_model}", "standard": "{working_model}", "legacy": "{working_model}" }}')
    else:
        print("\n❌ ALL MODELS FAILED. Your account may be in 'Provisioning Pending' state.")
        print("   Wait 15 minutes or contact Anthropic support.")

if __name__ == "__main__":
    main()
