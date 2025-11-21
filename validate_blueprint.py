#!/usr/bin/env python3
"""
GAADP ARCHITECTURE VALIDATOR
"""
import sys
import os
import yaml
import importlib.util

sys.path.append(os.getcwd())

def load_ontology_class():
    try:
        spec = importlib.util.spec_from_file_location("core.ontology", "core/ontology.py")
        ontology = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ontology)
        return ontology
    except Exception as e:
        print(f"CRITICAL: Could not import core.ontology: {e}")
        sys.exit(1)

def validate_configs(ontology):
    print("\nValidating Configuration vs Ontology...")
    try:
        with open(".blueprint/topology_config.yaml") as f:
            topo = yaml.safe_load(f)
        for role in topo['ratios']:
            if role != "GLOBAL" and not hasattr(ontology.AgentRole, role):
                print(f"  Invalid Role: {role}")
            else:
                print(f"  Valid Role: {role}")
    except FileNotFoundError:
        print("  topology_config.yaml missing")

def validate_structure():
    print("\nValidating File Structure...")
    required = [".blueprint/global_ontology.md", ".blueprint/prime_directives.md", "core/ontology.py", "infrastructure/graph_db.py"]
    all_passed = True
    for f in required:
        if os.path.exists(f):
            print(f"  Found: {f}")
        else:
            print(f"  MISSING: {f}")
            all_passed = False
    return all_passed

def main():
    print("GAADP SYSTEM INTEGRITY CHECK")
    ontology = load_ontology_class()
    structure_ok = validate_structure()
    validate_configs(ontology)
    if structure_ok:
        print("\nSYSTEM HARMONIZED.")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
