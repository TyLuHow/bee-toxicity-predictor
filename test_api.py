#!/usr/bin/env python3
"""Quick test of the FastAPI backend."""
import requests
import json

# Test data
test_compound = {
    "source": "PPDB",
    "year": 2020,
    "toxicity_type": "Contact",
    "herbicide": 0,
    "fungicide": 0,
    "insecticide": 1,
    "other_agrochemical": 0,
    "MolecularWeight": 350.0,
    "LogP": 3.5,
    "NumHDonors": 2,
    "NumHAcceptors": 4,
    "NumRotatableBonds": 5,
    "NumAromaticRings": 1,
    "TPSA": 70.0,
    "NumHeteroatoms": 5,
    "NumRings": 2,
    "NumSaturatedRings": 0,
    "NumAliphaticRings": 1,
    "FractionCSP3": 0.4,
    "MolarRefractivity": 95.0,
    "BertzCT": 500.0,
    "HeavyAtomCount": 25
}

print("Testing FastAPI endpoints...")
print("="*80)

BASE_URL = "http://localhost:8000"

try:
    # Test health endpoint
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    print(f"\n1. Health Check: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    # Test prediction endpoint
    response = requests.post(f"{BASE_URL}/predict", json=test_compound, timeout=5)
    print(f"\n2. Prediction: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    # Test model info endpoint
    response = requests.get(f"{BASE_URL}/model/info", timeout=5)
    print(f"\n3. Model Info: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    
except requests.exceptions.ConnectionError:
    print("\n✗ Could not connect to API. Make sure the server is running:")
    print("  python app/backend/main.py")
except Exception as e:
    print(f"\n✗ Test failed: {e}")

