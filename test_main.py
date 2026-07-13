from fastapi.testclient import TestClient
from main import app, risk_tier, THRESHOLDS

client = TestClient(app)

KNOWN_NORMAL_TRANSACTION = {
  "V1": 1.31453919005614, "V2": 0.590642763250075, "V3": -0.66659342070355, "V4": 0.716564340123453, "V5": 0.301977869959298, "V6": -1.12546746514505, "V7": 0.388880528388934, "V8": -0.288389638013591, "V9": -0.132137450666675, "V10": -0.597738927671465, "V11": -0.325346640449052, "V12": -0.21643537583917, "V13": 0.0842040799280351, "V14": -1.0546305617385, "V15": 0.967932032509217, "V16": 0.601226266062646, "V17": 0.631116638308353, "V18": 0.29507796789341, "V19": -0.136150702482415, "V20": -0.0580404050390117, "V21": -0.170307279387637, "V22": -0.429655035253663, "V23": -0.141340683112249, "V24": -0.200194626075649, "V25": 0.639491494374774, "V26": 0.399475692421549, "V27": -0.0343210089575542, "V28": 0.0316924132520412, "Amount": 0.76
}

EXPECTED_NORMAL_SCORE = 0.35835575467597836

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status" : "ok", "model_loaded" : True}
    
def test_predict_matches_known_score():
    response = client.post("/predict", json=KNOWN_NORMAL_TRANSACTION)
    assert response.status_code == 200
    data = response.json()
    assert data["anomaly_score"] == EXPECTED_NORMAL_SCORE
    assert data["risk_tier"] == "low"
    
def test_predict_rejects_missing_field():
    incomplete = {k : v for k, v in KNOWN_NORMAL_TRANSACTION.items() if k != "V1" }
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422
    
def test_predict_rejects_wrong_type():
    bad_payload = dict(KNOWN_NORMAL_TRANSACTION)
    bad_payload["V1"] = "not_a_number" # type: ignore
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    
def test_risk_tier_boundaries():
    assert risk_tier(THRESHOLDS["critical"]) == "critical"
    assert risk_tier(THRESHOLDS["critical"] - 0.0001) != "critical"
    assert risk_tier(THRESHOLDS["high"]) == "high"
    assert risk_tier(0.0) == "low"