from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_query_yields_10_results():
    response = client.get("/query?query=covid is a hoax")
    json_response = response.json()
    
    assert response.status_code == 200
    assert len(json_response["results"]) == 10
    assert json_response["message"] == "OK"

def test_query_yields_few_results():
    response = client.get("/query?query=covid and h1n1")
    json_response = response.json()
    
    assert response.status_code == 200
    assert 1 < len(json_response["results"]) < 10
    assert json_response["message"] == "OK"

def test_query_yields_non_obvious_results():
    response = client.get("/query?query=public transport")
    json_response = response.json()
        
    assert response.status_code == 200
    assert json_response["message"] == "OK"
