from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello UrFU"}
        
def test_read_predict_positive():
    response = client.post("/predict/",
        json={"text": "I love donuts!"}
    )
    json_data = response.json() 

    assert response.status_code == 200
    assert json_data['label'] == 'POSITIVE'

def test_read_predict_negative():
    response = client.post("/predict/",
        json={"text": "I hate machine learning!"}
    )
    json_data = response.json() 

    assert response.status_code == 200
    assert json_data['label'] == 'NEGATIVE'
                               
def test_read_predict_neutral():
    response = client.post("/predict/",
        json={"text": "I nevermind"}
    )
    json_data = response.json() 

    assert response.status_code == 200
    assert json_data['label'] == 'NEUTRAL'
