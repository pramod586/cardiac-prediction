import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../app'))
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    rv = client.get('/')
    assert rv.status_code == 200
    assert b'Cardiology Risk Predictor' in rv.data

def test_predict_valid_input(client):
    valid_data = {
        'age': '50', 'anaemia': '0', 'creatinine_phosphokinase': '250',
        'diabetes': '0', 'ejection_fraction': '60', 'high_blood_pressure': '0',
        'platelets': '265000', 'serum_creatinine': '1.0', 'serum_sodium': '136',
        'sex': '1', 'smoking': '0', 'time': '200'
    }
    rv = client.post('/predict', data=valid_data)
    assert rv.status_code == 200
    assert b'Assessment Complete' in rv.data

def test_predict_missing_fields(client):
    missing_data = { 'age': '50', 'anaemia': '0' }
    rv = client.post('/predict', data=missing_data)
    assert rv.status_code == 200
    assert b'One or more fields are missing' in rv.data

def test_predict_invalid_values(client):
    invalid_data = {
        'age': 'invalid_string', 'anaemia': '0', 'creatinine_phosphokinase': '250',
        'diabetes': '0', 'ejection_fraction': '60', 'high_blood_pressure': '0',
        'platelets': '265000', 'serum_creatinine': '1.0', 'serum_sodium': '136',
        'sex': '1', 'smoking': '0', 'time': '200'
    }
    rv = client.post('/predict', data=invalid_data)
    assert rv.status_code == 200
    assert b'One or more fields are missing' in rv.data or b'valid numerical data' in rv.data
