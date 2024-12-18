import pytest
from fastapi.testclient import TestClient
import sys, os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from app import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Quantexa Text Classifier API"}

def test_predict():
    response = client.post(
        "/prediction",
        json={
            "text": "From: palmer@cco.caltech.edu (David M. Palmer)\nSubject: Re: HST Servicing Mission Scheduled for 11 Days\nOrganization: California Institute of Technology, Pasadena\nLines: 30\nNNTP-Posting-Host: alumni.caltech.edu\n\nprb@access.digex.net (Pat) writes:\n\n>In article <C6A2At.E9z@zoo.toronto.edu> henry@zoo.toronto.edu (Henry Spencer) writes:\n>>\n>>No, the thing is designed to be retrievable, in a pinch.  Indeed, this\n>>dictated a rather odd design for the solar arrays, since they had to be\n>>retractable as well as extendable, and may thus have indirectly contributed\n>>to the array-flapping problems.\n\n\n>Why not design the solar arrays to be detachable.  if the shuttle is going\n>to return the HST,  what bother are some arrays.  just fit them with a quick\n> release.  one  space walk,  or use the second canadarm to remove the arrays.\n\nYou may want to put Hubble back in the payload bay for a reboost,\nand you don\'t want to clip off the panels each time.\n\nFor the Gamma-Ray Observatory, one of the design requirements was that\nthere be no stored-energy mechanisms (springs, explosive squibs, gas shocks,\netc.) used for deployment.  This was partially so that everything could\nbe reeled back in to put it back in the payload bay, and partially for\nsafety considerations.  (I\'ve heard that the wings on a cruise missile\nwould cut you in half if you were standing in their swath when they opened.)\n\nBack when the shuttle would be going up every other day with a cost to\norbit of $3.95 per pound :-), everybody designed things for easy servicing.\n\n-- \n\t\tDavid M. Palmer\t\tpalmer@alumni.caltech.edu\n\t\t\t\t\tpalmer@tgrs.gsfc.nasa.gov\n"
        },
    )
    assert response.status_code == 200
    assert "label" in response.json()

def test_invalid_input():
    response = client.post(
        "/prediction",
        json={"invalid_field": "This is invalid input"}
    )
    assert response.status_code == 422

def test_missing_text_field():
    response = client.post(
        "/prediction",
        json={}
    )
    assert response.status_code == 422

def test_empty_text_field():
    response = client.post(
        "/prediction",
        json={"text": ""}
    )
    assert response.status_code == 200
    assert "label" in response.json()

def test_get_specifications():
    response = client.get("/specifications")
    assert response.status_code == 200
    assert "info" in response.json()
    assert "paths" in response.json()

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "request_processing_seconds" in response.text
    assert "request_count" in response.text
