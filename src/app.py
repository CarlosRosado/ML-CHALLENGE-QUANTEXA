from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import yaml
from prometheus_client import start_http_server, Summary, Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from fastapi.openapi.utils import get_openapi
import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model import Model, load_dataset

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model path
model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')

app = FastAPI()

# Create a metric to track time spent and requests made.
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
REQUEST_COUNT = Counter('request_count', 'Total number of requests')


class ApiRequest(BaseModel):
    text: str

class ApiResponse(BaseModel):
    label: str

# Load the trained model
model = Model()
model.load(model_path)

@app.post("/prediction", response_model=ApiResponse, operation_id="prediction")
@REQUEST_TIME.time()
@REQUEST_COUNT.count_exceptions()
def predict(input: ApiRequest):
    """
    Endpoint for making predictions.

    Args:
        input (ApiRequest): The input text for which to make a prediction.

    Returns:
        ApiResponse: The predicted label for the input text.
    """
    try:
        prediction = model.predict([{"text": input.text}])[0]
        return ApiResponse(label=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    """
    Endpoint for returning a welcome message.

    Returns:
        dict: A welcome message.
    """
    return {"message": "Welcome to the Quantexa Text Classifier API"}

# Load OpenAPI specification from YAML file
with open("src/prediction-openapi.yaml", "r") as f:
    openapi_spec = yaml.safe_load(f)

@app.get("/specifications")
def get_specifications():
    """
    Endpoint for returning the OpenAPI specifications.

    Returns:
        dict: The OpenAPI specifications.
    """
    return openapi_spec

@app.get("/metrics")
def metrics():
    """
    Endpoint for returning the current metrics of the service.

    Returns:
        Response: The current metrics in Prometheus format.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

def custom_openapi():
    """
    Function for customizing the OpenAPI schema.

    Returns:
        dict: The customized OpenAPI schema.
    """
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=openapi_spec["info"]["title"],
        version=openapi_spec["info"]["version"],
        description=openapi_spec["info"]["description"],
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Start up the server to expose the metrics.
start_http_server(9090)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, workers=4)