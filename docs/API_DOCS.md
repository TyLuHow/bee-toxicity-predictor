# API Documentation: Honey Bee Toxicity Prediction

## Overview

The Honey Bee Toxicity Prediction API is a RESTful web service built with FastAPI that serves machine learning predictions for pesticide toxicity to honey bees. The API provides endpoints for making predictions, retrieving model information, accessing feature importance data, and viewing prediction history.

**Base URL**: `http://localhost:8000`  
**API Version**: 1.0.0  
**Protocol**: HTTP/HTTPS  
**Data Format**: JSON

---

## Quick Start

### 1. Start the API Server

```bash
# Option 1: Direct Python
python app/backend/main.py

# Option 2: Using uvicorn
uvicorn app.backend.main:app --reload --port 8000

# Option 3: Docker
docker-compose up backend
```

### 2. Access Interactive Documentation

Open your browser to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 3. Make Your First Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "PPDB",
    "year": 2020,
    "toxicity_type": "Contact",
    "herbicide": 0,
    "fungicide": 0,
    "insecticide": 1,
    "other_agrochemical": 0,
    "MolecularWeight": 350.5,
    "LogP": 3.2,
    "NumHDonors": 2,
    "NumHAcceptors": 4,
    "NumRotatableBonds": 5,
    "AromaticRings": 2,
    "TPSA": 65.3,
    "NumHeteroatoms": 5,
    "NumAromaticAtoms": 12,
    "NumSaturatedRings": 0,
    "NumAliphaticRings": 0,
    "RingCount": 2,
    "FractionCsp3": 0.25,
    "NumAromaticCarbocycles": 1,
    "NumSaturatedCarbocycles": 0
  }'
```

---

## Authentication

**Current Version**: No authentication required  
**Production Recommendation**: Implement API keys or OAuth2 for production deployments

---

## Endpoints

## 1. Health Check

Check if the API is running and healthy.

### Request

```http
GET /health
```

### Response

**Status Code**: `200 OK`

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "timestamp": "2025-11-07T10:30:00Z"
}
```

### Example

```bash
curl -X GET "http://localhost:8000/health"
```

```python
import requests

response = requests.get("http://localhost:8000/health")
print(response.json())
```

---

## 2. Make Prediction

Predict toxicity for a pesticide compound based on its molecular descriptors and properties.

### Request

```http
POST /predict
Content-Type: application/json
```

### Input Schema

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `source` | string | Yes | - | Data source (ECOTOX, PPDB, BPDB, or other) |
| `year` | integer | Yes | 1800-2030 | Publication year |
| `toxicity_type` | string | Yes | - | Toxicity test type (Contact, Oral, Other) |
| `herbicide` | integer | Yes | 0 or 1 | Is herbicide flag |
| `fungicide` | integer | Yes | 0 or 1 | Is fungicide flag |
| `insecticide` | integer | Yes | 0 or 1 | Is insecticide flag |
| `other_agrochemical` | integer | Yes | 0 or 1 | Is other agrochemical flag |
| `MolecularWeight` | float | Yes | ≥ 0 | Molecular weight (g/mol) |
| `LogP` | float | Yes | - | Partition coefficient (lipophilicity) |
| `NumHDonors` | integer | Yes | ≥ 0 | Number of hydrogen bond donors |
| `NumHAcceptors` | integer | Yes | ≥ 0 | Number of hydrogen bond acceptors |
| `NumRotatableBonds` | integer | Yes | ≥ 0 | Number of rotatable bonds |
| `AromaticRings` | integer | Yes | ≥ 0 | Number of aromatic rings |
| `TPSA` | float | Yes | ≥ 0 | Topological polar surface area (Ų) |
| `NumHeteroatoms` | integer | Yes | ≥ 0 | Number of heteroatoms |
| `NumAromaticAtoms` | integer | Yes | ≥ 0 | Number of aromatic atoms |
| `NumSaturatedRings` | integer | Yes | ≥ 0 | Number of saturated rings |
| `NumAliphaticRings` | integer | Yes | ≥ 0 | Number of aliphatic rings |
| `RingCount` | integer | Yes | ≥ 0 | Total ring count |
| `FractionCsp3` | float | Yes | 0-1 | Fraction of sp³ carbons |
| `NumAromaticCarbocycles` | integer | Yes | ≥ 0 | Number of aromatic carbocycles |
| `NumSaturatedCarbocycles` | integer | Yes | ≥ 0 | Number of saturated carbocycles |

### Response

**Status Code**: `200 OK`

```json
{
  "prediction": 1,
  "probability_toxic": 0.8745,
  "probability_non_toxic": 0.1255,
  "confidence": 0.8745,
  "label_text": "Toxic",
  "timestamp": "2025-11-07T10:30:00Z",
  "input_features": {
    "source": "PPDB",
    "insecticide": 1,
    "MolecularWeight": 350.5,
    ...
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | integer | Binary class (0=Non-Toxic, 1=Toxic) |
| `probability_toxic` | float | Probability of toxic class [0-1] |
| `probability_non_toxic` | float | Probability of non-toxic class [0-1] |
| `confidence` | float | Maximum probability (confidence level) |
| `label_text` | string | Human-readable label ("Toxic" or "Non-Toxic") |
| `timestamp` | string | ISO 8601 timestamp of prediction |
| `input_features` | object | Echo of input features (optional) |

### Error Responses

**Status Code**: `422 Unprocessable Entity`

```json
{
  "detail": [
    {
      "loc": ["body", "MolecularWeight"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### Examples

#### Python

```python
import requests

# Prepare input data
data = {
    "source": "PPDB",
    "year": 2020,
    "toxicity_type": "Contact",
    "herbicide": 0,
    "fungicide": 0,
    "insecticide": 1,
    "other_agrochemical": 0,
    "MolecularWeight": 350.5,
    "LogP": 3.2,
    "NumHDonors": 2,
    "NumHAcceptors": 4,
    "NumRotatableBonds": 5,
    "AromaticRings": 2,
    "TPSA": 65.3,
    "NumHeteroatoms": 5,
    "NumAromaticAtoms": 12,
    "NumSaturatedRings": 0,
    "NumAliphaticRings": 0,
    "RingCount": 2,
    "FractionCsp3": 0.25,
    "NumAromaticCarbocycles": 1,
    "NumSaturatedCarbocycles": 0
}

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json=data
)

# Parse response
result = response.json()
print(f"Prediction: {result['label_text']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Toxic Probability: {result['probability_toxic']:.2%}")
```

#### JavaScript

```javascript
const data = {
  source: "PPDB",
  year: 2020,
  toxicity_type: "Contact",
  herbicide: 0,
  fungicide: 0,
  insecticide: 1,
  other_agrochemical: 0,
  MolecularWeight: 350.5,
  LogP: 3.2,
  NumHDonors: 2,
  NumHAcceptors: 4,
  NumRotatableBonds: 5,
  AromaticRings: 2,
  TPSA: 65.3,
  NumHeteroatoms: 5,
  NumAromaticAtoms: 12,
  NumSaturatedRings: 0,
  NumAliphaticRings: 0,
  RingCount: 2,
  FractionCsp3: 0.25,
  NumAromaticCarbocycles: 1,
  NumSaturatedCarbocycles: 0
};

fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify(data)
})
  .then(response => response.json())
  .then(result => {
    console.log(`Prediction: ${result.label_text}`);
    console.log(`Confidence: ${(result.confidence * 100).toFixed(2)}%`);
  });
```

#### R

```r
library(httr)
library(jsonlite)

# Prepare data
data <- list(
  source = "PPDB",
  year = 2020,
  toxicity_type = "Contact",
  herbicide = 0,
  fungicide = 0,
  insecticide = 1,
  other_agrochemical = 0,
  MolecularWeight = 350.5,
  LogP = 3.2,
  NumHDonors = 2,
  NumHAcceptors = 4,
  NumRotatableBonds = 5,
  AromaticRings = 2,
  TPSA = 65.3,
  NumHeteroatoms = 5,
  NumAromaticAtoms = 12,
  NumSaturatedRings = 0,
  NumAliphaticRings = 0,
  RingCount = 2,
  FractionCsp3 = 0.25,
  NumAromaticCarbocycles = 1,
  NumSaturatedCarbocycles = 0
)

# Make prediction
response <- POST(
  "http://localhost:8000/predict",
  body = data,
  encode = "json"
)

# Parse response
result <- content(response, as = "parsed")
cat("Prediction:", result$label_text, "\n")
cat("Confidence:", sprintf("%.2f%%", result$confidence * 100), "\n")
```

---

## 3. Get Model Information

Retrieve metadata and performance metrics about the deployed model.

### Request

```http
GET /model/info
```

### Response

**Status Code**: `200 OK`

```json
{
  "model_name": "XGBoost Classifier",
  "version": "1.0.0",
  "algorithm": "XGBoost",
  "n_features": 24,
  "feature_names": ["source_BPDB", "source_ECOTOX", ...],
  "metrics": {
    "val_accuracy": 0.8558,
    "val_precision": 0.7273,
    "val_recall": 0.6780,
    "val_f1": 0.7368,
    "val_roc_auc": 0.8788,
    "test_accuracy": 0.8357,
    "test_f1": 0.7018,
    "test_roc_auc": 0.8583
  },
  "training_date": "2025-11-07",
  "dataset_size": 1035,
  "class_balance": {
    "non_toxic": 739,
    "toxic": 296
  }
}
```

### Example

```bash
curl -X GET "http://localhost:8000/model/info"
```

```python
import requests

response = requests.get("http://localhost:8000/model/info")
model_info = response.json()

print(f"Model: {model_info['model_name']}")
print(f"Test Accuracy: {model_info['metrics']['test_accuracy']:.2%}")
print(f"Test ROC-AUC: {model_info['metrics']['test_roc_auc']:.2%}")
```

---

## 4. Get Feature Importance

Retrieve global feature importance scores from SHAP analysis.

### Request

```http
GET /feature/importance
```

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_n` | integer | 10 | Number of top features to return |

### Response

**Status Code**: `200 OK`

```json
{
  "feature_importance": [
    {
      "feature": "insecticide",
      "importance": 1.366,
      "rank": 1
    },
    {
      "feature": "herbicide",
      "importance": 1.054,
      "rank": 2
    },
    {
      "feature": "fungicide",
      "importance": 0.740,
      "rank": 3
    },
    ...
  ],
  "method": "SHAP",
  "total_features": 24
}
```

### Example

```bash
curl -X GET "http://localhost:8000/feature/importance?top_n=5"
```

```python
import requests

response = requests.get(
    "http://localhost:8000/feature/importance",
    params={"top_n": 5}
)

importance_data = response.json()
for item in importance_data['feature_importance']:
    print(f"{item['rank']}. {item['feature']}: {item['importance']:.3f}")
```

---

## 5. Get Prediction History

Retrieve recent prediction history (for demonstration and monitoring).

### Request

```http
GET /history
```

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 10 | Maximum number of records to return |

### Response

**Status Code**: `200 OK`

```json
{
  "history": [
    {
      "timestamp": "2025-11-07T10:30:00Z",
      "prediction": 1,
      "label_text": "Toxic",
      "confidence": 0.8745,
      "features": {
        "insecticide": 1,
        "MolecularWeight": 350.5,
        ...
      }
    },
    ...
  ],
  "total_predictions": 142,
  "returned": 10
}
```

### Example

```bash
curl -X GET "http://localhost:8000/history?limit=5"
```

```python
import requests

response = requests.get(
    "http://localhost:8000/history",
    params={"limit": 5}
)

history = response.json()
print(f"Total predictions: {history['total_predictions']}")
for pred in history['history']:
    print(f"{pred['timestamp']}: {pred['label_text']} ({pred['confidence']:.2%})")
```

---

## 6. Get Prediction Explanation (SHAP)

Get detailed SHAP explanation for a specific prediction.

### Request

```http
POST /predict/explain
Content-Type: application/json
```

### Input Schema

Same as `/predict` endpoint.

### Response

**Status Code**: `200 OK`

```json
{
  "prediction": 1,
  "label_text": "Toxic",
  "confidence": 0.8745,
  "shap_values": {
    "insecticide": 0.45,
    "herbicide": 0.12,
    "LogP": 0.23,
    "MolecularWeight": 0.08,
    ...
  },
  "top_contributors": [
    {
      "feature": "insecticide",
      "contribution": 0.45,
      "direction": "positive"
    },
    {
      "feature": "LogP",
      "contribution": 0.23,
      "direction": "positive"
    }
  ],
  "base_value": 0.15,
  "explanation": "Prediction is primarily driven by insecticide flag and high LogP value."
}
```

### Example

```python
import requests

data = {
    # ... same as /predict input ...
}

response = requests.post(
    "http://localhost:8000/predict/explain",
    json=data
)

explanation = response.json()
print(f"Prediction: {explanation['label_text']}")
print(f"Confidence: {explanation['confidence']:.2%}")
print("\nTop Contributors:")
for contrib in explanation['top_contributors']:
    print(f"  {contrib['feature']}: {contrib['contribution']:+.3f} ({contrib['direction']})")
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid request format |
| 422 | Unprocessable Entity | Validation error (invalid input values) |
| 404 | Not Found | Endpoint does not exist |
| 405 | Method Not Allowed | Wrong HTTP method |
| 500 | Internal Server Error | Server-side error |
| 503 | Service Unavailable | Model not loaded or service down |

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong",
  "error_type": "ValidationError",
  "timestamp": "2025-11-07T10:30:00Z"
}
```

### Common Errors

#### Missing Required Field

```json
{
  "detail": [
    {
      "loc": ["body", "MolecularWeight"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

#### Invalid Data Type

```json
{
  "detail": [
    {
      "loc": ["body", "year"],
      "msg": "value is not a valid integer",
      "type": "type_error.integer"
    }
  ]
}
```

#### Out of Range Value

```json
{
  "detail": [
    {
      "loc": ["body", "year"],
      "msg": "ensure this value is greater than or equal to 1800",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

---

## Rate Limiting

**Current Version**: No rate limiting  
**Production Recommendation**: Implement rate limiting (e.g., 100 requests/minute per IP)

---

## CORS Configuration

**Current Setting**: All origins allowed (`*`)  
**Production Recommendation**: Restrict to specific domains

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

---

## Deployment

### Local Development

```bash
uvicorn app.backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -f Dockerfile.backend -t bee-toxicity-api .
docker run -p 8000:8000 bee-toxicity-api
```

### Docker Compose

```bash
docker-compose up backend
```

### Production Considerations

1. **Environment Variables**
   ```bash
   export MODEL_PATH="/app/outputs/models/best_model_xgboost.pkl"
   export PREPROCESSOR_PATH="/app/outputs/preprocessors/preprocessor.pkl"
   export LOG_LEVEL="INFO"
   ```

2. **Logging**
   - Configure structured logging (JSON format)
   - Log all predictions for auditing
   - Monitor error rates and response times

3. **Security**
   - Enable HTTPS (TLS/SSL)
   - Implement API key authentication
   - Add rate limiting
   - Validate all inputs rigorously
   - Sanitize error messages (don't expose internals)

4. **Monitoring**
   - Health checks: `/health` endpoint
   - Metrics: Prometheus/Grafana
   - Alerting: Error rates, latency, downtime

5. **Scaling**
   - Load balancer (nginx, AWS ALB)
   - Multiple API instances
   - Redis cache for frequent requests
   - Database for prediction history

---

## Performance

### Typical Response Times

- `/health`: <10ms
- `/predict`: 50-150ms
- `/model/info`: <20ms
- `/feature/importance`: <30ms
- `/history`: 20-50ms (depending on limit)

### Throughput

- **Single Instance**: ~50-100 requests/second
- **With Load Balancer**: 500+ requests/second

### Resource Usage

- **Memory**: ~500MB (includes model + dependencies)
- **CPU**: Single core sufficient for <100 req/s
- **Disk**: <100MB (model + code)

---

## Versioning

**Current Version**: 1.0.0  
**API Versioning Strategy**: URL-based versioning for future releases

Example for v2:
```
http://localhost:8000/v2/predict
```

### Changelog

**v1.0.0** (November 2025)
- Initial API release
- XGBoost model (83.6% accuracy)
- 6 core endpoints
- SHAP interpretability
- Docker support

---

## Support & Troubleshooting

### Common Issues

#### 1. Model Not Loading

**Symptoms**: 503 Service Unavailable, "Model not loaded" error

**Solutions**:
- Check model file exists: `outputs/models/best_model_xgboost.pkl`
- Check preprocessor exists: `outputs/preprocessors/preprocessor.pkl`
- Verify file permissions
- Check logs for loading errors

#### 2. Validation Errors (422)

**Symptoms**: "field required" or "value is not valid"

**Solutions**:
- Ensure all 24 features are provided
- Check data types (int vs float)
- Verify value ranges (e.g., year 1800-2030)
- Review input schema in this documentation

#### 3. Slow Predictions

**Symptoms**: Response time >500ms

**Solutions**:
- Check CPU usage
- Monitor memory usage
- Consider caching frequent requests
- Scale horizontally (multiple API instances)

### Getting Help

1. **Check Logs**
   ```bash
   # Docker
   docker-compose logs -f backend
   
   # Direct Python
   # Logs printed to stdout
   ```

2. **Interactive Documentation**
   - Visit http://localhost:8000/docs
   - Test endpoints directly in browser
   - View full request/response schemas

3. **Contact**
   - Course: IME 372 - Predictive Analytics
   - Project documentation: `README.md`, `MODEL_CARD.md`

---

## Code Examples

### Batch Predictions

```python
import requests
import pandas as pd

# Load batch input data
df = pd.read_csv("compounds_to_predict.csv")

# Make predictions for each compound
results = []
for _, row in df.iterrows():
    data = row.to_dict()
    response = requests.post("http://localhost:8000/predict", json=data)
    if response.status_code == 200:
        result = response.json()
        results.append({
            'compound_name': row['name'],
            'prediction': result['label_text'],
            'confidence': result['confidence']
        })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("predictions_output.csv", index=False)
```

### Async Requests (Python)

```python
import aiohttp
import asyncio

async def predict_async(session, data):
    async with session.post(
        "http://localhost:8000/predict",
        json=data
    ) as response:
        return await response.json()

async def batch_predict(compounds):
    async with aiohttp.ClientSession() as session:
        tasks = [predict_async(session, compound) for compound in compounds]
        results = await asyncio.gather(*tasks)
        return results

# Usage
compounds = [...]  # List of compound dictionaries
results = asyncio.run(batch_predict(compounds))
```

### Error Handling

```python
import requests
from requests.exceptions import RequestException

def safe_predict(data, max_retries=3):
    """Make prediction with error handling and retries."""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8000/predict",
                json=data,
                timeout=10  # 10 second timeout
            )
            response.raise_for_status()  # Raise exception for 4xx/5xx
            return response.json()
        except RequestException as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {e}")
                return None
            else:
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff

# Usage
result = safe_predict(compound_data)
if result:
    print(f"Prediction: {result['label_text']}")
else:
    print("Prediction failed")
```

---

## Testing

### Unit Tests

```bash
pytest tests/test_api.py -v
```

### Manual Testing

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_input.json

# Model info
curl http://localhost:8000/model/info
```

### Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 -T application/json -p sample_input.json \
  http://localhost:8000/predict

# Using wrk
wrk -t4 -c100 -d30s --latency \
  -s post_predict.lua \
  http://localhost:8000/predict
```

---

## References

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Pydantic**: https://pydantic-docs.helpmanual.io/
- **Uvicorn**: https://www.uvicorn.org/
- **Model Card**: `docs/MODEL_CARD.md`
- **Project README**: `README.md`

---

## License

**API License**: MIT  
**Model License**: CC-BY-NC-4.0 (Non-Commercial Use Only)

---

**Documentation Version**: 1.0.0  
**Last Updated**: November 7, 2025  
**Status**: ✅ Production Ready

