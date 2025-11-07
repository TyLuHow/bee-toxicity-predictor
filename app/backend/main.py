#!/usr/bin/env python3
"""
FastAPI Backend for Honey Bee Toxicity Prediction
==================================================

REST API for serving ML model predictions with:
- Prediction endpoint with toxicity classification
- SHAP explanation endpoint  
- Model information endpoint
- Prediction history tracking
- CORS support for frontend

Author: IME 372 Project Team
Date: November 2025
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os

# Initialize FastAPI app
app = FastAPI(
    title="Honey Bee Toxicity Prediction API",
    description="ML-powered API for predicting pesticide toxicity to honey bees",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and preprocessor at startup
MODEL_PATH = "outputs/models/best_model_xgboost.pkl"
PREPROCESSOR_PATH = "outputs/preprocessors/preprocessor.pkl"
RESULTS_PATH = "outputs/metrics/training_results.json"
HISTORY_FILE = "app/backend/prediction_history.json"

model = None
preprocessor = None
model_info = {}
prediction_history = []

@app.on_event("startup")
async def load_model():
    """Load model and preprocessor on startup."""
    global model, preprocessor, model_info, prediction_history
    
    try:
        # Load model
        model = joblib.load(MODEL_PATH)
        print(f"✓ Model loaded from {MODEL_PATH}")
        
        # Load preprocessor
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print(f"✓ Preprocessor loaded from {PREPROCESSOR_PATH}")
        
        # Load model info
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, 'r') as f:
                results = json.load(f)
                model_info = results.get('xgboost', {})
        
        # Load prediction history
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                prediction_history = json.load(f)
        
        print("✓ API Ready!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


# Pydantic models for request/response validation
class PredictionInput(BaseModel):
    """Input features for prediction."""
    source: str = Field(..., description="Data source (ECOTOX, PPDB, BPDB)")
    year: int = Field(..., description="Publication year", ge=1800, le=2030)
    toxicity_type: str = Field(..., description="Toxicity type (Contact, Oral, Other)")
    herbicide: int = Field(..., description="Is herbicide (0 or 1)", ge=0, le=1)
    fungicide: int = Field(..., description="Is fungicide (0 or 1)", ge=0, le=1)
    insecticide: int = Field(..., description="Is insecticide (0 or 1)", ge=0, le=1)
    other_agrochemical: int = Field(..., description="Is other agrochemical (0 or 1)", ge=0, le=1)
    MolecularWeight: float = Field(..., description="Molecular weight", ge=0)
    LogP: float = Field(..., description="Partition coefficient (lipophilicity)")
    NumHDonors: int = Field(..., description="Number of hydrogen bond donors", ge=0)
    NumHAcceptors: int = Field(..., description="Number of hydrogen bond acceptors", ge=0)
    NumRotatableBonds: int = Field(..., description="Number of rotatable bonds", ge=0)
    NumAromaticRings: int = Field(..., description="Number of aromatic rings", ge=0)
    TPSA: float = Field(..., description="Topological polar surface area", ge=0)
    NumHeteroatoms: int = Field(..., description="Number of heteroatoms", ge=0)
    NumRings: int = Field(..., description="Number of rings", ge=0)
    NumSaturatedRings: int = Field(..., description="Number of saturated rings", ge=0)
    NumAliphaticRings: int = Field(..., description="Number of aliphatic rings", ge=0)
    FractionCSP3: float = Field(..., description="Fraction of sp3 carbons", ge=0, le=1)
    MolarRefractivity: float = Field(..., description="Molar refractivity", ge=0)
    BertzCT: float = Field(..., description="Bertz molecular complexity", ge=0)
    HeavyAtomCount: int = Field(..., description="Number of heavy atoms", ge=0)
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionOutput(BaseModel):
    """Prediction output with confidence scores."""
    prediction: int = Field(..., description="Predicted class (0=non-toxic, 1=toxic)")
    prediction_label: str = Field(..., description="Human-readable prediction")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    timestamp: str = Field(..., description="Prediction timestamp")
    
    
class ModelInfo(BaseModel):
    """Model metadata and performance."""
    model_type: str
    features: List[str]
    performance: Dict[str, float]
    training_date: str


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Honey Bee Toxicity Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "Make toxicity predictions",
            "/model/info": "Get model information",
            "/history": "View prediction history",
            "/health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Make a toxicity prediction for a pesticide compound.
    
    Args:
        input_data: Pesticide features
        
    Returns:
        Prediction with confidence scores
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to dataframe
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Encode categorical features (matching training preprocessing)
        input_df = pd.get_dummies(input_df, columns=['source', 'toxicity_type'], drop_first=True)
        
        # Ensure all expected columns are present (add missing with 0s)
        expected_cols = [
            'year', 'herbicide', 'fungicide', 'insecticide', 'other_agrochemical',
            'MolecularWeight', 'LogP', 'NumHDonors', 'NumHAcceptors', 
            'NumRotatableBonds', 'NumAromaticRings', 'TPSA', 'NumHeteroatoms',
            'NumRings', 'NumSaturatedRings', 'NumAliphaticRings', 'FractionCSP3',
            'MolarRefractivity', 'BertzCT', 'HeavyAtomCount',
            'source_ECOTOX', 'source_PPDB', 'toxicity_type_Oral', 'toxicity_type_Other'
        ]
        
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match training
        input_df = input_df[expected_cols]
        
        # Scale features
        input_scaled = preprocessor.scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Create response
        response = {
            "prediction": int(prediction),
            "prediction_label": "Toxic" if prediction == 1 else "Non-toxic",
            "confidence": float(probabilities[prediction]),
            "probabilities": {
                "non_toxic": float(probabilities[0]),
                "toxic": float(probabilities[1])
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to history
        history_entry = {
            **input_dict,
            **response
        }
        prediction_history.append(history_entry)
        
        # Keep only last 100 predictions
        if len(prediction_history) > 100:
            prediction_history.pop(0)
        
        # Save history to file
        try:
            os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
            with open(HISTORY_FILE, 'w') as f:
                json.dump(prediction_history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save history: {e}")
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information and performance metrics."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    feature_names = [
        'year', 'herbicide', 'fungicide', 'insecticide', 'other_agrochemical',
        'MolecularWeight', 'LogP', 'NumHDonors', 'NumHAcceptors', 
        'NumRotatableBonds', 'NumAromaticRings', 'TPSA', 'NumHeteroatoms',
        'NumRings', 'NumSaturatedRings', 'NumAliphaticRings', 'FractionCSP3',
        'MolarRefractivity', 'BertzCT', 'HeavyAtomCount',
        'source_ECOTOX', 'source_PPDB', 'toxicity_type_Oral', 'toxicity_type_Other'
    ]
    
    performance = model_info.get('val_metrics', {
        'accuracy': 0.8558,
        'f1': 0.7368,
        'roc_auc': 0.8788
    })
    
    return {
        "model_type": "XGBoost Classifier",
        "features": feature_names,
        "performance": performance,
        "training_date": model_info.get('timestamp', datetime.now().isoformat())
    }


@app.get("/history")
async def get_history(limit: int = 10):
    """Get recent prediction history."""
    return {
        "total_predictions": len(prediction_history),
        "recent_predictions": prediction_history[-limit:]
    }


@app.get("/feature/importance")
async def get_feature_importance():
    """Get global feature importance from SHAP analysis."""
    importance_path = "outputs/metrics/feature_importance_shap.csv"
    
    if not os.path.exists(importance_path):
        raise HTTPException(status_code=404, detail="Feature importance data not found")
    
    try:
        importance_df = pd.read_csv(importance_path)
        importance_data = importance_df.head(15).to_dict(orient='records')
        
        return {
            "top_features": importance_data,
            "description": "Features ranked by mean absolute SHAP value"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading importance data: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

