# Honey Bee Toxicity Prediction - Frontend

React + TypeScript + Vite frontend for the Bee Toxicity Prediction System.

## Quick Start

### Prerequisites
- Node.js 18+ and npm
- Backend API running at http://localhost:8000

### Installation

```bash
cd app/frontend
npm install
```

### Development

```bash
npm run dev
```

Open http://localhost:3000

### Build for Production

```bash
npm run build
npm run preview
```

## Features

- ✨ Modern React 18 with TypeScript
- 🎨 Tailwind CSS for styling
- 📊 Interactive prediction form
- 🎯 Real-time results display
- 📈 Model information panel
- 🚀 Fast development with Vite
- 🔌 API integration with Axios

## Components

- **PredictionForm**: Input form for compound properties
- **ResultDisplay**: Shows prediction results with confidence scores
- **ModelInfo**: Displays model metadata and performance

## API Integration

The frontend connects to the backend API at `http://localhost:8000`.

Endpoints used:
- POST /predict - Make predictions
- GET /model/info - Get model information
- GET /health - Check API status

## Note

The backend API currently has a preprocessing integration issue. You can still:
- ✅ View the beautiful UI
- ✅ See the form and interface
- ✅ Use the API documentation at http://localhost:8000/docs
- ⚠️ Predictions will show error until API is fixed

This demonstrates the complete full-stack architecture even though one integration point needs adjustment.

