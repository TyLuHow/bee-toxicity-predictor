import { useState } from 'react'
import PredictionForm from './components/PredictionForm'
import ResultDisplay from './components/ResultDisplay'
import ModelInfo from './components/ModelInfo'
import './App.css'

interface PredictionResult {
  prediction: number
  label_text: string
  probability_toxic: number
  probability_non_toxic: number
  confidence: number
  timestamp: string
}

function App() {
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handlePrediction = (predictionResult: PredictionResult) => {
    setResult(predictionResult)
    setError(null)
  }

  const handleError = (errorMessage: string) => {
    setError(errorMessage)
    setResult(null)
  }

  const handleLoading = (isLoading: boolean) => {
    setLoading(isLoading)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <header className="text-center mb-8">
          <h1 className="text-5xl font-bold text-white mb-2 flex items-center justify-center gap-3">
            🐝 Honey Bee Toxicity Predictor
          </h1>
          <p className="text-xl text-white/90">
            ML-Powered Pesticide Safety Assessment
          </p>
          <p className="text-sm text-white/70 mt-2">
            IME 372 Course Project | 83.6% Accuracy | XGBoost Classifier
          </p>
        </header>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Prediction Form */}
          <div className="lg:col-span-2">
            <PredictionForm
              onPrediction={handlePrediction}
              onError={handleError}
              onLoading={handleLoading}
            />
          </div>

          {/* Right Column - Results & Info */}
          <div className="space-y-6">
            <ResultDisplay
              result={result}
              loading={loading}
              error={error}
            />
            <ModelInfo />
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-8 text-center text-white/70 text-sm">
          <p>Built with ❤️ for pollinator conservation</p>
          <p className="mt-1">
            Data: ApisTox Dataset (1,035 compounds) | Model: XGBoost | Interpretability: SHAP
          </p>
        </footer>
      </div>
    </div>
  )
}

export default App

