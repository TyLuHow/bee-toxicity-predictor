interface PredictionResult {
  prediction: number
  label_text: string
  probability_toxic: number
  probability_non_toxic: number
  confidence: number
  timestamp: string
}

interface Props {
  result: PredictionResult | null
  loading: boolean
  error: string | null
}

const ResultDisplay = ({ result, loading, error }: Props) => {
  if (loading) {
    return (
      <div className="card text-center">
        <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-indigo-600 mx-auto"></div>
        <p className="mt-4 text-gray-600">Analyzing compound...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="card bg-red-50 border-2 border-red-200">
        <h3 className="text-lg font-bold text-red-800 mb-2">❌ Error</h3>
        <p className="text-red-600">{error}</p>
        <p className="text-xs text-gray-500 mt-2">
          Note: The prediction endpoint has a known preprocessing issue. Try the API docs at localhost:8000/docs instead.
        </p>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="card text-center text-gray-500">
        <div className="text-6xl mb-4">🐝</div>
        <h3 className="text-lg font-semibold mb-2">Ready to Predict</h3>
        <p className="text-sm">
          Enter compound properties and click "Predict Toxicity" to see results.
        </p>
      </div>
    )
  }

  const isToxic = result.prediction === 1
  const confidence = (result.confidence * 100).toFixed(1)

  return (
    <div className={`card ${isToxic ? 'bg-red-50 border-2 border-red-300' : 'bg-green-50 border-2 border-green-300'}`}>
      <h3 className="text-2xl font-bold mb-4 text-center">
        {isToxic ? '⚠️ Toxic' : '✅ Non-Toxic'}
      </h3>
      
      <div className="space-y-4">
        {/* Prediction */}
        <div className="text-center">
          <div className={`text-5xl font-bold ${isToxic ? 'text-red-600' : 'text-green-600'}`}>
            {result.label_text}
          </div>
          <p className="text-sm text-gray-600 mt-2">
            Confidence: {confidence}%
          </p>
        </div>

        {/* Probabilities */}
        <div className="space-y-2">
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span>Non-Toxic</span>
              <span className="font-semibold">{(result.probability_non_toxic * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-green-500 h-2 rounded-full transition-all duration-500"
                style={{ width: `${result.probability_non_toxic * 100}%` }}
              ></div>
            </div>
          </div>
          
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span>Toxic</span>
              <span className="font-semibold">{(result.probability_toxic * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-red-500 h-2 rounded-full transition-all duration-500"
                style={{ width: `${result.probability_toxic * 100}%` }}
              ></div>
            </div>
          </div>
        </div>

        {/* Interpretation */}
        <div className="bg-white p-3 rounded-lg border">
          <h4 className="font-semibold text-sm mb-2">🧠 Interpretation</h4>
          <p className="text-xs text-gray-700">
            {isToxic ? (
              <>
                This compound is predicted to be <strong>toxic to honey bees</strong>. 
                {result.confidence > 0.8 ? ' High confidence prediction.' : ' Moderate confidence - consider laboratory validation.'}
              </>
            ) : (
              <>
                This compound is predicted to be <strong>safe for honey bees</strong>.
                {result.confidence > 0.8 ? ' High confidence prediction.' : ' Moderate confidence - monitoring recommended.'}
              </>
            )}
          </p>
        </div>

        {/* Timestamp */}
        <p className="text-xs text-gray-500 text-center">
          Predicted at: {new Date(result.timestamp).toLocaleString()}
        </p>
      </div>
    </div>
  )
}

export default ResultDisplay

