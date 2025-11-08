import { useEffect, useState } from 'react'
import { getModelInfo, ModelInfo as ModelInfoType } from '../services/api'

const ModelInfo = () => {
  const [info, setInfo] = useState<ModelInfoType | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchInfo = async () => {
      try {
        const data = await getModelInfo()
        setInfo(data)
      } catch (err) {
        setError('Unable to fetch model info')
      }
    }
    fetchInfo()
  }, [])

  if (error) {
    return (
      <div className="card bg-gray-50">
        <h3 className="font-bold text-gray-800 mb-3">📊 Model Information</h3>
        <div className="space-y-2 text-sm">
          <div><span className="font-semibold">Algorithm:</span> XGBoost</div>
          <div><span className="font-semibold">Test Accuracy:</span> 83.6%</div>
          <div><span className="font-semibold">ROC-AUC:</span> 85.8%</div>
          <div><span className="font-semibold">Dataset:</span> 1,035 compounds</div>
        </div>
      </div>
    )
  }

  return (
    <div className="card bg-gray-50">
      <h3 className="font-bold text-gray-800 mb-3">📊 Model Information</h3>
      
      <div className="space-y-2 text-sm">
        {info && (
          <>
            <div>
              <span className="font-semibold">Algorithm:</span> {info.algorithm || 'XGBoost'}
            </div>
            <div>
              <span className="font-semibold">Version:</span> {info.version || '1.0.0'}
            </div>
            {info.metrics && (
              <>
                {info.metrics.test_accuracy && (
                  <div>
                    <span className="font-semibold">Test Accuracy:</span>{' '}
                    {(info.metrics.test_accuracy * 100).toFixed(2)}%
                  </div>
                )}
                {info.metrics.test_roc_auc && (
                  <div>
                    <span className="font-semibold">ROC-AUC:</span>{' '}
                    {(info.metrics.test_roc_auc * 100).toFixed(2)}%
                  </div>
                )}
              </>
            )}
          </>
        )}
        <div><span className="font-semibold">Dataset:</span> 1,035 compounds</div>
        <div><span className="font-semibold">Features:</span> 24 descriptors</div>
      </div>

      <div className="mt-4 pt-4 border-t border-gray-200">
        <h4 className="font-semibold text-xs text-gray-700 mb-2">Top Predictors</h4>
        <ol className="text-xs text-gray-600 space-y-1">
          <li>1. Insecticide flag</li>
          <li>2. Herbicide flag</li>
          <li>3. Fungicide flag</li>
          <li>4. Publication year</li>
          <li>5. LogP (lipophilicity)</li>
        </ol>
      </div>

      <div className="mt-4 pt-4 border-t border-gray-200">
        <a
          href="http://localhost:8000/docs"
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-indigo-600 hover:text-indigo-800 underline block text-center"
        >
          📚 View API Documentation →
        </a>
      </div>
    </div>
  )
}

export default ModelInfo

