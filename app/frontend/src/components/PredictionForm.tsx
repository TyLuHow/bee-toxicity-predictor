import { useState } from 'react'
import { predictToxicity, PredictionInput } from '../services/api'

interface Props {
  onPrediction: (result: any) => void
  onError: (error: string) => void
  onLoading: (loading: boolean) => void
}

const PredictionForm = ({ onPrediction, onError, onLoading }: Props) => {
  const [formData, setFormData] = useState<Partial<PredictionInput>>({
    source: 'PPDB',
    year: 2020,
    toxicity_type: 'Contact',
    insecticide: 1,
    herbicide: 0,
    fungicide: 0,
    other_agrochemical: 0,
    MolecularWeight: 350.5,
    LogP: 3.2,
    NumHDonors: 2,
    NumHAcceptors: 4,
    NumRotatableBonds: 5,
    AromaticRings: 2,
    NumAromaticRings: 2,
    TPSA: 65.3,
    NumHeteroatoms: 5,
    NumAromaticAtoms: 12,
    NumSaturatedRings: 0,
    NumAliphaticRings: 0,
    RingCount: 2,
    NumRings: 2,
    FractionCsp3: 0.25,
    FractionCSP3: 0.25,
    NumAromaticCarbocycles: 1,
    NumSaturatedCarbocycles: 0,
    MolarRefractivity: 95.5,
    BertzCT: 850.0,
    HeavyAtomCount: 25
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    onLoading(true)
    try {
      const result = await predictToxicity(formData as PredictionInput)
      onPrediction(result)
    } catch (err: any) {
      onError(err.response?.data?.detail || 'Prediction failed. Please check your inputs.')
    } finally {
      onLoading(false)
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : value
    }))
  }

  return (
    <div className="card">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">
        Enter Compound Properties
      </h2>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Compound Info */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="label">Source</label>
            <select name="source" value={formData.source} onChange={handleChange} className="input-field">
              <option value="PPDB">PPDB</option>
              <option value="ECOTOX">ECOTOX</option>
              <option value="BPDB">BPDB</option>
            </select>
          </div>
          
          <div>
            <label className="label">Year</label>
            <input
              type="number"
              name="year"
              value={formData.year}
              onChange={handleChange}
              className="input-field"
              min="1800"
              max="2030"
            />
          </div>
          
          <div>
            <label className="label">Toxicity Type</label>
            <select name="toxicity_type" value={formData.toxicity_type} onChange={handleChange} className="input-field">
              <option value="Contact">Contact</option>
              <option value="Oral">Oral</option>
              <option value="Other">Other</option>
            </select>
          </div>
        </div>

        {/* Chemical Type Flags */}
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="font-semibold text-gray-700 mb-3">Chemical Type</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                name="insecticide"
                checked={formData.insecticide === 1}
                onChange={(e) => setFormData(prev => ({ ...prev, insecticide: e.target.checked ? 1 : 0 }))}
                className="w-4 h-4 text-indigo-600"
              />
              <span className="text-sm">Insecticide</span>
            </label>
            
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                name="herbicide"
                checked={formData.herbicide === 1}
                onChange={(e) => setFormData(prev => ({ ...prev, herbicide: e.target.checked ? 1 : 0 }))}
                className="w-4 h-4 text-indigo-600"
              />
              <span className="text-sm">Herbicide</span>
            </label>
            
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                name="fungicide"
                checked={formData.fungicide === 1}
                onChange={(e) => setFormData(prev => ({ ...prev, fungicide: e.target.checked ? 1 : 0 }))}
                className="w-4 h-4 text-indigo-600"
              />
              <span className="text-sm">Fungicide</span>
            </label>
            
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                name="other_agrochemical"
                checked={formData.other_agrochemical === 1}
                onChange={(e) => setFormData(prev => ({ ...prev, other_agrochemical: e.target.checked ? 1 : 0 }))}
                className="w-4 h-4 text-indigo-600"
              />
              <span className="text-sm">Other</span>
            </label>
          </div>
        </div>

        {/* Molecular Descriptors */}
        <div className="bg-blue-50 p-4 rounded-lg">
          <h3 className="font-semibold text-gray-700 mb-3">Molecular Descriptors</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div>
              <label className="label text-xs">Molecular Weight</label>
              <input type="number" step="0.1" name="MolecularWeight" value={formData.MolecularWeight} onChange={handleChange} className="input-field text-sm" />
            </div>
            
            <div>
              <label className="label text-xs">LogP</label>
              <input type="number" step="0.1" name="LogP" value={formData.LogP} onChange={handleChange} className="input-field text-sm" />
            </div>
            
            <div>
              <label className="label text-xs">H-Bond Donors</label>
              <input type="number" name="NumHDonors" value={formData.NumHDonors} onChange={handleChange} className="input-field text-sm" />
            </div>
            
            <div>
              <label className="label text-xs">H-Bond Acceptors</label>
              <input type="number" name="NumHAcceptors" value={formData.NumHAcceptors} onChange={handleChange} className="input-field text-sm" />
            </div>
            
            <div>
              <label className="label text-xs">Rotatable Bonds</label>
              <input type="number" name="NumRotatableBonds" value={formData.NumRotatableBonds} onChange={handleChange} className="input-field text-sm" />
            </div>
            
            <div>
              <label className="label text-xs">Aromatic Rings</label>
              <input type="number" name="AromaticRings" value={formData.AromaticRings} onChange={handleChange} className="input-field text-sm" />
            </div>
            
            <div>
              <label className="label text-xs">TPSA</label>
              <input type="number" step="0.1" name="TPSA" value={formData.TPSA} onChange={handleChange} className="input-field text-sm" />
            </div>
            
            <div>
              <label className="label text-xs">Heteroatoms</label>
              <input type="number" name="NumHeteroatoms" value={formData.NumHeteroatoms} onChange={handleChange} className="input-field text-sm" />
            </div>
            
            <div>
              <label className="label text-xs">Heavy Atoms</label>
              <input type="number" name="HeavyAtomCount" value={formData.HeavyAtomCount} onChange={handleChange} className="input-field text-sm" />
            </div>
          </div>
        </div>

        <button
          type="submit"
          className="btn-primary w-full text-lg py-3"
        >
          🔮 Predict Toxicity
        </button>
      </form>
      
      <p className="text-xs text-gray-500 mt-4 text-center">
        💡 Tip: Insecticides are typically more toxic to bees
      </p>
    </div>
  )
}

export default PredictionForm

