# Vercel Deployment Guide
## Honey Bee Toxicity Prediction API

This guide will help you deploy the FastAPI backend to Vercel.

---

## Prerequisites

1. **Vercel Account** - Sign up at https://vercel.com (free tier available)
2. **Vercel CLI** (optional) - For command-line deployment
3. **Git Repository** (recommended) - For automatic deployments

---

## Option 1: Deploy via Vercel Dashboard (Easiest)

### Step 1: Prepare Your Repository

Your project is already configured! We've added:
- ✅ `vercel.json` - Vercel configuration
- ✅ `requirements-vercel.txt` - Optimized dependencies

### Step 2: Push to GitHub (if not already)

```bash
# Initialize git (if not already)
git init

# Add files
git add .
git commit -m "Prepare for Vercel deployment"

# Create GitHub repo and push
git remote add origin https://github.com/YOUR_USERNAME/bee-toxicity-api.git
git push -u origin main
```

### Step 3: Import to Vercel

1. Go to https://vercel.com/dashboard
2. Click **"Add New..."** → **"Project"**
3. Import your GitHub repository
4. Vercel will auto-detect the configuration
5. Click **"Deploy"**

### Step 4: Configure Environment (if needed)

In Vercel dashboard:
- Go to **Settings** → **Environment Variables**
- Add any needed variables (none required for basic setup)

### Step 5: Access Your API

Vercel will provide a URL like:
```
https://bee-toxicity-api-YOUR_USERNAME.vercel.app
```

Test it:
```bash
curl https://YOUR_APP.vercel.app/health
```

---

## Option 2: Deploy via Vercel CLI

### Step 1: Install Vercel CLI

```bash
npm install -g vercel
```

### Step 2: Login

```bash
vercel login
```

### Step 3: Deploy

```bash
# From project root
vercel

# Follow prompts:
# - Set up and deploy? Yes
# - Which scope? Your account
# - Link to existing project? No
# - Project name? bee-toxicity-api
# - Directory? ./
# - Override settings? No
```

### Step 4: Production Deployment

```bash
vercel --prod
```

---

## Important Considerations for Vercel

### Lambda Size Limitations

**Vercel Free Tier**: 
- Max Lambda size: 50MB (uncompressed)
- Our model + dependencies may exceed this

**Solutions**:

1. **Reduce Dependencies** (already done in `requirements-vercel.txt`)
   - Removed SHAP/LIME (large packages)
   - Kept only essential ML libraries

2. **Use Vercel Pro** ($20/month)
   - Max Lambda size: 250MB
   - Can include SHAP/LIME

3. **Split Services** (microservices)
   - Core API on Vercel (predictions only)
   - Interpretability service elsewhere (AWS Lambda, Heroku)

4. **Use External Storage**
   - Store model in S3/GitHub LFS
   - Download on cold start (slower but works)

### Model File Storage

**Problem**: `best_model_xgboost.pkl` needs to be accessible

**Options**:

**A. Include in Deployment** (simplest)
```bash
# Model is already in outputs/models/
# Vercel will include it automatically
# May hit size limits
```

**B. External Storage** (recommended for production)
```python
# In app/backend/main.py, modify load_model():
import boto3
import os

@app.on_event("startup")
async def load_model():
    # Download model from S3
    s3 = boto3.client('s3')
    s3.download_file(
        'your-bucket',
        'models/best_model_xgboost.pkl',
        '/tmp/model.pkl'
    )
    model = joblib.load('/tmp/model.pkl')
```

**C. GitHub LFS** (good for open source)
```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "*.pkl"
git add .gitattributes
git commit -m "Add Git LFS for models"
```

---

## Configuration Details

### vercel.json Explained

```json
{
  "version": 2,                    // Vercel config version
  "name": "bee-toxicity-api",      // Project name
  "builds": [
    {
      "src": "app/backend/main.py", // Entry point
      "use": "@vercel/python",      // Python runtime
      "config": {
        "maxLambdaSize": "15mb"     // Adjust if needed
      }
    }
  ],
  "routes": [
    {
      "src": "/(.*)",               // Route all requests
      "dest": "app/backend/main.py" // To FastAPI app
    }
  ]
}
```

### Environment Variables

If you need to set environment variables:

```bash
# Via CLI
vercel env add MODEL_PATH production
# Enter: outputs/models/best_model_xgboost.pkl

# Via Dashboard
# Settings → Environment Variables → Add
```

---

## Testing Deployment

### Health Check

```bash
curl https://YOUR_APP.vercel.app/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true
}
```

### Make Prediction

```bash
curl -X POST https://YOUR_APP.vercel.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "source": "PPDB",
    "year": 2020,
    "toxicity_type": "Contact",
    "insecticide": 1,
    "herbicide": 0,
    "fungicide": 0,
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

### API Documentation

```
https://YOUR_APP.vercel.app/docs
```

---

## Troubleshooting

### Error: Lambda size exceeded

**Solution 1**: Remove heavy dependencies
```bash
# Edit requirements-vercel.txt
# Comment out: shap, lime, seaborn, matplotlib
```

**Solution 2**: Upgrade to Vercel Pro

**Solution 3**: Use model compression
```python
# Compress model file
import joblib
model = joblib.load('model.pkl')
joblib.dump(model, 'model_compressed.pkl', compress=3)
```

### Error: Model file not found

**Solution**: Check paths in `app/backend/main.py`
```python
# Use relative path from project root
MODEL_PATH = "outputs/models/best_model_xgboost.pkl"
```

### Error: Import errors

**Solution**: Ensure all dependencies in `requirements-vercel.txt`
```bash
# Test locally first
pip install -r requirements-vercel.txt
python app/backend/main.py
```

### Cold Start Delays

**Problem**: First request after inactivity is slow (10-30 seconds)

**Solutions**:
- Use Vercel Pro (faster cold starts)
- Implement health check pinging
- Accept cold starts as normal for serverless

---

## Alternative Deployment Options

If Vercel doesn't work due to size limits:

### 1. **Railway** (Recommended Alternative)
- Supports larger apps
- Easy deployment
- Free tier available
- Guide: https://railway.app/

### 2. **Heroku**
- Traditional PaaS
- No Lambda size limits
- Free tier (with limitations)
- Requires `Procfile`

### 3. **AWS Lambda + API Gateway**
- More control
- Larger size limits
- More complex setup
- Use Serverless Framework

### 4. **Google Cloud Run**
- Container-based
- No size limits (uses Docker)
- Pay per use
- Easy scaling

### 5. **DigitalOcean App Platform**
- Simple deployment
- No serverless limitations
- $5/month minimum

---

## Cost Comparison

| Platform | Free Tier | Model Support | Complexity |
|----------|-----------|---------------|------------|
| **Vercel** | Yes (50MB) | Limited | Low |
| Vercel Pro | $20/mo | Good | Low |
| Railway | Yes (500 hrs) | Excellent | Low |
| Heroku | Limited | Good | Low |
| AWS Lambda | Yes | Good | Medium |
| Cloud Run | Yes | Excellent | Medium |
| DigitalOcean | No ($5/mo) | Excellent | Low |

---

## Recommended Approach

### For This Project:

**Option A: Quick Demo (Vercel Free)**
1. Deploy with minimal dependencies
2. Accept that SHAP/LIME won't work
3. Basic predictions only
4. Good for presentation demo

**Option B: Full Features (Railway/Heroku)**
1. Full dependency support
2. SHAP/LIME interpretability works
3. Better for production
4. Railway recommended (easier than Heroku)

**Option C: Production (Cloud Run)**
1. Use Docker (already have Dockerfile)
2. No size limitations
3. Excellent performance
4. More setup but best long-term

---

## What I've Prepared for You

✅ **vercel.json** - Vercel configuration  
✅ **requirements-vercel.txt** - Optimized dependencies  
✅ **VERCEL_DEPLOYMENT.md** - This guide  
✅ **Existing Dockerfiles** - Alternative deployment  

---

## Next Steps

### To Deploy Now:

1. **Choose Platform**:
   - Quick demo? → Vercel
   - Full features? → Railway
   - Production? → Cloud Run

2. **For Vercel** (simplest):
   ```bash
   # Install CLI
   npm install -g vercel
   
   # Login
   vercel login
   
   # Deploy
   vercel
   ```

3. **For Railway** (recommended):
   - Go to https://railway.app/
   - Click "Start a New Project"
   - Connect GitHub repo
   - Deploy automatically

4. **Test Deployment**:
   ```bash
   curl https://YOUR_URL/health
   ```

---

## Support

- **Vercel Docs**: https://vercel.com/docs
- **Railway Docs**: https://docs.railway.app/
- **FastAPI on Vercel**: https://vercel.com/guides/deploying-fastapi-with-vercel

---

**Need Help?** Let me know which platform you want to use and I can provide more specific instructions!

