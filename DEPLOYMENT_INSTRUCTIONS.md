# Quick Deployment Instructions
## For Tyler - Honey Bee Toxicity API

**Email**: tylerlubyhoward@gmail.com

---

## 🚨 SECURITY FIRST

**IMPORTANT**: Change your GitHub/Vercel password immediately since you shared it in plain text. Use these secure methods instead:

### For GitHub:
- **Personal Access Token** (not password): https://github.com/settings/tokens
  - Click "Generate new token (classic)"
  - Select scopes: `repo`, `workflow`
  - Copy token and use instead of password

### For Vercel:
- Uses secure browser authentication (no password needed)

---

## 🚀 Automated Deployment (3 Steps)

### Windows Users:
```bash
# Run this from PowerShell or Command Prompt
deploy.bat
```

### Mac/Linux Users:
```bash
# Make script executable
chmod +x deploy.sh

# Run it
./deploy.sh
```

The script will:
1. ✅ Initialize Git repository
2. ✅ Guide you through GitHub setup
3. ✅ Install Vercel CLI
4. ✅ Deploy to Vercel

**Just follow the prompts!**

---

## 📋 What You'll Need to Do

### Step 1: Create GitHub Repository
When the script pauses, go to:
1. https://github.com/new
2. Repository name: `bee-toxicity-api`
3. Make it **Public**
4. **Don't** initialize with README
5. Click "Create repository"

**Important**: When pushing to GitHub:
- Email: `tylerlubyhoward@gmail.com`
- Password: Use **Personal Access Token** (create at https://github.com/settings/tokens)

### Step 2: Vercel Authentication
When prompted:
1. Browser will open automatically
2. Log in with: `tylerlubyhoward@gmail.com`
3. Use your Vercel password
4. Click "Authorize"
5. Return to terminal

### Step 3: Done!
The script handles the rest automatically.

---

## 🐌 Manual Deployment (If Scripts Fail)

### A. GitHub Setup

```bash
# 1. Initialize git
git init

# 2. Configure git
git config user.email "tylerlubyhoward@gmail.com"
git config user.name "Tyler Howard"

# 3. Add files
git add .

# 4. Commit
git commit -m "Initial commit: Bee Toxicity Prediction"

# 5. Create repo on GitHub (browser)
# Go to: https://github.com/new
# Name: bee-toxicity-api
# Click: Create repository

# 6. Connect and push
git remote add origin https://github.com/YOUR_USERNAME/bee-toxicity-api.git
git push -u origin main
```

### B. Vercel Deployment

```bash
# 1. Install Vercel CLI
npm install -g vercel

# 2. Login (opens browser)
vercel login

# 3. Deploy
vercel --prod
```

---

## ✅ Testing Your Deployment

Once deployed, Vercel will show you a URL like:
```
https://bee-toxicity-api-xxx.vercel.app
```

### Test it:

**Health Check:**
```bash
curl https://YOUR_URL/health
```

**API Documentation:**
Open in browser:
```
https://YOUR_URL/docs
```

**Make a Prediction:**
Use the `/docs` interface or:
```bash
curl -X POST https://YOUR_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"source":"PPDB","year":2020,"toxicity_type":"Contact","insecticide":1,"herbicide":0,"fungicide":0,"other_agrochemical":0,"MolecularWeight":350.5,"LogP":3.2,"NumHDonors":2,"NumHAcceptors":4,"NumRotatableBonds":5,"AromaticRings":2,"TPSA":65.3,"NumHeteroatoms":5,"NumAromaticAtoms":12,"NumSaturatedRings":0,"NumAliphaticRings":0,"RingCount":2,"FractionCsp3":0.25,"NumAromaticCarbocycles":1,"NumSaturatedCarbocycles":0}'
```

---

## ⚠️ Troubleshooting

### Issue: "git: command not found"
**Solution**: Install Git from https://git-scm.com/download/win

### Issue: "npm: command not found"
**Solution**: Install Node.js from https://nodejs.org/ (includes npm)

### Issue: "Permission denied" (Mac/Linux)
**Solution**: 
```bash
chmod +x deploy.sh
```

### Issue: GitHub authentication fails
**Solution**: Use Personal Access Token:
1. Go to https://github.com/settings/tokens
2. Generate new token (classic)
3. Select: repo, workflow
4. Copy token
5. Use token as password when Git prompts

### Issue: Vercel deployment fails (Lambda size)
**Solution**: The deployment uses `requirements-vercel.txt` which is optimized for Vercel's limits. If still too large:
1. Use Railway instead: https://railway.app
2. Or upgrade to Vercel Pro ($20/mo)

---

## 📊 After Deployment

### Update Your Presentation
Replace localhost URLs with your Vercel URL:
- Old: `http://localhost:8000/docs`
- New: `https://bee-toxicity-api-xxx.vercel.app/docs`

### Share Your API
GitHub repo: `https://github.com/YOUR_USERNAME/bee-toxicity-api`  
Live API: `https://bee-toxicity-api-xxx.vercel.app`  
Documentation: `https://bee-toxicity-api-xxx.vercel.app/docs`

---

## 🎉 Success Checklist

- [ ] GitHub repository created
- [ ] Code pushed to GitHub
- [ ] Vercel deployment successful
- [ ] Health check returns `{"status": "healthy"}`
- [ ] API docs accessible at `/docs`
- [ ] Can make test prediction
- [ ] URL added to presentation slides

---

## 🆘 Need Help?

**If automated scripts fail:**
1. Try manual deployment steps above
2. Check troubleshooting section
3. Verify all prerequisites installed
4. Check error messages carefully

**Alternative platforms** (if Vercel doesn't work):
- **Railway**: Easier, no size limits (recommended!)
- **Heroku**: Traditional PaaS
- **Render**: Simple deployment

---

**Ready? Run the deployment script for your OS!**

Windows: `deploy.bat`  
Mac/Linux: `./deploy.sh`

🐝 Good luck!

