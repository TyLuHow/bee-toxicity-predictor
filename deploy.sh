#!/bin/bash
# Automated Deployment Script
# Honey Bee Toxicity Prediction System
# ======================================

set -e  # Exit on error

echo "🐝 Automated Deployment Script"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check prerequisites
echo "📋 Step 1: Checking prerequisites..."

if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git not found. Please install Git first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Git installed${NC}"

if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}⚠️  Node.js not found. Installing Vercel CLI will require Node.js.${NC}"
    echo "   Download from: https://nodejs.org/"
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ Node.js installed${NC}"
fi

if ! command -v npm &> /dev/null; then
    echo -e "${YELLOW}⚠️  npm not found${NC}"
else
    echo -e "${GREEN}✓ npm installed${NC}"
fi

echo ""

# Step 2: GitHub Setup
echo "📦 Step 2: Setting up GitHub repository..."
echo ""

# Check if git is initialized
if [ ! -d .git ]; then
    echo "Initializing Git repository..."
    git init
    echo -e "${GREEN}✓ Git initialized${NC}"
else
    echo -e "${GREEN}✓ Git already initialized${NC}"
fi

# Add all files
echo "Adding files to Git..."
git add .

# Commit
echo "Creating commit..."
git commit -m "Initial commit: Honey Bee Toxicity Prediction System" || echo "No changes to commit"

echo ""
echo -e "${YELLOW}════════════════════════════════════════${NC}"
echo -e "${YELLOW}  ACTION REQUIRED: GitHub Authentication${NC}"
echo -e "${YELLOW}════════════════════════════════════════${NC}"
echo ""
echo "To create GitHub repository, you need to:"
echo ""
echo "Option A - GitHub CLI (Recommended):"
echo "  1. Install: https://cli.github.com/"
echo "  2. Run: gh auth login"
echo "  3. Run: gh repo create bee-toxicity-api --public --source=. --push"
echo ""
echo "Option B - Manual (Browser):"
echo "  1. Go to: https://github.com/new"
echo "  2. Repository name: bee-toxicity-api"
echo "  3. Make it Public"
echo "  4. Don't initialize with README (we already have one)"
echo "  5. Click 'Create repository'"
echo "  6. Copy the commands shown and run them here"
echo ""
echo "Your email: tylerlubyhoward@gmail.com"
echo ""
read -p "Press Enter when you've created the GitHub repo..."

# Get GitHub repo URL
echo ""
read -p "Enter your GitHub repository URL (e.g., https://github.com/username/bee-toxicity-api.git): " GITHUB_REPO_URL

if [ ! -z "$GITHUB_REPO_URL" ]; then
    # Remove existing origin if exists
    git remote remove origin 2>/dev/null || true
    
    # Add new origin
    git remote add origin "$GITHUB_REPO_URL"
    echo -e "${GREEN}✓ GitHub remote added${NC}"
    
    # Push to GitHub
    echo "Pushing to GitHub..."
    echo ""
    echo -e "${YELLOW}You may be prompted for GitHub credentials:${NC}"
    echo "  Email: tylerlubyhoward@gmail.com"
    echo "  Password: Use a Personal Access Token (NOT your password)"
    echo ""
    echo "To create a token:"
    echo "  1. Go to: https://github.com/settings/tokens"
    echo "  2. Click 'Generate new token (classic)'"
    echo "  3. Select: repo (all), workflow"
    echo "  4. Click 'Generate token'"
    echo "  5. Copy the token and use it as password"
    echo ""
    
    git push -u origin main || git push -u origin master
    echo -e "${GREEN}✓ Pushed to GitHub!${NC}"
else
    echo -e "${YELLOW}⚠️  Skipping GitHub push${NC}"
fi

echo ""

# Step 3: Vercel Setup
echo "🚀 Step 3: Setting up Vercel deployment..."
echo ""

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "Installing Vercel CLI..."
    npm install -g vercel
    echo -e "${GREEN}✓ Vercel CLI installed${NC}"
else
    echo -e "${GREEN}✓ Vercel CLI already installed${NC}"
fi

echo ""
echo -e "${YELLOW}════════════════════════════════════════${NC}"
echo -e "${YELLOW}  ACTION REQUIRED: Vercel Authentication${NC}"
echo -e "${YELLOW}════════════════════════════════════════${NC}"
echo ""
echo "Vercel will open a browser for secure authentication."
echo "Your email: tylerlubyhoward@gmail.com"
echo ""
echo "Steps:"
echo "  1. Browser will open automatically"
echo "  2. Log in with your email"
echo "  3. Authorize Vercel CLI"
echo "  4. Return to this terminal"
echo ""
read -p "Press Enter to start Vercel authentication..."

vercel login

echo ""
echo -e "${GREEN}✓ Vercel authentication complete${NC}"
echo ""

# Step 4: Deploy to Vercel
echo "🚀 Step 4: Deploying to Vercel..."
echo ""

echo "Deploying..."
vercel --prod --yes

echo ""
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ DEPLOYMENT COMPLETE!${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""
echo "Your API is now live!"
echo ""
echo "Vercel will show you the deployment URL above."
echo "It should look like: https://bee-toxicity-api-xxx.vercel.app"
echo ""
echo "Next steps:"
echo "  1. Test health: curl https://YOUR_URL/health"
echo "  2. View docs: https://YOUR_URL/docs"
echo "  3. Make prediction: Use the /docs interface"
echo ""
echo "🐝 Happy bee saving!"

