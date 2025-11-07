@echo off
REM Automated Deployment Script (Windows)
REM Honey Bee Toxicity Prediction System
REM =======================================

echo.
echo 🐝 Automated Deployment Script (Windows)
echo ========================================
echo.

REM Step 1: Check prerequisites
echo 📋 Step 1: Checking prerequisites...
echo.

where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Git not found. Please install Git first.
    echo    Download from: https://git-scm.com/download/win
    pause
    exit /b 1
)
echo ✓ Git installed

where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️  Node.js not found. Please install Node.js first.
    echo    Download from: https://nodejs.org/
    pause
    exit /b 1
)
echo ✓ Node.js installed

where npm >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ npm not found
    pause
    exit /b 1
)
echo ✓ npm installed

echo.

REM Step 2: GitHub Setup
echo 📦 Step 2: Setting up GitHub repository...
echo.

REM Check if git is initialized
if not exist .git (
    echo Initializing Git repository...
    git init
    echo ✓ Git initialized
) else (
    echo ✓ Git already initialized
)

REM Configure git user (if not set)
git config user.email "tylerlubyhoward@gmail.com" 2>nul
git config user.name "Tyler Howard" 2>nul

REM Add all files
echo Adding files to Git...
git add .

REM Commit
echo Creating commit...
git commit -m "Initial commit: Honey Bee Toxicity Prediction System" 2>nul || echo No changes to commit

echo.
echo ════════════════════════════════════════
echo   ACTION REQUIRED: GitHub Setup
echo ════════════════════════════════════════
echo.
echo To create GitHub repository:
echo.
echo 1. Open: https://github.com/new
echo 2. Repository name: bee-toxicity-api
echo 3. Make it Public
echo 4. Don't initialize with README
echo 5. Click 'Create repository'
echo.
echo Your email: tylerlubyhoward@gmail.com
echo.
pause

REM Get GitHub repo URL
echo.
set /p GITHUB_REPO_URL="Enter your GitHub repository URL: "

if not "%GITHUB_REPO_URL%"=="" (
    echo Adding GitHub remote...
    git remote remove origin 2>nul
    git remote add origin %GITHUB_REPO_URL%
    echo ✓ GitHub remote added
    
    echo.
    echo Pushing to GitHub...
    echo.
    echo ⚠️  IMPORTANT: GitHub Authentication
    echo.
    echo For password, use a Personal Access Token (NOT your login password):
    echo   1. Go to: https://github.com/settings/tokens
    echo   2. Click 'Generate new token (classic)'
    echo   3. Select: repo, workflow
    echo   4. Generate and copy the token
    echo   5. Use token as password when prompted
    echo.
    pause
    
    git push -u origin main 2>nul || git push -u origin master
    echo ✓ Pushed to GitHub
) else (
    echo ⚠️  Skipping GitHub push
)

echo.

REM Step 3: Vercel Setup
echo 🚀 Step 3: Setting up Vercel...
echo.

REM Check if Vercel CLI is installed
where vercel >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing Vercel CLI...
    call npm install -g vercel
    echo ✓ Vercel CLI installed
) else (
    echo ✓ Vercel CLI already installed
)

echo.
echo ════════════════════════════════════════
echo   ACTION REQUIRED: Vercel Login
echo ════════════════════════════════════════
echo.
echo Vercel will open your browser for authentication.
echo.
echo Your email: tylerlubyhoward@gmail.com
echo.
echo Steps:
echo   1. Browser will open
echo   2. Log in with your email
echo   3. Authorize Vercel CLI
echo   4. Return to this window
echo.
pause

call vercel login

echo.
echo ✓ Vercel authentication complete
echo.

REM Step 4: Deploy
echo 🚀 Step 4: Deploying to Vercel...
echo.

echo Deploying to production...
call vercel --prod --yes

echo.
echo ════════════════════════════════════════
echo   ✅ DEPLOYMENT COMPLETE!
echo ════════════════════════════════════════
echo.
echo Your API is now live!
echo.
echo Check the URL printed above.
echo It should be: https://bee-toxicity-api-xxx.vercel.app
echo.
echo Next steps:
echo   1. Test: curl YOUR_URL/health
echo   2. View docs: YOUR_URL/docs
echo.
echo 🐝 Happy bee saving!
echo.
pause

