#!/bin/bash
# Quick start script for setting up GitHub repository
# Run this after creating the repository on GitHub

set -e  # Exit on error

echo "🚀 Dartboard RAG - GitHub Setup"
echo "================================"
echo ""

# Step 1: Initialize git
echo "📝 Step 1: Initializing git..."
git init
echo "✓ Git initialized"
echo ""

# Step 2: Add files
echo "📝 Step 2: Adding files..."
git add .
echo "✓ Files added"
echo ""

# Step 3: Check what will be committed
echo "📋 Files to be committed:"
git status --short
echo ""

# Step 4: Create initial commit
echo "📝 Step 3: Creating initial commit..."
git commit -m "Initial commit: Dartboard RAG implementation

- Complete Dartboard algorithm with greedy selection
- Document loaders (PDF, Markdown, Code)
- Vector store abstraction (FAISS, Pinecone)
- Hybrid retrieval (vector + Dartboard)
- Comprehensive test suite (6 tests passing)
- Evaluation metrics (NDCG, MAP, diversity)
- Documentation and implementation plans"
echo "✓ Initial commit created"
echo ""

# Step 5: Prompt for GitHub username
echo "📝 Step 4: Setting up remote..."
read -p "Enter your GitHub username: " GITHUB_USER

# Check if user wants SSH or HTTPS
echo ""
echo "Choose authentication method:"
echo "1) HTTPS (easier, works everywhere)"
echo "2) SSH (recommended, requires SSH key setup)"
read -p "Enter choice (1 or 2): " AUTH_CHOICE

if [ "$AUTH_CHOICE" = "2" ]; then
    REMOTE_URL="git@github.com:$GITHUB_USER/vastai.git"
else
    REMOTE_URL="https://github.com/$GITHUB_USER/vastai.git"
fi

git remote add origin $REMOTE_URL
echo "✓ Remote added: $REMOTE_URL"
echo ""

# Step 6: Rename branch to main
echo "📝 Step 5: Renaming branch to main..."
git branch -M main
echo "✓ Branch renamed to main"
echo ""

# Step 7: Push to GitHub
echo "📝 Step 6: Pushing to GitHub..."
echo "You may be prompted for credentials..."
git push -u origin main
echo "✓ Pushed to GitHub!"
echo ""

echo "================================"
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Visit https://github.com/$GITHUB_USER/vastai"
echo "2. Verify files are there"
echo "3. Check repository is Private"
echo "4. (Optional) Create first PR branch:"
echo "   git checkout -b feat/document-loaders"
echo "   git push -u origin feat/document-loaders"
echo ""
echo "See GITHUB_SETUP.md for detailed instructions"
