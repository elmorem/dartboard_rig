# GitHub Repository Setup Guide

## Step 1: Create Private Repository on GitHub

### Via GitHub Website

1. Go to https://github.com/new
2. **Repository name:** `dartboard_rig` (or your preferred name)
3. **Description:** "Dartboard RAG system for diversity-aware document retrieval"
4. **Visibility:** ✅ **Private**
5. **Initialize repository:**
   - ❌ DO NOT add README (we have one)
   - ❌ DO NOT add .gitignore (we have one)
   - ❌ DO NOT add license yet
6. Click **"Create repository"**

### Via GitHub CLI (Alternative)

```bash
# Install gh if needed
brew install gh

# Login
gh auth login

# Create private repo
gh repo create dartboard_rig --private --description "Dartboard RAG system"
```

---

## Step 2: Initialize Git in Local Directory

```bash
# Navigate to project
cd /Users/markelmore/_code/dartboard_rig

# Initialize git (if not already done)
git init

# Check status
git status
```

---

## Step 3: Review Files Before Committing

### Check what will be committed
```bash
# See all files
ls -la

# See what git will track
git status
```

### Important: Verify .gitignore
```bash
# Ensure .gitignore is working
cat .gitignore

# Test that .venv is ignored
git status | grep .venv
# Should see nothing (if .venv exists)
```

### Files that SHOULD be committed:
- ✅ `README.md`
- ✅ `CLAUDE.md`
- ✅ `.gitignore`
- ✅ `requirements.txt`
- ✅ `dartboard/` (all Python code)
- ✅ `demo_*.py`, `test_*.py`
- ✅ Documentation (`*.md` files)

### Files that should NOT be committed:
- ❌ `.venv/` (virtual environment)
- ❌ `__pycache__/` (Python cache)
- ❌ `.env` (secrets)
- ❌ `*.pyc` (compiled Python)
- ❌ `.DS_Store` (macOS)
- ❌ Large data files

---

## Step 4: Initial Commit

```bash
# Add all files
git add .

# Review what will be committed
git status

# If you see files that shouldn't be committed, remove them:
# git reset <filename>

# Create initial commit
git commit -m "Initial commit: Dartboard RAG implementation

- Complete Dartboard algorithm with greedy selection
- Document loaders (PDF, Markdown, Code)
- Vector store abstraction (FAISS, Pinecone)
- Hybrid retrieval (vector + Dartboard)
- Comprehensive test suite (6 tests passing)
- Evaluation metrics (NDCG, MAP, diversity)
- Documentation and implementation plans"
```

---

## Step 5: Connect to GitHub Remote

```bash
# Add remote (replace USERNAME with your GitHub username)
git remote add origin https://github.com/USERNAME/dartboard_rig.git

# Or use SSH (recommended)
git remote add origin git@github.com:USERNAME/dartboard_rig.git

# Verify remote
git remote -v
```

---

## Step 6: Push to GitHub

```bash
# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main

# Enter credentials if prompted
```

---

## Step 7: Verify on GitHub

1. Go to https://github.com/USERNAME/dartboard_rig
2. Verify files are there
3. Check that README displays correctly
4. Verify repository is **Private** (lock icon)

---

## Step 8: Configure Repository Settings (Optional)

### Add Topics/Tags
1. Go to repository page
2. Click "⚙️" next to "About"
3. Add topics: `rag`, `retrieval`, `nlp`, `python`, `machine-learning`

### Protect Main Branch
1. Go to Settings → Branches
2. Add rule for `main` branch
3. Enable:
   - ✅ Require pull request reviews
   - ✅ Require status checks to pass

### Add Collaborators (if needed)
1. Go to Settings → Collaborators
2. Add team members

---

## Step 9: Create First PR Branch (Optional)

```bash
# Create branch for document loaders PR
git checkout -b feat/document-loaders

# This branch is ready to be pushed as PR #1
git push -u origin feat/document-loaders

# Create PR on GitHub
gh pr create --title "feat: Add document loaders (PDF, MD, Code)" \
  --body "Implements comprehensive document loading system.
  
  **What's included:**
  - PDFLoader with metadata extraction
  - MarkdownLoader with frontmatter support
  - CodeRepositoryLoader for repos
  - DirectoryLoader for auto-detection
  - Comprehensive tests (4/4 passing)
  
  **Dependencies:**
  - pypdf
  - pyyaml
  
  **Testing:**
  \`\`\`bash
  python test_loaders.py
  \`\`\`"
```

---

## Troubleshooting

### Problem: .venv is being tracked
```bash
# Remove from git
git rm -r --cached .venv

# Verify .gitignore includes .venv/
echo ".venv/" >> .gitignore

# Commit
git commit -m "Remove .venv from tracking"
```

### Problem: Large files causing push to fail
```bash
# Find large files
find . -type f -size +50M

# Add to .gitignore
echo "path/to/large/file" >> .gitignore

# Remove from git
git rm --cached path/to/large/file

# Commit
git commit -m "Remove large files"
```

### Problem: Accidentally committed secrets
```bash
# Remove from git history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all

# Force push (CAUTION: rewrites history)
git push origin --force --all
```

### Problem: Need to change repository visibility
1. Go to Settings → General
2. Scroll to "Danger Zone"
3. Click "Change repository visibility"
4. Select Private/Public

---

## Post-Setup Checklist

- [ ] Repository created on GitHub (private)
- [ ] Local git initialized
- [ ] .gitignore working correctly
- [ ] Initial commit created
- [ ] Remote added
- [ ] Pushed to GitHub
- [ ] README displays correctly
- [ ] No secrets committed
- [ ] Repository visibility correct (private)
- [ ] Topics/tags added (optional)
- [ ] Branch protection enabled (optional)

---

## Quick Reference Commands

```bash
# Check status
git status

# Add files
git add .

# Commit
git commit -m "message"

# Push
git push

# Create branch
git checkout -b branch-name

# Switch branches
git checkout main

# Pull latest
git pull origin main

# View remotes
git remote -v
```

---

*GitHub Setup Guide - 2025-11-20*
