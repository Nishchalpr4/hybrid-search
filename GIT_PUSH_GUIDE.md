# Git Setup & Push Guide

## Step 1: Verify You Have Git

```bash
git --version
```

If not installed, download from: https://git-scm.com/

## Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Fill in:
   - **Repository name:** `hybrid-search`
   - **Description:** "Working prototype: Hybrid search combining BM25 + semantic embeddings"
   - **Public** (so others can see it)
   - **Do NOT** check "Initialize with README" (we already have one)
3. Click "Create repository"

You'll get a page with your repo URL. Copy it.

## Step 3: Open PowerShell & Navigate

```powershell
cd "f:\Search Optimization"
```

## Step 4: Initialize Git (One Time)

```bash
git init
git config user.name "Your Name"
git config user.email "your.email@github.com"
```

## Step 5: Add Files

```bash
git add .
```

Check what's being added:
```bash
git status
```

You should see files like:
```
quick_demo.py
eval_demo.py
search_pipeline.py
generate_data.py
eval.py
data_books.json
eval_queries.json
requirements.txt
README.md
.gitignore
```

## Step 6: Create First Commit

```bash
git commit -m "Initial commit: Hybrid search prototype - BM25 + semantic embeddings"
```

## Step 7: Rename Branch to Main

```bash
git branch -M main
```

## Step 8: Add Remote Repository

Replace `yourusername` with your GitHub username:

```bash
git remote add origin https://github.com/yourusername/hybrid-search.git
```

Verify it worked:
```bash
git remote -v
```

Should show:
```
origin  https://github.com/yourusername/hybrid-search.git (fetch)
origin  https://github.com/yourusername/hybrid-search.git (push)
```

## Step 9: Push to GitHub

```bash
git push -u origin main
```

First time will ask for credentials:
- Username: Your GitHub username
- Password: Your GitHub Personal Access Token (not your password!)

### Getting a Personal Access Token

1. Go to https://github.com/settings/tokens/new
2. Click "Generate new token"
3. Check: `repo` (full control of private repositories)
4. Set expiration to "90 days"
5. Click "Generate token"
6. Copy the token
7. Use as password when pushing

## Done! 🎉

Your repo is now on GitHub at:
```
https://github.com/yourusername/hybrid-search
```

## Future Pushes

After first push, subsequent commits are easy:

```bash
git add .
git commit -m "Your message"
git push origin main
```

## If Something Goes Wrong

**"fatal: not a git repository"**
```bash
git init
```

**"permission denied"**
Make sure you created a Personal Access Token (see above)

**"fatal: 'origin' does not appear to be a 'git' repository"**
```bash
git remote add origin https://github.com/yourusername/hybrid-search.git
```

**See current status**
```bash
git status
```

**See commit history**
```bash
git log --oneline
```

## What Gets Uploaded

✅ Uploaded:
- quick_demo.py, eval_demo.py, etc. (all .py files)
- data_books.json, eval_queries.json (data files)
- requirements.txt, .gitignore, README.md
- All code

❌ NOT uploaded (.gitignore prevents it):
- cache/*.npy (embeddings file, auto-generated)
- venv/ folder (virtual environment)
- __pycache__/ (Python cache)
- .pyc files

This keeps your repo lean (~2 MB instead of 500 MB).

## Sharing Your Repo

Once pushed, share the link:
```
https://github.com/yourusername/hybrid-search
```

Others can clone with:
```bash
git clone https://github.com/yourusername/hybrid-search.git
cd hybrid-search
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python quick_demo.py
```
