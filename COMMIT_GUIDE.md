# GitHub Commit Guide

## Current Status

✅ Project is ready for GitHub submission with:
- 544 evaluation test cases
- 1,994 SFT training samples
- 1,239 RL training samples (1,199 train + 40 eval)
- Complete English documentation
- Test files excluded from git

## Step-by-Step Commit Process

### 1. Review Files to be Committed

```bash
# Check current directory
pwd
# Should be: /share/project/buyuyan/Agent/power-seeking/opts

# List main files
ls -la

# Check .gitignore is working
git status --ignored
```

### 2. Initialize Git Repository (if not done)

```bash
git init
```

### 3. Configure Git (if needed)

```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### 4. Add Files

```bash
# Add all files (test files will be automatically ignored)
git add .

# Verify what will be committed
git status
```

Expected to be committed:
- ✅ configs/
- ✅ prompts/
- ✅ schemas/
- ✅ scripts/
- ✅ src/
- ✅ evaluation/data/
- ✅ intervention/sft/ (including 1,994 training samples)
- ✅ intervention/rl/ (including 1,239 training samples)
- ✅ README.md
- ✅ CONTRIBUTING.md
- ✅ TESTING.md
- ✅ requirements.txt
- ✅ .gitignore

Should be ignored:
- ❌ test_*.py
- ❌ TEST_*.md
- ❌ SUMMARY.md
- ❌ STATUS.md
- ❌ PROJECT_ORGANIZATION.md
- ❌ __pycache__/
- ❌ *.pyc

### 5. Create Initial Commit

```bash
git commit -m "Initial commit: Agent tool-calling over-privilege tendency research

- Complete case synthesis and evaluation pipeline
- 544 evaluation test cases across 8 domains and 5 escalation types
- SFT intervention with 1,994 training samples
- RL intervention with SLIME integration (1,239 samples)
- Comprehensive English documentation
- Test scripts and examples"
```

### 6. Create GitHub Repository

Go to GitHub and create a new repository:
- Repository name: e.g., `agent-over-privilege`
- Description: "Research on over-privilege tendency in LLM agent tool-calling scenarios"
- Public or Private: Your choice
- Don't initialize with README (we already have one)

### 7. Add Remote and Push

```bash
# Add remote (replace with your actual repository URL)
git remote add origin https://github.com/yourusername/agent-over-privilege.git

# Push to GitHub
git push -u origin main

# If the default branch is 'master' instead of 'main':
# git branch -M main
# git push -u origin main
```

### 8. Verify on GitHub

After pushing, check on GitHub:
- [ ] README.md displays correctly
- [ ] Directory structure is clear
- [ ] Data files are present
- [ ] Test files are NOT present
- [ ] Documentation is readable

## Troubleshooting

### Large Files Warning

If you get warnings about large files:
```bash
# Check file sizes
find . -type f -size +50M

# Our largest file is sft_train.jsonl (~14MB), which is fine
# GitHub allows files up to 100MB
```

### Authentication Issues

If you have authentication issues:
```bash
# Use personal access token instead of password
# Generate token at: https://github.com/settings/tokens

# Or use SSH
git remote set-url origin git@github.com:yourusername/agent-over-privilege.git
```

### Accidentally Committed Test Files

If test files were committed:
```bash
# Remove from git but keep locally
git rm --cached test_*.py TEST_*.md SUMMARY.md STATUS.md PROJECT_ORGANIZATION.md

# Commit the removal
git commit -m "Remove test files from repository"

# Push
git push
```

## Post-Commit Tasks

After successful push:

1. **Add Topics/Tags** on GitHub:
   - llm
   - agent
   - tool-calling
   - security
   - privilege-escalation
   - reinforcement-learning
   - supervised-fine-tuning

2. **Add Description**:
   "Research on over-privilege tendency in LLM agent tool-calling scenarios, including benchmark synthesis, evaluation framework, and intervention methods (SFT & RL)"

3. **Optional: Add LICENSE**:
   - Go to repository → Add file → Create new file
   - Name it `LICENSE`
   - Choose a license template (e.g., MIT, Apache 2.0)

4. **Optional: Create Release**:
   - Go to Releases → Create a new release
   - Tag: v1.0.0
   - Title: "Initial Release"
   - Description: Summary of features

## Quick Reference

```bash
# Complete workflow
git init
git add .
git status  # Verify
git commit -m "Initial commit: Agent tool-calling over-privilege tendency research"
git remote add origin https://github.com/yourusername/repo-name.git
git push -u origin main
```

---

**Ready to commit!** Follow the steps above to push to GitHub.
