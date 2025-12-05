# GitHub Repository Setup Guide

Follow these steps to create and push your project to GitHub.

## Step 1: Create GitHub Account (if you don't have one)

1. Go to [github.com](https://github.com)
2. Sign up for a free account
3. Verify your email address

## Step 2: Create a New Repository on GitHub

1. Log in to GitHub
2. Click the **"+"** icon in the top right corner
3. Select **"New repository"**
4. Fill in the repository details:
   - **Repository name**: `face07` (or your preferred name)
   - **Description**: "Face Recognition System with Mask Augmentation"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click **"Create repository"**

## Step 3: Initialize Git in Your Local Project

Open PowerShell or Command Prompt in your project directory (`C:\Users\Oshan\Desktop\face07`) and run:

```bash
# Initialize git repository
git init

# Configure your git identity (if not already done)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step 4: Add Files to Git

```bash
# Add all files (respecting .gitignore)
git add .

# Check what will be committed
git status
```

## Step 5: Make Your First Commit

```bash
# Create initial commit
git commit -m "Initial commit: Face Recognition System with Mask Augmentation"
```

## Step 6: Connect to GitHub Repository

After creating the repository on GitHub, you'll see a page with setup instructions. Copy the repository URL (it will look like):
- `https://github.com/yourusername/face07.git` (HTTPS)
- or `git@github.com:yourusername/face07.git` (SSH)

Then run:

```bash
# Add remote repository (replace with your actual URL)
git remote add origin https://github.com/yourusername/face07.git

# Verify remote was added
git remote -v
```

## Step 7: Push to GitHub

```bash
# Push to GitHub (first time)
git branch -M main
git push -u origin main
```

If prompted for credentials:
- **Username**: Your GitHub username
- **Password**: Use a Personal Access Token (not your GitHub password)
  - Create one at: GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
  - Give it `repo` permissions

## Step 8: Verify Upload

1. Go to your GitHub repository page
2. Refresh the page
3. You should see all your files uploaded

## Future Updates

When you make changes to your code:

```bash
# Check what changed
git status

# Add changed files
git add .

# Commit changes
git commit -m "Description of your changes"

# Push to GitHub
git push
```

## Troubleshooting

### If you get authentication errors:
- Use Personal Access Token instead of password
- Or set up SSH keys for easier authentication

### If you need to change remote URL:
```bash
git remote set-url origin https://github.com/yourusername/face07.git
```

### If you need to remove files from git (but keep locally):
```bash
git rm --cached filename
```

## Additional GitHub Features to Consider

1. **Issues**: Track bugs and feature requests
2. **Pull Requests**: Collaborate with others
3. **Releases**: Tag versions of your project
4. **GitHub Pages**: Host documentation or demo
5. **Actions**: Set up CI/CD pipelines

## Quick Command Reference

```bash
git init                          # Initialize repository
git add .                         # Stage all changes
git commit -m "message"           # Commit changes
git push                          # Push to GitHub
git pull                          # Pull latest changes
git status                        # Check repository status
git log                           # View commit history
```

