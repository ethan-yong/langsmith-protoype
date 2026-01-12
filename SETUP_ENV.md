# 🔐 Environment Setup Guide

This guide shows you how to set up your environment variables securely.

## Quick Setup

### Step 1: Create Your .env File

In PowerShell:
```powershell
# Copy the template to create your .env file
Copy-Item .env.template .env
```

Or manually:
1. Copy `.env.template` 
2. Rename it to `.env`

### Step 2: Add Your Hugging Face Token

1. Open `.env` in any text editor (Notepad, VS Code, etc.)
2. Find the line:
   ```
   HF_TOKEN=your_token_here
   ```
3. Replace `your_token_here` with your actual Hugging Face token
4. Save the file

**Example:**
```env
HF_TOKEN=hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890
```

### Step 3: Verify (Optional)

Check that the system can read it:
```powershell
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Token loaded!' if os.getenv('HF_TOKEN') != 'your_token_here' else 'Please update your token')"
```

---

## ✅ Security Features

Your `.env` file is protected by:

1. **`.gitignore`** - Won't be committed to Git
2. **Local only** - Stays on your computer
3. **No sharing** - Never share this file with anyone

---

## 🔑 What Goes in .env?

### Required:
- `HF_TOKEN` - Your Hugging Face access token

### Optional:
- `LANGSMITH_API_KEY` - For LangSmith tracing
- `TESSERACT_CMD` - Path to Tesseract (Windows)
- `POPPLER_PATH` - Path to Poppler (Windows)
- Model and processing configurations

---

## 🚀 Using .env vs Environment Variables

### Method 1: .env File (Recommended)
✅ Easy to manage  
✅ Persistent across sessions  
✅ Organized in one place  

**How it works:**
- The system automatically loads `.env` on startup
- Uses `python-dotenv` package
- No manual export needed

### Method 2: PowerShell Environment Variable
```powershell
# Temporary (current session only)
$env:HF_TOKEN="your_token_here"

# Permanent (all sessions)
[System.Environment]::SetEnvironmentVariable('HF_TOKEN', 'your_token_here', 'User')
```

---

## 📋 Full .env Example

```env
# Required
HF_TOKEN=hf_your_actual_token_here

# Optional - LangSmith
LANGSMITH_API_KEY=ls_your_key_here
LANGCHAIN_TRACING_V2=true

# Optional - Windows Paths
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
POPPLER_PATH=C:\poppler\bin

# Optional - Customization
PDF_DIRECTORY=./documents
CHUNK_SIZE=1500
RETRIEVAL_K=6
```

---

## 🔍 Troubleshooting

### "Token not found" Error

**Check if .env exists:**
```powershell
Test-Path .env
```

**Verify content:**
```powershell
Get-Content .env | Select-String "HF_TOKEN"
```

### .env Not Loading

Make sure:
1. File is named exactly `.env` (not `.env.txt`)
2. File is in the project root directory
3. `python-dotenv` is installed: `pip install python-dotenv`

---

## 🎯 Next Steps

Once your `.env` is set up:

1. ✅ Token is configured
2. ✅ Run the system: `python run_with_existing_pdfs.py`
3. ✅ System automatically uses your token

No need to login or set environment variables manually! 🎉
