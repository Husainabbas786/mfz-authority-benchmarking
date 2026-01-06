# Setting Up OpenAI API Key

This application requires an OpenAI API key to generate embeddings for activity matching.

## Option 1: Environment Variable (Recommended for Local Development)

### Windows (Command Prompt):
```cmd
set OPENAI_API_KEY=your_api_key_here
```

### Windows (PowerShell):
```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

### Linux/Mac:
```bash
export OPENAI_API_KEY=your_api_key_here
```

**Note:** These settings are temporary and will be lost when you close the terminal. To make them permanent, add them to your system's environment variables.

## Option 2: Streamlit Secrets (For Streamlit Cloud Deployment)

1. Go to your Streamlit Cloud app settings
2. Navigate to "Secrets"
3. Add the following:
```toml
OPENAI_API_KEY = "your_api_key_here"
```

## Option 3: Create a `.env` file (Local Development)

1. Create a file named `.env` in the project root
2. Add:
```
OPENAI_API_KEY=your_api_key_here
```

**Note:** The `.env` file is already in `.gitignore` and won't be committed to GitHub.

## Getting Your OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (you won't be able to see it again!)

## Security Reminder

⚠️ **Never commit your API key to GitHub!** The key has been removed from the code and should only be set as an environment variable or in Streamlit secrets.

