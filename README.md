# ParentingBot
ParentingBot answering parenting questions based on resources from ElternLeben.de

## Mock API
**Navigate to the API folder**:
   ```bash
   cd mock_api
   ```

**Run the FastAPI application**:
   ```bash
   uvicorn mock_api:app --reload
   ```
   The API will be available at http://127.0.0.1:8000

## Environment Setup

This project uses environment variables to manage sensitive information like API tokens. These variables are stored in a local `.env` file that is **not** included in the repository (for security).

To get started:

1. **Duplicate the example environment file**:
   ```bash
   cp .env.example .env
   ```

2. **Open the `.env` file** in your code editor and replace placeholder values with your actual credentials:
   ```env
   API_TOKEN=your_huggingface_api_token_here
   ```

3. You're ready to run the project! Your environment variables will now be loaded automatically.

> **Note**: Files that begin with a dot (like `.env` and `.env.example`) are hidden by default on macOS and Linux.  
> To see them:
> - In **Terminal**, run `ls -a`  
> - In **Finder**, press `Cmd + Shift + .`  
> - In editors like **VS Code**, these files are shown automatically.

---

**⚠️ Do not commit your `.env` file to version control.**  
It's already listed in `.gitignore` to protect your secrets.
