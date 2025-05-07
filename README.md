# ParentingBot "ElternLeben Bot"
This repository provides all the files necessary to run the ParentingBot "ElternLeben Bot: Hilfe, wann immer du sie brauchst" in Streamlit locally.

ElternLeben Bot is meant to provide support to parents by providing information and answering parenting-related questions, using as a knowledge source the resources from [ElternLeben.de](https://www.elternleben.de/).


## Step 1: Clone the repository
Close this repository to your local computer. Due to use of a Mock API, a cloud option is not available at this moment.

## Step 2: Activate the Mock API
**Navigate to the API folder**:
   ```bash
   cd mock_api
   ```

**Then run the FastAPI application for the Mock API**:
   ```bash
   uvicorn mock_api:app --reload
   ```
   The API will be available at http://127.0.0.1:8000

## Step 3: Setup the environment

This project uses environment variables to manage sensitive information like API tokens. These variables are stored in a local `.env` file that is **not** included in the repository (for security).


1. **Duplicate the example environment file**:
   ```bash
   cp .env.example .env
   ```

2. **Open the `.env` file** in your code editor and replace placeholder values with your actual credentials:
   ```env
   API_TOKEN=your_huggingface_api_token_here
   ```

3. **Your environment variables** will now be loaded automatically. 

## Step 4: Deploy to streamlit
1. Install streamlit:
   ```bash
   pip install streamlit
   ```
2. Run the app from repository folder
   ```bash
   streamlit run ParentingBot.py

   ```

> **Note**: Files that begin with a dot (like `.env` and `.env.example`) are hidden by default on macOS and Linux.  
> To see them:
> - In **Terminal**, run `ls -a`  
> - In **Finder**, press `Cmd + Shift + .`  
> - In editors like **VS Code**, these files are shown automatically.
