# ParentingBot "ElternLeben Bot"
This repository provides all the files necessary to run the ParentingBot "ElternLeben Bot: Hilfe, wann immer du sie brauchst" in Streamlit locally using Python.

ElternLeben Bot is meant to provide support to parents by providing information and answering parenting-related questions, using as a knowledge source the resources from [ElternLeben.de](https://www.elternleben.de/).


## Step 1: Get a Hugging Face API Token

To run the ElternLeben Bot you’ll need a Hugging Face API token. Following are the steps to get one.

1. **Create a Hugging Face account**  
   Sign up at [https://huggingface.co/join](https://huggingface.co/join).

2. **Go to your access tokens page**  
   Visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) while logged in.

3. **Generate a new token**  
   - Click **"New token"**.
   - Give your token a name (e.g., `streamlit-app-token`).
   - Set the **role** to **Read**.
   - Click **"Generate"** to create the token.

4. **Copy and store the token**  
   You’ll need this token to access Hugging Face models via the API. Paste it into your .env file as shown in the example .env example file. 


## Step 2: Clone the repository
Clone this repository to your local computer. Due to use of a Mock API, a cloud option (such as deploying to [streamlit.io](streamlit.io)) is not available at this moment.

## Step 3: Activate the Mock API
**Navigate to the API folder**:
   ```bash
   cd mock_api
   ```

**Then run the FastAPI application for the Mock API**:
   ```bash
   uvicorn mock_api:app --reload
   ```
   The API will be available at http://127.0.0.1:8000

## Step 4: Setup the environment

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

## Step 5: Deploy to streamlit
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
