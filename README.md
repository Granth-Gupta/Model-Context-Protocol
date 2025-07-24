# Model Context Protocol - Setup Guide
This repository contains a Model Context Protocol (MCP) implementation with three main components: MCP servers, MCP client, and MCP UI. Follow the steps below to set up and run the complete system.

Prerequisites
Python 3.8+ with uv package manager

Node.js 16+ with npm

Git

Quick Start
1. Clone the Repository
bash
git clone https://github.com/Granth-Gupta/Model-Context-Protocol.git
cd Model-Context-Protocol
MCP Servers
The MCP servers are already deployed and accessible via the following URLs:

Employee Server: https://mcp-server-employee-273927490120.us-central1.run.app/

Leaving Server: https://mcp-server-leaving-273927490120.us-central1.run.app/

These servers are ready to use and don't require local setup.

MCP Client
2. Set Up Python Environment
Navigate to the MCP client directory and install dependencies:

bash
cd MCP_client
pip install -r requirements.txt
3. Run the MCP Client
bash
uv run main.py
The client will connect to the deployed MCP servers and provide the interface for interacting with the Model Context Protocol.

MCP UI
4. Set Up Node.js Environment
Navigate to the UI directory and install dependencies:

bash
cd MCP_UI
npm install
5. Run the Development Server
bash
npm run dev
The UI will be available at http://localhost:3000 (or the port specified in the console output).
