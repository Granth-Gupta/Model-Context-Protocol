# 🚀 Model Context Protocol - Setup Guide

This repository contains a **Model Context Protocol (MCP)** implementation with three main components: MCP servers, MCP client, and MCP UI. Follow the steps below to set up and run the complete system.

## 📋 Prerequisites

- **Python 3.8+** with `uv` package manager
- **Node.js 16+** with `npm`
- **Git**

## ⚡ Quick Start

### 1. Clone the Repository

git clone https://github.com/Granth-Gupta/Model-Context-Protocol.git
cd Model-Context-Protocol


---

## 🖥️ MCP Servers

The MCP servers are already deployed and accessible via the following URLs:

> **📡 Employee Server**: `https://mcp-server-employee-273927490120.us-central1.run.app/`
> 
> **📡 Leaving Server**: `https://mcp-server-leaving-273927490120.us-central1.run.app/`

✅ **These servers are ready to use and don't require local setup.**

---

## 🐍 MCP Client

### 2. Set Up Python Environment

Navigate to the MCP client directory and install dependencies:

cd MCP_client
pip install -r requirements.txt


### 3. Run the MCP Client

uv run main.py


💡 The client will connect to the deployed MCP servers and provide the interface for interacting with the Model Context Protocol.

---

## 🎨 MCP UI

### 4. Set Up Node.js Environment

Navigate to the UI directory and install dependencies:

cd MCP_UI
npm install


### 5. Run the Development Server

npm run dev


🌐 The UI will be available at `http://localhost:5173` (or the port specified in the console output).

---

## 📝 Complete Setup Workflow

Follow these steps in order:

1. **📥 Clone Repository**

git clone https://github.com/Granth-Gupta/Model-Context-Protocol.git
cd Model-Context-Protocol


2. **🐍 Set Up MCP Client**

cd MCP_client
pip install -r requirements.txt
uv run main.py

3. **🎨 Set Up MCP UI** *(in a new terminal)*
cd MCP_UI
npm install
npm run dev

---

## 🔧 Server Configuration

The system is configured to use the following server endpoints:

| 🏷️ Component | 🌐 URL |
|---------------|--------|
| **Employee Server** | `https://mcp-server-employee-273927490120.us-central1.run.app/` |
| **Leaving Server** | `https://mcp-server-leaving-273927490120.us-central1.run.app/` |

---

## 🛠️ Troubleshooting

Common issues and solutions:

- **🐍 Python Dependencies**: Ensure you have `uv` installed. If not, install it with:
pip install uv

- **📦 Node.js Issues**: Make sure you're using Node.js 16+ and npm is properly installed:
node --version
npm --version

- **🚪 Port Conflicts**: If the default ports are in use, the applications will automatically use alternative ports

---

## 🎯 Usage

Once all components are running:

1. 🔄 The **MCP Client** handles the core protocol communication
2. 🖥️ The **MCP UI** provides a user-friendly interface for interaction  
3. 🌐 The **MCP Servers** process requests and return responses

🚀 **Access the web interface** through your browser at the URL displayed when running `npm run dev` to start using the Model Context Protocol system.

<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/0df0a356-a989-4544-8619-6cb4e77bd963" />

