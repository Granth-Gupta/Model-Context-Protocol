import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const App = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [availableTools, setAvailableTools] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('Connecting...');
  const [showToolsDropdown, setShowToolsDropdown] = useState(false);
  const [selectedTool, setSelectedTool] = useState('');
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const toolsDropdownRef = useRef(null);

  // Improved API base URL configuration
  const API_BASE_URL = process.env.REACT_APP_MCP_CLIENT_URL ||
    (process.env.NODE_ENV === 'production'
        ? 'https://mcp-client-273927490120.us-central1.run.app'  // Fallback to your known client URL
        : 'http://localhost:8000');

  // Auto-scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (toolsDropdownRef.current && !toolsDropdownRef.current.contains(event.target)) {
        setShowToolsDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Initialize MCP client connection
  useEffect(() => {
    const initializeClient = async () => {
      try {
        setConnectionStatus('Connecting...');

        const healthResponse = await fetch(`${API_BASE_URL}/health`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });

        if (!healthResponse.ok) {
          throw new Error('Server health check failed');
        }

        const statusResponse = await fetch(`${API_BASE_URL}/status`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });

        if (!statusResponse.ok) {
          throw new Error('Failed to get system status');
        }

        const status = await statusResponse.json();

        setIsConnected(status.initialized);
        setAvailableTools(status.available_tools.map(tool => tool.name));
        setConnectionStatus(status.initialized ? 'Connected' : 'Disconnected');

        if (status.initialized) {
          addMessage('system', `Welcome! Your MCP Client is ready with ${status.total_tools} powerful tools at your disposal.`);
        } else {
          addMessage('system', 'MCP Client connected but not fully initialized');
        }

      } catch (error) {
        console.error('Failed to initialize MCP client:', error);
        setIsConnected(false);
        setConnectionStatus('Connection Failed');
        addMessage('error', `Connection failed: ${error.message}`);
      }
    };

    initializeClient();

    const statusInterval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) {
          setIsConnected(false);
          setConnectionStatus('Disconnected');
        }
      } catch (error) {
        setIsConnected(false);
        setConnectionStatus('Connection Lost');
      }
    }, 30000);

    return () => clearInterval(statusInterval);
  }, []);

  const addMessage = (type, content, tools = null) => {
    const newMessage = {
      id: Date.now() + Math.random(),
      type,
      content,
      tools,
      timestamp: new Date().toLocaleTimeString()
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = inputMessage.trim();
    setInputMessage('');
    setIsLoading(true);

    addMessage('user', userMessage);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userMessage }),
      });

      if (!response.ok) {
        if (response.status === 503) {
          throw new Error('MCP client not initialized. Please check server status.');
        } else if (response.status === 500) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || 'Internal server error');
        } else {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
      }

      const data = await response.json();
      addMessage('bot', data.response, data.tools_used.length > 0 ? data.tools_used : null);

    } catch (error) {
      console.error('Error sending message:', error);
      addMessage('error', `Failed to send message: ${error.message}`);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e);
    }
  };

  const refreshConnection = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/status`);
      const status = await response.json();

      setIsConnected(status.initialized);
      setAvailableTools(status.available_tools.map(tool => tool.name));
      setConnectionStatus(status.initialized ? 'Connected' : 'Disconnected');

      addMessage('system', `Connection refreshed. Status: ${status.initialized ? 'Connected' : 'Disconnected'}`);
    } catch (error) {
      setIsConnected(false);
      setConnectionStatus('Connection Failed');
      addMessage('error', 'Failed to refresh connection');
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    addMessage('system', 'Chat cleared! Ready for a fresh start.');
  };

  const insertToolSuggestion = (toolName) => {
    setInputMessage(`Please use the ${toolName} tool to `);
    setSelectedTool(toolName);
    setShowToolsDropdown(false);
    inputRef.current?.focus();
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <div className="header-left">
            <div className="logo-section">
              <div className="logo-icon">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none">
                  <path d="M12 2L13.09 8.26L20 9L13.09 9.74L12 16L10.91 9.74L4 9L10.91 8.26L12 2Z" fill="currentColor"/>
                </svg>
              </div>
              <h1 className="app-title">WorkSpace Hub</h1>
            </div>
          </div>

          <div className="header-right">
            <div className="tools-dropdown" ref={toolsDropdownRef}>
              <button
                className={`tools-btn ${availableTools.length > 0 ? 'has-tools' : ''}`}
                onClick={() => setShowToolsDropdown(!showToolsDropdown)}
                disabled={availableTools.length === 0}
              >
                <span className="tools-icon">🔧</span>
                <span className="tools-text">Tools ({availableTools.length})</span>
                <span className={`dropdown-arrow ${showToolsDropdown ? 'rotated' : ''}`}>▼</span>
              </button>

              {showToolsDropdown && availableTools.length > 0 && (
                <div className="tools-dropdown-menu">
                  <div className="dropdown-header">Available Tools</div>
                  {availableTools.map((tool, index) => (
                    <div
                      key={index}
                      className="tool-item"
                      onClick={() => insertToolSuggestion(tool)}
                    >
                      <span className="tool-icon">⚡</span>
                      <span className="tool-name">{tool}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <button
              onClick={refreshConnection}
              className="action-btn"
              disabled={isLoading}
              title="Refresh connection"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z" fill="currentColor"/>
              </svg>
            </button>

            <button
              onClick={clearChat}
              className="action-btn"
              title="Clear chat"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z" fill="currentColor"/>
              </svg>
            </button>
          </div>
        </div>

        <div className="status-bar">
          <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
            <div className="status-dot"></div>
            <span className="status-text">{connectionStatus}</span>
          </div>
          <div className="info-badges">
            <div className="badge ai-badge">AI Powered</div>
            {availableTools.length > 0 && (
              <div className="badge tools-badge">{availableTools.length} Tools Ready</div>
            )}
          </div>
        </div>
      </header>

      <main className="chat-container">
        <div className="messages">
          {messages.length === 0 && (
            <div className="welcome-section">
              <div className="welcome-icon">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
                  <path d="M12 2L13.09 8.26L20 9L13.09 9.74L12 16L10.91 9.74L4 9L10.91 8.26L12 2Z" fill="#4285f4"/>
                </svg>
              </div>
              <h2 className="welcome-title">Welcome to MCP Client</h2>
              <p className="welcome-subtitle">Your intelligent assistant is ready to help</p>
            </div>
          )}

          {messages.map((message) => (
            <div key={message.id} className={`message-wrapper ${message.type}`}>
              <div className="message-bubble">
                <div className="message-header">
                  <div className="message-avatar">
                    {message.type === 'user' ? (
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
                      </svg>
                    ) : message.type === 'bot' ? (
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 2L13.09 8.26L20 9L13.09 9.74L12 16L10.91 9.74L4 9L10.91 8.26L12 2Z"/>
                      </svg>
                    ) : message.type === 'system' ? (
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                      </svg>
                    ) : (
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/>
                      </svg>
                    )}
                  </div>
                  <div className="message-info">
                    <span className="message-sender">
                      {message.type === 'user' ? 'You' :
                       message.type === 'bot' ? 'Assistant' :
                       message.type === 'system' ? 'System' : 'Error'}
                    </span>
                    <span className="message-time">{message.timestamp}</span>
                  </div>
                </div>
                <div className="message-content">
                  <div className="message-text">{message.content}</div>
                </div>
                {message.tools && (
                  <div className="message-tools">
                    <span className="tools-label">Tools used:</span>
                    <div className="tool-chips">
                      {message.tools.map((tool, index) => (
                        <span key={index} className="tool-chip">{tool}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="message-wrapper bot">
              <div className="message-bubble loading">
                <div className="message-header">
                  <div className="message-avatar">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M12 2L13.09 8.26L20 9L13.09 9.74L12 16L10.91 9.74L4 9L10.91 8.26L12 2Z"/>
                    </svg>
                  </div>
                  <div className="message-info">
                    <span className="message-sender">Assistant</span>
                    <span className="message-time">{new Date().toLocaleTimeString()}</span>
                  </div>
                </div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <div className="typing-dots">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                    <span className="typing-text">Thinking...</span>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      <footer className="input-section">
        <div className="input-container">
          <div className="input-wrapper">
            <textarea
              ref={inputRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message here..."
              rows={1}
              disabled={!isConnected || isLoading}
              className="message-input"
            />
            <button
              type="submit"
              onClick={handleSendMessage}
              disabled={!inputMessage.trim() || !isConnected || isLoading}
              className="send-button"
            >
              {isLoading ? (
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="3" fill="currentColor">
                    <animate attributeName="opacity" dur="1s" values="0;1;0" repeatCount="indefinite"/>
                  </circle>
                </svg>
              ) : (
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                  <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" fill="currentColor"/>
                </svg>
              )}
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
