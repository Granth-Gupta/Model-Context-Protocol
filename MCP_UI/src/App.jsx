import React, { useState, useEffect } from 'react';
import './App.css';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_MCP_CLIENT_URL ||
    (import.meta.env.MODE === 'production'
        ? 'https://mcp-client-273927490120.us-central1.run.app'
        : 'http://localhost:8000');

function App() {
  // State management
  const [isLoading, setIsLoading] = useState(true);
  const [apiStatus, setApiStatus] = useState('connecting');
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState(null);
  const [serverHealth, setServerHealth] = useState(null);
  const [showHealthDetails, setShowHealthDetails] = useState(false);

  // Test API connection on component mount
  useEffect(() => {
    console.log('App component mounting...');
    console.log('API_BASE_URL:', API_BASE_URL);
    testConnection();
  }, []);

  const testConnection = async () => {
    try {
      setIsLoading(true);
      setError(null);

      console.log('Testing connection to:', `${API_BASE_URL}/health`);
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(10000)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const healthData = await response.json();
      console.log('Health check response:', healthData);

      setServerHealth(healthData);
      setApiStatus('connected');
      setIsLoading(false);

      setChatHistory([
        {
          type: 'system',
          message: `Welcome to MCP Client! 🎉\n\nSystem Status: ${healthData.status}\nEmployee Server: ${healthData.employees_connected ? '✅ Online' : '❌ Offline'}\nLeave Server: ${healthData.leave_connected ? '✅ Online' : '❌ Offline'}\n\nYou can now ask questions about employees and leave management!`,
          timestamp: new Date().toLocaleTimeString()
        }
      ]);

    } catch (error) {
      console.error('API connection failed:', error);
      setApiStatus('failed');
      setError(error.message);
      setIsLoading(false);

      setChatHistory([
        {
          type: 'error',
          message: `Connection Failed 🔴\n\nCouldn't connect to MCP Client API.\nError: ${error.message}\n\nPlease ensure the backend is running on:\n${API_BASE_URL}`,
          timestamp: new Date().toLocaleTimeString()
        }
      ]);
    }
  };

  const sendMessage = async () => {
    if (!message.trim()) return;

    const userMessage = message.trim();
    setMessage('');
    setIsTyping(true);

    setChatHistory(prev => [...prev, {
      type: 'user',
      message: userMessage,
      timestamp: new Date().toLocaleTimeString()
    }]);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage }),
        signal: AbortSignal.timeout(30000)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      setChatHistory(prev => [...prev, {
        type: 'assistant',
        message: data.response,
        tools_used: data.tools_used || [],
        timestamp: new Date().toLocaleTimeString()
      }]);

    } catch (error) {
      console.error('Chat request failed:', error);
      setChatHistory(prev => [...prev, {
        type: 'error',
        message: `Failed to send message: ${error.message}`,
        timestamp: new Date().toLocaleTimeString()
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setChatHistory([]);
  };

  const refreshConnection = () => {
    setChatHistory([]);
    testConnection();
  };

  // Loading state with modern spinner
  if (isLoading) {
    return (
      <div className="loading-screen">
        <div className="loading-content">
          <div className="modern-spinner">
            <div className="spinner-ring"></div>
            <div className="spinner-ring"></div>
            <div className="spinner-ring"></div>
          </div>
          <h2>Connecting to MCP System</h2>
          <p>Establishing secure connection...</p>
          <div className="connection-details">
            <code>{API_BASE_URL}</code>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      {/* Modern Header */}
      <header className="header">
        <div className="header-content">
          <div className="brand">
            <div className="brand-icon">🤖</div>
            <div className="brand-text">
              <h1>MCP Assistant</h1>
              <span>Model Context Protocol Client</span>
            </div>
          </div>

          <div className="header-controls">
            <div className="status-card">
              <div className={`status-indicator ${apiStatus}`}>
                <div className="status-dot"></div>
                <span>{apiStatus === 'connected' ? 'Online' : 'Offline'}</span>
              </div>

              {serverHealth && (
                <button
                  className="info-btn"
                  onClick={() => setShowHealthDetails(!showHealthDetails)}
                  title="System Information"
                >
                  ℹ️
                </button>
              )}

              <button
                className="refresh-btn"
                onClick={refreshConnection}
                title="Refresh Connection"
              >
                🔄
              </button>
            </div>
          </div>
        </div>

        {/* Health Details Dropdown */}
        {showHealthDetails && serverHealth && (
          <div className="health-dropdown">
            <div className="health-grid">
              <div className="health-item">
                <span className="health-label">Service</span>
                <span className="health-value">{serverHealth.service}</span>
              </div>
              <div className="health-item">
                <span className="health-label">Port</span>
                <span className="health-value">{serverHealth.port}</span>
              </div>
              <div className="health-item">
                <span className="health-label">Employee Server</span>
                <span className={`health-status ${serverHealth.employees_connected ? 'online' : 'offline'}`}>
                  {serverHealth.employees_connected ? '✅ Connected' : '❌ Disconnected'}
                </span>
              </div>
              <div className="health-item">
                <span className="health-label">Leave Server</span>
                <span className={`health-status ${serverHealth.leave_connected ? 'online' : 'offline'}`}>
                  {serverHealth.leave_connected ? '✅ Connected' : '❌ Disconnected'}
                </span>
              </div>
            </div>
          </div>
        )}
      </header>

      {/* Error Banner */}
      {error && (
        <div className="error-banner">
          <div className="error-content">
            <span className="error-icon">⚠️</span>
            <div className="error-text">
              <strong>Connection Error</strong>
              <p>{error}</p>
            </div>
            <button onClick={refreshConnection} className="error-retry-btn">
              Try Again
            </button>
          </div>
        </div>
      )}

      {/* Main Chat Interface */}
      <main className="main-content">
        <div className="chat-container">

          {/* Chat Header */}
          <div className="chat-header">
            <h2>💬 Chat Assistant</h2>
            <div className="chat-controls">
              <button onClick={clearChat} className="clear-btn" title="Clear Chat">
                🗑️ Clear
              </button>
            </div>
          </div>

          {/* Messages Area */}
          <div className="messages-container">
            {chatHistory.length === 0 ? (
              <div className="welcome-screen">
                <div className="welcome-icon">🎯</div>
                <h3>Welcome to MCP Assistant!</h3>
                <p>I can help you with employee and leave management tasks.</p>

                <div className="example-queries">
                  <h4>Try asking:</h4>
                  <div className="query-examples">
                    <button className="example-btn" onClick={() => setMessage("Show me employee E001's information")}>
                      👤 Employee Info
                    </button>
                    <button className="example-btn" onClick={() => setMessage("Find employees in Engineering department")}>
                      🔍 Department Search
                    </button>
                    <button className="example-btn" onClick={() => setMessage("How many leave days does E002 have?")}>
                      📅 Leave Balance
                    </button>
                    <button className="example-btn" onClick={() => setMessage("Apply leave for E001 on 2025-08-15")}>
                      ✍️ Apply Leave
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="messages-list">
                {chatHistory.map((chat, index) => (
                  <div key={index} className={`message-bubble ${chat.type}`}>
                    <div className="message-header">
                      <div className="message-avatar">
                        {chat.type === 'user' ? '👤' :
                         chat.type === 'assistant' ? '🤖' :
                         chat.type === 'system' ? '🔧' : '⚠️'}
                      </div>
                      <div className="message-info">
                        <span className="message-sender">
                          {chat.type === 'user' ? 'You' :
                           chat.type === 'assistant' ? 'Assistant' :
                           chat.type === 'system' ? 'System' : 'Error'}
                        </span>
                        <span className="message-time">{chat.timestamp}</span>
                      </div>
                    </div>

                    <div className="message-content">
                      {chat.message.split('\n').map((line, i) => (
                        <div key={i} className="message-line">{line}</div>
                      ))}
                    </div>

                    {chat.tools_used && chat.tools_used.length > 0 && (
                      <div className="tools-used">
                        <span className="tools-label">🔧 Tools:</span>
                        <div className="tools-list">
                          {chat.tools_used.map((tool, i) => (
                            <span key={i} className="tool-tag">{tool}</span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}

                {/* Typing Indicator */}
                {isTyping && (
                  <div className="message-bubble assistant typing">
                    <div className="message-header">
                      <div className="message-avatar">🤖</div>
                      <div className="message-info">
                        <span className="message-sender">Assistant</span>
                      </div>
                    </div>
                    <div className="typing-animation">
                      <div className="typing-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                      <span className="typing-text">Thinking...</span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Input Area */}
          <div className="input-area">
            <div className="input-container">
              <div className="input-wrapper">
                <textarea
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder={apiStatus === 'connected'
                    ? "Ask me anything about employees or leave management..."
                    : "Please wait for connection..."}
                  className="message-input"
                  disabled={apiStatus !== 'connected' || isTyping}
                  rows="1"
                />
                <button
                  onClick={sendMessage}
                  disabled={!message.trim() || apiStatus !== 'connected' || isTyping}
                  className="send-button"
                  title="Send Message"
                >
                  {isTyping ? '⏳' : '➤'}
                </button>
              </div>
              <div className="input-hint">
                Press Enter to send • Shift+Enter for new line
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Modern Footer */}
      <footer className="footer">
        <div className="footer-content">
          <div className="footer-info">
            <span>🔗 API: <code>{API_BASE_URL.replace('https://', '').replace('http://', '')}</code></span>
            <span>•</span>
            <span>Environment: <code>{import.meta.env.MODE || 'development'}</code></span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
