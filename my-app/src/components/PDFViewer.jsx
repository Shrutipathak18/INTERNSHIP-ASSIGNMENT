import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

const PDFViewer = () => {
  const { documentId } = useParams();
  const navigate = useNavigate();
  const [question, setQuestion] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (documentId) {
      fetchConversationHistory(documentId);
    }
  }, [documentId]);

  const fetchConversationHistory = async (documentId) => {
    try {
      const response = await axios.get(`${API_URL}/conversations/${documentId}`);
      if (response?.data && Array.isArray(response.data)) {
        setChatHistory(response.data);
        if (response.data.length > 0) {
          const lastChat = response.data[response.data.length - 1];
          if (lastChat && lastChat.id) {
            setCurrentConversationId(lastChat.id);
          }
        }
      }
    } catch (err) {
      console.error('Error fetching conversation history:', err);
      setError(err.response?.data?.detail || "Failed to fetch conversation history");
    }
  };

  const handleAsk = async (e) => {
    e.preventDefault();
    if (!question.trim() || !documentId) return;

    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_URL}/ask`, {
        document_id: documentId,
        question: question.trim(),
        conversation_id: currentConversationId
      });

      if (response?.data) {
        const newChat = {
          id: response.data.conversation_id,
          question: question.trim(),
          answer: response.data.answer,
          timestamp: new Date().toISOString()
        };
        
        setChatHistory(prev => [...(prev || []), newChat]);
        if (response.data.conversation_id) {
          setCurrentConversationId(response.data.conversation_id);
        }
        setQuestion('');
      }
    } catch (err) {
      const errorMessage =
        err.response?.data?.detail ||
        (typeof err.response?.data === "string"
          ? err.response.data
          : "Failed to get answer");
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk(e);
    }
  };

  return (
    <div className="app-container">
      <div className="background-shapes">
        <div className="shape one"></div>
        <div className="shape two"></div>
      </div>

      <div className="container">
        <div className="app-header">
          <button onClick={() => navigate('/')} className="back-button">
            ← Back to Documents
          </button>
          <h1 className="title">Ask Questions</h1>
        </div>

        {error && <div className="error-message">{error}</div>}

        <div className="section">
          <form onSubmit={handleAsk} className="question-form">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your question here..."
              className="input-text"
            />
            <button type="submit" disabled={loading} className="button">
              {loading ? "Waiting for answer..." : "Ask"}
            </button>
          </form>
        </div>

        <div className="section chat-history">
          <h2>Chat History</h2>
          {chatHistory.length === 0 ? (
            <p>No questions asked yet.</p>
          ) : (
            chatHistory.map((chat) => (
              <div key={chat.id} className="chat-item">
                <p className="question"><strong>Q:</strong> {chat.question}</p>
                <p className="answer"><strong>A:</strong> {chat.answer}</p>
                <p className="timestamp">{new Date(chat.timestamp).toLocaleString()}</p>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default PDFViewer; 