import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';
import '../styles/pdfInterface.css';

// Add axios interceptor for authentication
axios.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

function PDFUpload() {
  const [file, setFile] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState('');
  const [pollCount, setPollCount] = useState(0);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [question, setQuestion] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isAsking, setIsAsking] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [documentToDelete, setDocumentToDelete] = useState(null);
  const navigate = useNavigate();
  const { user } = useAuth();
  const [conversationId, setConversationId] = useState(null);

  const MAX_POLLS = 60;
  const POLL_INTERVAL = 5000;

  useEffect(() => {
    let isMounted = true;
    let pollTimeout = null;
    
    const fetchAndUpdate = async () => {
      if (!isMounted) return;
      
      try {
        const response = await axios.get('http://localhost:8000/documents');
        if (response?.data && Array.isArray(response.data)) {
          setDocuments(response.data);
          
          const allProcessed = response.data.every(doc => doc && doc.is_processed);
          if (allProcessed) {
            setPollCount(0);
            return;
          }

          const hasUnprocessedDocs = response.data.some(doc => doc && !doc.is_processed);
          if (hasUnprocessedDocs && pollCount < MAX_POLLS) {
            pollTimeout = setTimeout(() => {
              if (isMounted) {
                setPollCount(prev => prev + 1);
                fetchAndUpdate();
              }
            }, POLL_INTERVAL);
          }
        }
      } catch (err) {
        console.error('Error fetching documents:', err);
        if (isMounted) {
          setError(err.response?.data?.detail || "Failed to fetch documents");
        }
      }
    };

    fetchAndUpdate();
    
    return () => {
      isMounted = false;
      if (pollTimeout) {
        clearTimeout(pollTimeout);
      }
    };
  }, [pollCount]);

  // Add polling for document status
  useEffect(() => {
    const pollInterval = setInterval(() => {
      if (documents.some(doc => 
        doc.processing_status === 'processing' || 
        doc.processing_status === 'extracting_text' || 
        doc.processing_status === 'creating_index'
      )) {
        fetchAndUpdate();
      }
    }, 5000); // Poll every 5 seconds

    return () => clearInterval(pollInterval);
  }, [documents]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile);
      setError('');
    } else {
      setError('Please select a valid PDF file');
      setFile(null);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file to upload');
      return;
    }

    setIsUploading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/upload', formData);
      setFile(null);
      e.target.reset();
      fetchDocuments();
    } catch (error) {
      setError('Failed to upload file. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleAsk = async (e) => {
    e.preventDefault();
    if (!question.trim() || !selectedDocument) return;

    setIsAsking(true);
    try {
      const response = await axios.post('http://localhost:8000/ask', {
        document_id: selectedDocument.id,
        question: question.trim()
      });

      setChatHistory(prev => [...prev, {
        question: question.trim(),
        answer: response.data.answer,
        timestamp: new Date().toISOString()
      }]);

      setQuestion('');
    } catch (error) {
      setError('Failed to get answer. Please try again.');
    } finally {
      setIsAsking(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleAsk(e);
    }
  };

  const fetchDocuments = async () => {
    try {
      const response = await axios.get('http://localhost:8000/documents');
      setDocuments(response.data);
    } catch (error) {
      console.error('Error fetching documents:', error);
    }
  };

  const fetchAndUpdate = async () => {
    try {
      const response = await axios.get('http://localhost:8000/documents');
      setDocuments(response.data);
      
      // Update selected document if it exists
      if (selectedDocument) {
        const updatedDoc = response.data.find(doc => doc.id === selectedDocument.id);
        if (updatedDoc) {
          setSelectedDocument(updatedDoc);
          // If document is now processed, fetch conversation history
          if (updatedDoc.is_processed && !selectedDocument.is_processed) {
            fetchConversationHistory(updatedDoc.id);
          }
        }
      }
    } catch (error) {
      console.error('Error fetching documents:', error);
    }
  };

  const handleViewDocument = (documentId) => {
    navigate(`/view/${documentId}`);
  };

  const handleDeleteClick = (doc) => {
    setDocumentToDelete(doc);
  };

  const handleDeleteConfirm = async () => {
    if (!documentToDelete) return;

    setIsDeleting(true);
    try {
      await axios.delete(`http://localhost:8000/documents/${documentToDelete.id}`);
      setDocuments(documents.filter(doc => doc.id !== documentToDelete.id));
      if (selectedDocument?.id === documentToDelete.id) {
        setSelectedDocument(null);
        setChatHistory([]);
      }
      setError('');
    } catch (error) {
      setError('Failed to delete document. Please try again.');
    } finally {
      setIsDeleting(false);
      setDocumentToDelete(null);
    }
  };

  const handleDeleteCancel = () => {
    setDocumentToDelete(null);
  };

  const renderDocumentStatus = (doc) => {
    if (!doc) return null;
    
    if (doc.is_processed) {
      return null;
    }

    let statusText = "Processing...";
    let statusClass = "processing";
    
    if (doc.processing_status === "extracting_text") {
      statusText = "Extracting text...";
      statusClass = "extracting";
    } else if (doc.processing_status === "creating_index") {
      statusText = "Creating search index...";
      statusClass = "indexing";
    } else if (doc.processing_status === "failed") {
      statusText = `Processing failed: ${doc.processing_error || "Unknown error"}`;
      statusClass = "error";
    }

    return (
      <span className={`processing-status ${statusClass}`}>
        {statusText}
      </span>
    );
  };

  const fetchConversationHistory = async (docId) => {
    try {
      const response = await axios.get(`http://localhost:8000/conversations/${docId}`);
      const conversations = response.data.map(conv => ({
        type: 'question',
        content: conv.question,
        timestamp: new Date(conv.timestamp)
      })).concat(response.data.map(conv => ({
        type: 'answer',
        content: conv.answer,
        timestamp: new Date(conv.timestamp)
      }))).sort((a, b) => a.timestamp - b.timestamp);
      
      setChatHistory(conversations);
    } catch (error) {
      console.error('Error fetching conversation history:', error);
      setError('Failed to load conversation history');
    }
  };

  const handleDocumentSelect = async (doc) => {
    setSelectedDocument(doc);
    setConversationId(null); // Reset conversation when selecting a new document
    setChatHistory([]); // Clear chat history for new document
    
    if (doc.is_processed) {
      try {
        const response = await axios.get(`http://localhost:8000/conversations/${doc.id}`);
        if (response.data.length > 0) {
          // Get the latest conversation
          const latestConversation = response.data[0];
          setConversationId(latestConversation.id);
          
          // Format chat history from conversation messages
          const formattedHistory = latestConversation.messages.map(msg => ({
            role: msg.role,
            content: msg.content,
            timestamp: msg.timestamp
          }));
          setChatHistory(formattedHistory);
        }
      } catch (error) {
        console.error('Error fetching conversation history:', error);
      }
    }
  };

  const handleAskQuestion = async () => {
    if (!question.trim() || !selectedDocument) return;

    setIsAsking(true);
    setError('');

    try {
      // Add user's question to chat history immediately
      const userMessage = {
        role: 'user',
        content: question.trim(),
        timestamp: new Date().toISOString()
      };
      
      // Store the current question before clearing it
      const currentQuestion = question.trim();
      setQuestion(''); // Clear the input immediately
      
      // Update chat history with user's question
      setChatHistory(prevHistory => [...prevHistory, userMessage]);

      // Prepare the request payload
      const requestPayload = {
        document_id: selectedDocument.id,
        question: currentQuestion
      };

      // Only add conversation_id if it exists
      if (conversationId) {
        requestPayload.conversation_id = conversationId;
      }

      const response = await axios.post('http://localhost:8000/ask', requestPayload);

      // Add assistant's response to chat history
      const assistantMessage = {
        role: 'assistant',
        content: response.data.answer,
        timestamp: new Date().toISOString()
      };
      
      // Update chat history with assistant's response
      setChatHistory(prevHistory => [...prevHistory, assistantMessage]);
      
      // Update conversation ID if this is a new conversation
      if (response.data.conversation_id) {
        setConversationId(response.data.conversation_id);
      }
    } catch (error) {
      console.error('Error asking question:', error);
      setError(error.response?.data?.detail || 'Failed to get answer');
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <div className="pdf-interface">
      <div className="pdf-header">
        <h1>PDF Q&A System</h1>
        <div className="user-info">
          <span className="welcome-text">Welcome,</span>
          <span className="username">{user?.username || 'User'}</span>
        </div>
      </div>

      <div className="pdf-container">
        <div className="upload-section">
          <h2>Upload PDF</h2>
          <form onSubmit={handleUpload} className="upload-form">
            <div className="file-input-container">
              <input
                type="file"
                accept=".pdf"
                onChange={handleFileChange}
                className="file-input"
                id="file-input"
              />
              <label htmlFor="file-input" className="file-input-label">
                {file ? file.name : 'Choose PDF file'}
              </label>
            </div>
            <button 
              type="submit" 
              className="upload-button"
              disabled={isUploading || !file}
            >
              {isUploading ? 'Uploading...' : 'Upload'}
            </button>
          </form>
          {error && <div className="error-message">{error}</div>}
        </div>

        <div className="documents-section">
          <h2>Your Documents</h2>
          <div className="documents-list">
            {documents.map((doc) => (
              <div 
                key={doc.id} 
                className={`document-item ${selectedDocument?.id === doc.id ? 'selected' : ''}`}
                onClick={() => handleDocumentSelect(doc)}
              >
                <div className="document-info">
                  <h3>{doc.filename}</h3>
                  {renderDocumentStatus(doc)}
                </div>
                <div className="document-actions">
                  {doc.is_processed && (
                    <button 
                      className="select-button"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDocumentSelect(doc);
                      }}
                    >
                      Select
                    </button>
                  )}
                  <button
                    className="delete-button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteClick(doc);
                    }}
                    disabled={isDeleting}
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="qa-section">
        <h2>Ask Questions</h2>
        {selectedDocument ? (
          <>
            <div className="chat-history">
              {chatHistory.map((message, index) => (
                <div key={index} className={`chat-message ${message.role}`}>
                  <div className="message-content">
                    {message.role === 'user' ? 'Q: ' : 'A: '}
                    {message.content}
                  </div>
                  <div className="message-timestamp">
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              ))}
              {isAsking && (
                <div className="chat-message assistant">
                  <div className="message-content">
                    Thinking...
                  </div>
                </div>
              )}
            </div>
            <div className="question-input">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask a question about the document..."
                disabled={!selectedDocument || isAsking}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleAskQuestion();
                  }
                }}
              />
              <button 
                onClick={handleAskQuestion}
                disabled={!selectedDocument || !question.trim() || isAsking}
              >
                {isAsking ? 'Asking...' : 'Ask'}
              </button>
            </div>
          </>
        ) : (
          <div className="no-document-selected">
            Select a document to start asking questions
          </div>
        )}
      </div>

      {documentToDelete && (
        <div className="delete-modal">
          <div className="delete-modal-content">
            <h3>Delete Document</h3>
            <p>Are you sure you want to delete "{documentToDelete.filename}"?</p>
            <div className="delete-modal-actions">
              <button 
                className="cancel-button"
                onClick={handleDeleteCancel}
                disabled={isDeleting}
              >
                Cancel
              </button>
              <button 
                className="confirm-delete-button"
                onClick={handleDeleteConfirm}
                disabled={isDeleting}
              >
                {isDeleting ? 'Deleting...' : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default PDFUpload; 