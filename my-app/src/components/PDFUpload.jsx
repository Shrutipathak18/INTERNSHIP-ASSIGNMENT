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

  const handleDocumentSelect = (doc) => {
    setSelectedDocument(doc);
    setQuestion('');
    setChatHistory([]);
    if (doc.is_processed) {
      fetchConversationHistory(doc.id);
    }
  };

  const handleAskQuestion = async (e) => {
    e.preventDefault();
    if (!question.trim() || !selectedDocument) return;

    setIsAsking(true);
    setError('');

    try {
      const response = await axios.post('http://localhost:8000/ask', {
        document_id: selectedDocument.id,
        question: question,
        conversation_id: conversationId
      });

      setChatHistory(prev => [...prev, 
        { type: 'question', content: question, timestamp: new Date() },
        { type: 'answer', content: response.data.answer, timestamp: new Date() }
      ]);
      setQuestion('');
    } catch (error) {
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
            <div className="chat-container">
              {chatHistory.map((message, index) => (
                <div key={index} className={`chat-message ${message.type}`}>
                  <div className="message-content">{message.content}</div>
                  <div className="message-timestamp">
                    {message.timestamp instanceof Date ? 
                      message.timestamp.toLocaleTimeString() : 
                      new Date(message.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              ))}
            </div>
            <form onSubmit={handleAskQuestion} className="question-form">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask a question about the document..."
                className="question-input"
                disabled={isAsking || !selectedDocument.is_processed}
              />
              <button 
                type="submit" 
                className="ask-button"
                disabled={!question.trim() || isAsking || !selectedDocument.is_processed}
              >
                {isAsking ? 'Asking...' : 'Ask'}
              </button>
            </form>
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