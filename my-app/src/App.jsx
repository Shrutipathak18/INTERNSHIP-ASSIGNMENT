import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import Navbar from './components/Navbar';
import Login from './pages/Login';
import Register from './pages/Register';
import PDFUpload from './components/PDFUpload';
import PDFViewer from './components/PDFViewer';
import ErrorBoundary from './components/ErrorBoundary';

function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <>
                  <Navbar />
                  <ErrorBoundary>
                    <PDFUpload />
                  </ErrorBoundary>
                </>
              </ProtectedRoute>
            }
          />
          <Route
            path="/view/:documentId"
            element={
              <ProtectedRoute>
                <>
                  <Navbar />
                  <ErrorBoundary>
                    <PDFViewer />
                  </ErrorBoundary>
                </>
              </ProtectedRoute>
            }
          />
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;
