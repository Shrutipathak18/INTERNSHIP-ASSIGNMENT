
# PDF Q&A System

Video link: 
https://www.loom.com/share/c727801d152f4f41860d354faebd4b8a?sid=2b010402-a975-4d2f-b27b-5b5bebd4e860

A full-stack application that allows users to upload PDF documents and ask questions about their content using natural language processing.

## Features

- PDF document upload and processing
- Natural language question answering
- User authentication and authorization
- Document management
- Conversation history
- Modern, responsive UI

## Tech Stack

### Backend
- FastAPI
- LangChain
- PyMuPDF (for PDF processing)
- SQLite
- JWT Authentication

### Frontend
- React.js
- Axios
- React Router
- Modern CSS

## Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

## Setup Instructions

### Environment Setup

1. Create a `.env` file in the backend directory with the following content:
```env
SECRET_KEY=your-secret-key-here
```

2. Create a `.env` file in the my-app directory with the following content:
```env
VITE_API_URL=http://localhost:8000
```

3. Create required directories in the backend folder:
```bash
mkdir uploads
mkdir faiss_indexes
mkdir temp
```

### Backend Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Start the backend server:
```bash
uvicorn main:app --reload
```

The backend server will run on `http://localhost:8000`

### Frontend Setup

1. Install dependencies:
```bash
cd my-app
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend will run on `http://localhost:5173`

## Project Structure

```
pdf-qa-system/
├── backend/
│   ├── uploads/           # PDF storage
│   ├── faiss_indexes/     # Document indexes
│   ├── temp/             # Temporary files
│   ├── main.py           # FastAPI application
│   ├── requirements.txt  # Python dependencies
│   └── .env             # Backend environment variables
├── my-app/
│   ├── src/             # React source code
│   ├── public/          # Static files
│   ├── package.json     # Node dependencies
│   └── .env            # Frontend environment variables
└── README.md           # Project documentation
```

## Important Notes

1. **Environment Variables**:
   - Never commit `.env` files
   - Use the provided templates to create your own `.env` files
   - Keep your `SECRET_KEY` secure

2. **Storage Directories**:
   - `uploads/`: Stores uploaded PDF files
   - `faiss_indexes/`: Stores document embeddings
   - `temp/`: Stores temporary processing files
   - These directories are gitignored and should be created locally

3. **Database**:
   - SQLite database file is gitignored
   - Will be created automatically on first run

4. **Security**:
   - Keep your JWT secret key secure
   - Don't commit sensitive information
   - Use environment variables for configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Application Architecture

### Backend Architecture

1. **API Layer**
   - FastAPI application with CORS middleware
   - JWT authentication
   - File upload handling
   - Question answering endpoints

2. **Processing Layer**
   - PDF text extraction
   - Document chunking
   - Embedding generation
   - Question answering using LangChain

3. **Data Layer**
   - SQLite database
   - Document metadata storage
   - Conversation history
   - User management

4. **Storage Layer**
   - Local file system for PDFs
   - FAISS indexes for document embeddings
   - Temporary file management

### Frontend Architecture

1. **Components**
   - Authentication (Login/Register)
   - PDF Upload
   - Document List
   - Q&A Interface
   - Navigation

2. **State Management**
   - React Context for authentication
   - Local state for UI components
   - API integration with Axios

3. **Routing**
   - Protected routes
   - Public routes
   - Navigation handling

## API Documentation

### Authentication Endpoints

#### Register User
```http
POST /register
Content-Type: application/json

{
    "email": "user@example.com",
    "username": "username",
    "password": "password"
}
```

#### Login
```http
POST /token
Content-Type: application/x-www-form-urlencoded

username=username&password=password
```

### Document Endpoints

#### Upload PDF
```http
POST /upload
Content-Type: multipart/form-data
Authorization: Bearer <token>

file: <pdf_file>
```

#### Get Documents
```http
GET /documents
Authorization: Bearer <token>
```

#### Delete Document
```http
DELETE /documents/{document_id}
Authorization: Bearer <token>
```

### Q&A Endpoints

#### Ask Question
```http
POST /ask
Content-Type: application/json
Authorization: Bearer <token>

{
    "document_id": "document_id",
    "question": "Your question here",
    "conversation_id": "optional_conversation_id"
}
```

#### Get Conversations
```http
GET /conversations/{document_id}
Authorization: Bearer <token>
```

## Error Handling

The application includes comprehensive error handling for:
- File upload errors
- Authentication failures
- Processing errors
- API communication issues
- Invalid user inputs

## Security Features

- JWT-based authentication
- Password hashing
- Protected routes
- File type validation
- Size limits for uploads

## Performance Optimizations

- Connection pooling
- Caching system
- Memory management
- Batch processing
- Asynchronous operations


Access the Website:
Open your browser
Go to http://localhost:5173

User Registration:
Click on "Register" if you're a new user
Fill in the registration form:
Email address
Username
Password
Click "Register" to create your account

Login:
Enter your username and password
Click "Login" to access the system

Uploading PDF Documents:
After logging in, you'll see the main interface
In the left section, find the "Upload PDF" area
Click "Choose File" to select a PDF document
Click "Upload" to start the upload process

on clicking the uploaded file ask option will get active
Wait for the upload to complete
You'll see the document appear in the documents list
Viewing Documents:
Your uploaded documents will appear in the documents list
Each document shows:
Filename

Upload date
Processing status (pending/processing/completed/failed)(in terminal)
Wait for the document to be processed (status will change to "completed")

Asking Questions:
Select a processed document from the list
In the Q&A section (right side):
Type your question in the input box
Click "Ask" or press Enter

Wait for the answer to appear
The answer will appear in the chat-like interface
You can ask follow-up questions about the same document

Managing Documents:
To delete a document:
Click the delete button (red button) next to the document
Confirm the deletion in the popup

To switch between documents:
Click on a different document in the list
The Q&A section will update accordingly

Viewing Conversation History:
All questions and answers are saved
Previous conversations are visible in the chat interface
You can scroll up to see older conversations

Logging Out:
Click the "Logout" button in the navigation bar
You'll be redirected to the login page

Important Notes:
PDF files must be less than 10MB
Only PDF files are accepted
Document processing may take a few moments
You can ask multiple questions about the same document
The system maintains conversation context for follow-up questions

Troubleshooting:
If a document fails to process, you can try uploading it again
If you get an error message, check:
File size (should be < 10MB)
File format (must be PDF)
Internet connection
Server status


 