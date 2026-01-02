# Intelligent Document Processing System

An AI-powered system for extracting financial data from investment fund prospectuses using Django REST Framework, React, and Google Gemini 2.0.

## Project Structure

```
Project_OCR/
├── backend/
│   ├── api/                    # Django app for document processing
│   │   ├── models.py          # Document and ExtractedFundData models
│   │   ├── serializers.py     # DRF serializers
│   │   ├── views.py           # API endpoints
│   │   ├── services.py        # Gemini AI integration and processing
│   │   └── urls.py            # API routes
│   ├── config/                 # Django project settings
│   ├── media/                  # Uploaded PDF files
│   ├── manage.py
│   ├── requirements.txt
│   └── .env                    # Environment variables
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── FileUpload.jsx      # File upload component
│   │   │   ├── Dashboard.jsx       # Results dashboard
│   │   │   └── *.css               # Component styles
│   │   ├── services/
│   │   │   └── api.js              # API service layer
│   │   ├── App.jsx
│   │   └── main.jsx
│   └── package.json
└── venv/                       # Python virtual environment
```

## Features

### Backend
- **Document Management**: Upload, store, and manage PDF documents
- **AI Processing**: Extract financial data using Google Gemini 2.0 Flash
- **Asynchronous Processing**: Non-blocking document processing with threading
- **RESTful API**: Complete CRUD operations for documents
- **Data Models**:
  - `Document`: Stores metadata and processing status
  - `ExtractedFundData`: Normalized extracted financial information

### Frontend
- **File Upload**: Drag-and-drop PDF upload interface
- **Dashboard**: View all documents and their processing status
- **Data Visualization**: Display extracted fund information in tables
- **PDF Viewer**: Side-by-side document preview
- **Real-time Updates**: Auto-refresh for processing documents

### Extracted Data Fields
- Fund Name
- Fund Code
- Management Fee
- Custodian Bank
- Portfolio Holdings (array of assets)

## Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 16+
- PostgreSQL database (Neon)
- Google Gemini API key

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file in the backend directory:
   ```env
   DATABASE_URL=postgresql://user:password@host/database?sslmode=require
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

5. **Run migrations:**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

6. **Create superuser (optional):**
   ```bash
   python manage.py createsuperuser
   ```

7. **Create required directories:**
   ```bash
   mkdir -p media logs
   ```

8. **Run development server:**
   ```bash
   python manage.py runserver
   ```

   Backend will be available at: `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Configure API URL (optional):**
   Create a `.env` file in the frontend directory:
   ```env
   VITE_API_URL=http://localhost:8000/api
   ```

4. **Run development server:**
   ```bash
   npm run dev
   ```

   Frontend will be available at: `http://localhost:5173`

## API Endpoints

### Documents
- `GET /api/documents/` - List all documents
- `POST /api/documents/` - Upload a new document
- `GET /api/documents/{id}/` - Get document details
- `DELETE /api/documents/{id}/` - Delete a document
- `POST /api/documents/{id}/reprocess/` - Reprocess a document
- `GET /api/documents/{id}/download/` - Download PDF file
- `GET /api/documents/stats/` - Get processing statistics

### Utility
- `GET /api/health/` - Health check endpoint

## Usage

1. **Upload a PDF**: 
   - Drag and drop a PDF file or click to browse
   - Click "Upload & Process"
   - Document status will change from "pending" to "processing" to "completed"

2. **View Results**:
   - Click on a document in the list
   - View extracted financial data
   - See portfolio holdings in a table
   - Preview the original PDF

3. **Manage Documents**:
   - Reprocess failed documents
   - Delete documents
   - Download original PDFs

## Technologies Used

### Backend
- Django 6.0
- Django REST Framework
- PostgreSQL (Neon)
- Google Generative AI (Gemini 2.0 Flash)
- PyPDF2 for PDF processing
- Python-dotenv for environment variables

### Frontend
- React 18
- Vite
- CSS3 with modern layouts
- Fetch API for HTTP requests

## Evaluation Support

The system stores extracted data in JSON format, making it easy to:
- Compare with ground truth data
- Calculate Levenshtein Distance for text fields
- Compute TEDS scores for table structures
- Export data for further analysis

## Future Enhancements

- [ ] Celery integration for better async processing
- [ ] Redis caching for improved performance
- [ ] Batch processing multiple documents
- [ ] Advanced analytics and reporting
- [ ] Export to various formats (CSV, Excel)
- [ ] User authentication and authorization
- [ ] Document versioning
- [ ] OCR support for scanned documents

## Troubleshooting

### Backend Issues
- **Database connection**: Verify DATABASE_URL in .env
- **Gemini API errors**: Check GEMINI_API_KEY is valid
- **Module errors**: Ensure all dependencies are installed

### Frontend Issues
- **API connection**: Check VITE_API_URL points to correct backend
- **CORS errors**: Verify CORS settings in Django settings.py

## License

MIT License

## Competition Notes

This system is designed for a tech competition focusing on Intelligent Document Processing. Key features for evaluation:
- Accurate extraction of financial data
- Clean, maintainable code structure
- Modern tech stack
- Scalable architecture
- User-friendly interface
