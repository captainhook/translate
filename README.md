# Translate
A FastAPI-based web application that translates PDF documents between multiple languages using Meta's M2M100 machine translation model. The application supports bidirectional translation between English, French, Arabic, Spanish, and Chinese 

### Key Features
- Uploads and processes PDF files up to 10MB
- Extracts text from PDF documents while handling multi-page documents
- Translates text using Meta's M2M100 1.2B model
- Supports proper rendering of right-to-left scripts (like Arabic)
- Generates a new PDF with the translated content
- Provides both web interface and command-line usage
- Includes GPU acceleration support when available

### Technical Details
- Built with FastAPI, PyTorch, and Hugging Face Transformers
- Handles memory management for large documents
- Implements error handling for PDF extraction, translation, and PDF creation
- Uses Jinja2 templates for the web interface
- Supports both synchronous CLI operation and asynchronous web API endpoints
- The project can be run as either a web service or used directly from the command line for batch processing of PDF documents.