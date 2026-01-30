"# LifeLens AI - Complete Image & Text Analysis Platform

**Hackathon-Ready AI Analysis Tool** with advanced text extraction and processing capabilities.

## 🚀 Overview

LifeLens AI is a comprehensive analysis platform that combines AI-powered image analysis with advanced text extraction and NLP capabilities. Upload images or PDFs and get instant insights using both AI vision models and local text processing.

## ✨ Features

### 1. **AI-Powered Analysis** (Original Features)
- **Image Analysis**: Deep visual analysis using OpenAI GPT-5.2 Vision API
- **PDF Analysis**: Two modes:
  - Text extraction and AI summarization
  - Visual analysis of PDF pages as images
- **User Authentication**: Secure Google OAuth integration via Emergent Auth
- **Analysis History**: Save and retrieve past analyses (authenticated users)
- **Download Reports**: Export analyses as TXT or PDF files

### 2. **NEW: Text Extraction & Processing** 
All features work **WITHOUT paid APIs** - completely open-source!

#### 🔍 **Text Extraction**
- **OCR for Images**: Extracts text from images using Tesseract OCR
- **PDF Text Extraction**: Pulls text content from PDF documents
- **Clean & Normalize**: Automatic text cleaning and formatting

#### 📝 **Text Summarization**
- **TF-IDF Based**: Uses term frequency-inverse document frequency for intelligent sentence scoring
- **Adjustable Length**: Generate summaries from 1-10 sentences
- **Maintains Context**: Preserves original sentence order for coherence

#### 📌 **Important Points Extraction**
- **Keyword Frequency**: Identifies key topics and themes
- **Sentence Scoring**: Ranks sentences by importance
- **Fact Detection**: Prioritizes sentences with numbers and statistics
- **Smart Selection**: Filters out noise, keeps valuable information

#### 💬 **Question Answering (Q&A)**
- **Semantic Search**: Finds relevant answers using cosine similarity
- **Confidence Scoring**: Shows how well the answer matches the question
- **Context Aware**: Returns surrounding sentences for better understanding
- **Real-time**: Instant answers from extracted text

## 🛠️ Tech Stack

### Frontend
- **React 19** - Modern UI framework
- **Tailwind CSS** - Utility-first styling
- **Framer Motion** - Smooth animations
- **Axios** - HTTP client
- **React Router** - Navigation
- **Sonner** - Toast notifications
- **Lucide React** - Beautiful icons

### Backend
- **FastAPI** (Python) - High-performance API framework
- **Motor** - Async MongoDB driver
- **Emergent Integrations** - LLM integration (OpenAI GPT-5.2)

### NEW: Text Processing Stack (All Free/Open-Source!)
- **Tesseract OCR** - Image text extraction
- **PyPDF** - PDF text extraction
- **NLTK** - Natural language processing
- **scikit-learn** - Machine learning for TF-IDF and similarity
- **NumPy** - Numerical computing

### Database & Infrastructure
- **MongoDB** - Document storage
- **Supervisor** - Process management

## 📦 Installation & Setup

### Prerequisites
```bash
# System packages
apt-get install tesseract-ocr poppler-utils

# Python packages (already in requirements.txt)
pip install pytesseract pillow nltk scikit-learn
```

### Backend Setup
```bash
cd /app/backend
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd /app/frontend
yarn install
```

### Environment Variables
Create `/app/backend/.env`:
```env
MONGO_URL=\"mongodb://localhost:27017\"
DB_NAME=\"test_database\"
CORS_ORIGINS=\"*\"
EMERGENT_LLM_KEY=your_key_here
```

### Run Services
```bash
sudo supervisorctl restart all
```

## 🎯 Usage Guide

### 1. Upload a File
- Drag & drop or click to select
- Supports: JPEG, PNG, WEBP images and PDF files
- Max file size: 10MB

### 2. Choose Analysis Type

#### **AI Analysis Tab**
1. Click \"AI Analysis\" button
2. Get professional AI-generated insights
3. Download results as TXT or PDF

#### **Text Analysis Tab** (NEW!)
1. Click \"Extract Text\" button
2. View extracted text content
3. Use powerful text processing features:

**Summarize**
- Click \"Summarize\" to generate a concise summary
- Reduces long documents to key sentences
- Perfect for quick overviews

**Important Points**
- Click \"Important Points\" to extract key information
- Identifies the most significant sentences
- Great for bullet-point summaries

**Ask Questions**
- Type your question in the input field
- Click \"Ask\" to get instant answers
- System finds and returns relevant content
- Shows confidence score

### 3. View Results
- All results display in structured, readable format
- Switch between AI and Text analysis tabs
- Export or save for later reference

## 🔥 Key Differentiators

### Why LifeLens AI Stands Out:

1. **Dual Mode Analysis**: Both AI-powered AND code-based text processing
2. **No API Lock-in**: Text features work completely offline
3. **Production Ready**: Real authentication, database, full error handling
4. **Hackathon Perfect**: 
   - Works without constant API calls
   - No usage limits on text processing
   - Live demo-ready
   - Judges can test immediately

## 📊 API Endpoints

### Original Endpoints
- `POST /api/analyze-image` - AI image analysis
- `POST /api/analyze-pdf?mode={text|image}` - AI PDF analysis
- `GET /api/history` - Get user's analysis history
- `GET /api/download-{text|pdf}/{analysis_id}` - Download reports

### NEW Text Processing Endpoints
- `POST /api/extract-text` - Extract text from image/PDF
- `POST /api/summarize/{analysis_id}?num_sentences=5` - Summarize text
- `POST /api/important-points/{analysis_id}?num_points=5` - Extract key points
- `POST /api/answer-question/{analysis_id}?question={text}` - Answer questions

### Authentication Endpoints
- `POST /api/auth/session` - Create user session
- `GET /api/auth/me` - Get current user
- `POST /api/auth/logout` - Logout user

## 🧪 Testing

### Test Text Extraction
```bash
# Create a test image with text
echo \"Hello World
This is a test\" | convert -size 800x600 -background white -fill black -pointsize 48 label:@- test.png

# Upload and extract
curl -X POST http://localhost:8001/api/extract-text \
  -F \"file=@test.png\" \
  -H \"Content-Type: multipart/form-data\"
```

### Test Summarization
```python
# After extracting text, use the analysis_id
import requests

analysis_id = \"your_analysis_id_here\"
response = requests.post(
    f\"http://localhost:8001/api/summarize/{analysis_id}?num_sentences=3\"
)
print(response.json())
```

## 🎨 UI Features

### Modern Design Elements
- **Brutalist Design**: Bold borders, strong shadows, clear typography
- **Smooth Animations**: Framer Motion powered transitions
- **Responsive Layout**: Works on all screen sizes
- **Tab Interface**: Easy switching between AI and Text analysis
- **Loading States**: Clear feedback during processing
- **Error Handling**: User-friendly error messages

### Color Coding
- **Purple** - Q&A results
- **Green** - Text extraction & summaries
- **Blue** - Important points
- **Brand Purple** - AI analysis

## 🏆 Hackathon Highlights

### Judge Testing Checklist
✅ Upload an image with text → Extract text → Summarize  
✅ Upload a PDF → Get AI analysis  
✅ Ask questions about extracted content  
✅ Extract important points  
✅ Sign in and view history  
✅ Download results  

### Demo Flow
1. Show landing page and features
2. Upload sample image → extract text
3. Generate summary and important points
4. Ask question and get answer
5. Show AI analysis tab for comparison
6. Demonstrate authentication and history

## 🔒 Security & Privacy

- Secure authentication via Emergent OAuth
- Session-based user management
- File validation and size limits
- No permanent file storage (privacy-first)
- CORS protection
- Input sanitization

## 📈 Future Enhancements

- Multi-language support for OCR
- Batch processing for multiple files
- Advanced NLP features (sentiment analysis, entity recognition)
- Custom trained models for domain-specific analysis
- Collaborative features (share analyses)
- API key management for power users

## 🤝 Contributing

This is a hackathon project built for demonstrating AI/ML capabilities in a production-ready application.

## 📄 License

Built with ❤️ for hackathons and education.

## 🎓 Technologies Learned

- FastAPI async programming
- React 19 features
- Tesseract OCR integration
- NLP with NLTK and scikit-learn
- TF-IDF vectorization
- Cosine similarity for Q&A
- MongoDB with Motor
- OAuth authentication flows
- Modern UI/UX patterns

---

## 💡 Quick Start

```bash
# 1. Start all services
sudo supervisorctl restart all

# 2. Check status
sudo supervisorctl status

# 3. Open browser
# Visit: https://your-app-url.com

# 4. Upload a file and start analyzing!
```

## 🐛 Troubleshooting

### Backend not starting?
```bash
tail -f /var/log/supervisor/backend.err.log
```

### Frontend compilation errors?
```bash
cd /app/frontend
yarn install
sudo supervisorctl restart frontend
```

### OCR not working?
```bash
tesseract --version  # Should show version 5.x
```

---

**Built for:**
- Hackathons
- Educational projects  
- AI/ML demonstrations
- Full-stack portfolio pieces

**Ready to impress judges with real-world AI + NLP integration!** 🚀
"
