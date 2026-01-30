"import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { motion } from 'framer-motion';
import { Upload, Loader2, Image as ImageIcon, CheckCircle2, FileText, Download, Sparkles, ListChecks, MessageCircle, FileSearch } from 'lucide-react';
import axios from 'axios';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export const AppPage = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [analysisId, setAnalysisId] = useState(null);
  const [isDragActive, setIsDragActive] = useState(false);
  const [fileType, setFileType] = useState(null);
  const [pdfMode, setPdfMode] = useState('text');
  
  // NEW: Text extraction and analysis states
  const [extractedText, setExtractedText] = useState(null);
  const [textAnalysisId, setTextAnalysisId] = useState(null);
  const [isExtracting, setIsExtracting] = useState(false);
  const [summary, setSummary] = useState(null);
  const [importantPoints, setImportantPoints] = useState(null);
  const [question, setQuestion] = useState('');
  const [qaResult, setQaResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState('ai'); // 'ai' or 'text'

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = (file) => {
    // Validate file type
    const validImageTypes = ['image/jpeg', 'image/png', 'image/webp'];
    const validPdfType = 'application/pdf';
    
    if (!validImageTypes.includes(file.type) && file.type !== validPdfType) {
      toast.error('Invalid file type. Please upload JPEG, PNG, WEBP images or PDF files.');
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      toast.error('File size too large. Please upload files under 10MB.');
      return;
    }

    setSelectedFile(file);
    setAnalysis(null);
    setAnalysisId(null);
    setExtractedText(null);
    setTextAnalysisId(null);
    setSummary(null);
    setImportantPoints(null);
    setQaResult(null);
    setQuestion('');
    
    // Determine file type
    if (file.type === validPdfType) {
      setFileType('pdf');
      setPreview(null); // No preview for PDFs
    } else {
      setFileType('image');
      // Create preview for images
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
    
    toast.success('File loaded successfully!');
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragActive(false);
    
    const file = e.dataTransfer.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      toast.error('Please select a file first');
      return;
    }

    setIsAnalyzing(true);
    setAnalysis(null);
    setAnalysisId(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      let endpoint = `${API}/analyze-image`;
      let config = {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        withCredentials: true
      };

      if (fileType === 'pdf') {
        endpoint = `${API}/analyze-pdf?mode=${pdfMode}`;
      }

      const response = await axios.post(endpoint, formData, config);

      setAnalysis(response.data.analysis);
      setAnalysisId(response.data.analysis_id);
      toast.success('AI Analysis complete!');
    } catch (error) {
      console.error('Error analyzing file:', error);
      const errorMessage = error.response?.data?.detail || 'Failed to analyze file. Please try again.';
      toast.error(errorMessage);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // NEW: Text extraction function
  const handleExtractText = async () => {
    if (!selectedFile) {
      toast.error('Please select a file first');
      return;
    }

    setIsExtracting(true);
    setExtractedText(null);
    setTextAnalysisId(null);
    setSummary(null);
    setImportantPoints(null);
    setQaResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await axios.post(`${API}/extract-text`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        withCredentials: true
      });

      setExtractedText(response.data.text);
      setTextAnalysisId(response.data.analysis_id);
      toast.success(`Text extracted! ${response.data.word_count} words found.`);
      setActiveTab('text'); // Switch to text tab
    } catch (error) {
      console.error('Error extracting text:', error);
      const errorMessage = error.response?.data?.detail || 'Failed to extract text. Please try again.';
      toast.error(errorMessage);
    } finally {
      setIsExtracting(false);
    }
  };

  // NEW: Summarization function
  const handleSummarize = async () => {
    if (!textAnalysisId) {
      toast.error('Please extract text first');
      return;
    }

    setIsProcessing(true);
    try {
      const response = await axios.post(`${API}/summarize/${textAnalysisId}?num_sentences=5`, {}, {
        withCredentials: true
      });

      setSummary(response.data);
      toast.success('Summary generated!');
    } catch (error) {
      console.error('Error summarizing:', error);
      toast.error('Failed to generate summary');
    } finally {
      setIsProcessing(false);
    }
  };

  // NEW: Important points function
  const handleImportantPoints = async () => {
    if (!textAnalysisId) {
      toast.error('Please extract text first');
      return;
    }

    setIsProcessing(true);
    try {
      const response = await axios.post(`${API}/important-points/${textAnalysisId}?num_points=5`, {}, {
        withCredentials: true
      });

      setImportantPoints(response.data);
      toast.success('Important points extracted!');
    } catch (error) {
      console.error('Error extracting points:', error);
      toast.error('Failed to extract important points');
    } finally {
      setIsProcessing(false);
    }
  };

  // NEW: Question answering function
  const handleAskQuestion = async () => {
    if (!textAnalysisId) {
      toast.error('Please extract text first');
      return;
    }

    if (!question.trim()) {
      toast.error('Please enter a question');
      return;
    }

    setIsProcessing(true);
    try {
      const response = await axios.post(
        `${API}/answer-question/${textAnalysisId}?question=${encodeURIComponent(question)}`,
        {},
        { withCredentials: true }
      );

      setQaResult(response.data);
      toast.success('Answer found!');
    } catch (error) {
      console.error('Error answering question:', error);
      toast.error('Failed to answer question');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = async (format) => {
    if (!analysisId) {
      toast.error('No analysis to download');
      return;
    }

    try {
      const response = await axios.get(
        `${API}/download-${format}/${analysisId}`,
        { 
          responseType: 'blob',
          withCredentials: true 
        }
      );
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `analysis_${analysisId}.${format === 'text' ? 'txt' : 'pdf'}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      
      toast.success(`Downloaded as ${format.toUpperCase()}`);
    } catch (error) {
      console.error('Download error:', error);
      toast.error('Failed to download file');
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setAnalysis(null);
    setAnalysisId(null);
    setIsAnalyzing(false);
    setFileType(null);
    setPdfMode('text');
    setExtractedText(null);
    setTextAnalysisId(null);
    setSummary(null);
    setImportantPoints(null);
    setQaResult(null);
    setQuestion('');
    setActiveTab('ai');
  };

  return (
    <div className=\"min-h-screen pt-24 pb-20 px-6\">
      <div className=\"max-w-7xl mx-auto\">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className=\"text-center mb-12\"
        >
          <h1 className=\"font-heading font-bold text-4xl sm:text-5xl tracking-tight mb-4\">
            Analyze Your Files
          </h1>
          <p className=\"font-body text-lg text-muted-foreground\">
            Upload an image or PDF for AI analysis or text extraction
          </p>
        </motion.div>

        <div className=\"grid grid-cols-1 lg:grid-cols-2 gap-8\">
          {/* Upload Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`relative border-2 border-dashed rounded-none p-12 transition-all ${
                isDragActive
                  ? 'border-brand bg-brand/5'
                  : 'border-border bg-white/70 backdrop-blur-xl'
              } hover:border-brand hover:bg-brand/5`}
              data-testid=\"upload-dropzone\"
            >
              {!selectedFile ? (
                <div className=\"text-center\">
                  <Upload className=\"w-16 h-16 mx-auto mb-4 text-brand\" />
                  <h3 className=\"font-heading font-bold text-xl mb-2\">Upload File</h3>
                  <p className=\"font-body text-sm text-muted-foreground mb-6\">
                    Drag and drop or click to browse
                  </p>
                  <input
                    type=\"file\"
                    accept=\"image/jpeg,image/png,image/webp,application/pdf\"
                    onChange={handleFileSelect}
                    className=\"hidden\"
                    id=\"file-upload\"
                    data-testid=\"file-input\"
                  />
                  <label htmlFor=\"file-upload\">
                    <Button
                      className=\"bg-primary text-primary-foreground hover:bg-primary/90 h-12 px-8 rounded-none border-2 border-transparent hover:border-brand transition-all shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-none hover:translate-x-[2px] hover:translate-y-[2px]\"
                      as=\"span\"
                      data-testid=\"upload-button\"
                    >
                      Select File
                    </Button>
                  </label>
                  <p className=\"font-mono text-xs uppercase tracking-widest text-muted-foreground mt-4\">
                    JPEG, PNG, WEBP, PDF • Max 10MB
                  </p>
                </div>
              ) : (
                <div>
                  {fileType === 'image' && preview ? (
                    <div className=\"relative aspect-video mb-4 border-2 border-border overflow-hidden\">
                      <img
                        src={preview}
                        alt=\"Preview\"
                        className=\"w-full h-full object-contain bg-secondary\"
                        data-testid=\"image-preview\"
                      />
                    </div>
                  ) : (
                    <div className=\"relative aspect-video mb-4 border-2 border-border overflow-hidden bg-secondary flex items-center justify-center\">
                      <div className=\"text-center\">
                        <FileText className=\"w-20 h-20 text-brand mx-auto mb-2\" />
                        <p className=\"font-body font-medium\">{selectedFile.name}</p>
                        <p className=\"font-mono text-xs text-muted-foreground\">PDF Document</p>
                      </div>
                    </div>
                  )}
                  
                  {fileType === 'pdf' && (
                    <div className=\"mb-4 p-4 border-2 border-border bg-secondary\">
                      <p className=\"font-body font-medium text-sm mb-2\">PDF Analysis Mode:</p>
                      <div className=\"flex gap-2\">
                        <Button
                          size=\"sm\"
                          variant={pdfMode === 'text' ? 'default' : 'outline'}
                          onClick={() => setPdfMode('text')}
                          className={pdfMode === 'text' ? 'bg-brand text-white' : ''}
                          data-testid=\"pdf-mode-text\"
                        >
                          Text Extraction
                        </Button>
                        <Button
                          size=\"sm\"
                          variant={pdfMode === 'image' ? 'default' : 'outline'}
                          onClick={() => setPdfMode('image')}
                          className={pdfMode === 'image' ? 'bg-brand text-white' : ''}
                          data-testid=\"pdf-mode-image\"
                        >
                          Visual Analysis
                        </Button>
                      </div>
                    </div>
                  )}
                  
                  <div className=\"space-y-3\">
                    {/* AI Analysis Button */}
                    <Button
                      onClick={handleAnalyze}
                      disabled={isAnalyzing}
                      className=\"w-full bg-brand text-white hover:bg-brand-dark h-12 px-8 rounded-none border-2 border-transparent hover:border-primary transition-all shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-none hover:translate-x-[2px] hover:translate-y-[2px] disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none\"
                      data-testid=\"analyze-button\"
                    >
                      {isAnalyzing ? (
                        <>
                          <Loader2 className=\"w-5 h-5 mr-2 animate-spin\" />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Sparkles className=\"w-5 h-5 mr-2\" />
                          AI Analysis
                        </>
                      )}
                    </Button>

                    {/* NEW: Text Extraction Button */}
                    <Button
                      onClick={handleExtractText}
                      disabled={isExtracting}
                      className=\"w-full bg-green-600 text-white hover:bg-green-700 h-12 px-8 rounded-none border-2 border-transparent hover:border-green-800 transition-all shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-none hover:translate-x-[2px] hover:translate-y-[2px] disabled:opacity-50 disabled:cursor-not-allowed\"
                      data-testid=\"extract-text-button\"
                    >
                      {isExtracting ? (
                        <>
                          <Loader2 className=\"w-5 h-5 mr-2 animate-spin\" />
                          Extracting...
                        </>
                      ) : (
                        <>
                          <FileSearch className=\"w-5 h-5 mr-2\" />
                          Extract Text
                        </>
                      )}
                    </Button>

                    <Button
                      onClick={handleReset}
                      disabled={isAnalyzing || isExtracting}
                      className=\"w-full bg-white text-primary border-2 border-primary h-12 px-6 rounded-none hover:bg-secondary transition-all shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-none hover:translate-x-[2px] hover:translate-y-[2px] disabled:opacity-50\"
                      data-testid=\"reset-button\"
                    >
                      Reset
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </motion.div>

          {/* Results Section */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <div className=\"border-2 border-border rounded-none bg-card min-h-[400px]\">
              {/* Tab Navigation */}
              <div className=\"border-b-2 border-border flex\">
                <button
                  onClick={() => setActiveTab('ai')}
                  className={`flex-1 px-6 py-4 font-heading font-bold text-sm transition-colors ${
                    activeTab === 'ai'
                      ? 'bg-brand text-white border-b-4 border-brand'
                      : 'bg-white text-muted-foreground hover:bg-secondary'
                  }`}
                >
                  <Sparkles className=\"w-4 h-4 inline mr-2\" />
                  AI Analysis
                </button>
                <button
                  onClick={() => setActiveTab('text')}
                  className={`flex-1 px-6 py-4 font-heading font-bold text-sm transition-colors ${
                    activeTab === 'text'
                      ? 'bg-brand text-white border-b-4 border-brand'
                      : 'bg-white text-muted-foreground hover:bg-secondary'
                  }`}
                >
                  <FileText className=\"w-4 h-4 inline mr-2\" />
                  Text Analysis
                </button>
              </div>

              {/* Tab Content */}
              <div className=\"p-8\">
                {activeTab === 'ai' && (
                  <>
                    {!analysis && !isAnalyzing ? (
                      <div className=\"flex flex-col items-center justify-center h-full text-center py-12\">
                        <ImageIcon className=\"w-16 h-16 text-muted-foreground/50 mb-4\" />
                        <h3 className=\"font-heading font-bold text-xl mb-2\">No AI Analysis Yet</h3>
                        <p className=\"font-body text-sm text-muted-foreground\">
                          Upload a file and click AI Analysis to see insights
                        </p>
                      </div>
                    ) : isAnalyzing ? (
                      <div className=\"flex flex-col items-center justify-center h-full py-12\">
                        <Loader2 className=\"w-16 h-16 text-brand animate-spin mb-4\" />
                        <h3 className=\"font-heading font-bold text-xl mb-2\">Analyzing File...</h3>
                        <p className=\"font-body text-sm text-muted-foreground\">
                          AI is processing your {fileType}
                        </p>
                      </div>
                    ) : (
                      <div data-testid=\"analysis-result\">
                        <div className=\"flex items-center gap-2 mb-6\">
                          <CheckCircle2 className=\"w-6 h-6 text-brand\" />
                          <h3 className=\"font-heading font-bold text-2xl\">Analysis Complete</h3>
                        </div>
                        <div className=\"prose prose-sm max-w-none mb-6\">
                          <p className=\"font-body text-foreground leading-relaxed whitespace-pre-wrap\">
                            {analysis}
                          </p>
                        </div>
                        
                        {/* Download Buttons */}
                        <div className=\"border-t-2 border-border pt-6\">
                          <p className=\"font-body font-medium text-sm mb-3\">Download Analysis:</p>
                          <div className=\"flex gap-3\">
                            <Button
                              onClick={() => handleDownload('text')}
                              variant=\"outline\"
                              className=\"flex-1 border-2 border-border hover:bg-secondary\"
                              data-testid=\"download-txt-button\"
                            >
                              <Download className=\"w-4 h-4 mr-2\" />
                              Download as TXT
                            </Button>
                            <Button
                              onClick={() => handleDownload('pdf')}
                              variant=\"outline\"
                              className=\"flex-1 border-2 border-border hover:bg-secondary\"
                              data-testid=\"download-pdf-button\"
                            >
                              <Download className=\"w-4 h-4 mr-2\" />
                              Download as PDF
                            </Button>
                          </div>
                        </div>
                      </div>
                    )}
                  </>
                )}

                {activeTab === 'text' && (
                  <>
                    {!extractedText && !isExtracting ? (
                      <div className=\"flex flex-col items-center justify-center h-full text-center py-12\">
                        <FileText className=\"w-16 h-16 text-muted-foreground/50 mb-4\" />
                        <h3 className=\"font-heading font-bold text-xl mb-2\">No Text Extracted Yet</h3>
                        <p className=\"font-body text-sm text-muted-foreground\">
                          Upload a file and click Extract Text to begin
                        </p>
                      </div>
                    ) : isExtracting ? (
                      <div className=\"flex flex-col items-center justify-center h-full py-12\">
                        <Loader2 className=\"w-16 h-16 text-green-600 animate-spin mb-4\" />
                        <h3 className=\"font-heading font-bold text-xl mb-2\">Extracting Text...</h3>
                        <p className=\"font-body text-sm text-muted-foreground\">
                          Processing your {fileType}
                        </p>
                      </div>
                    ) : (
                      <div className=\"space-y-6\">
                        {/* Extracted Text */}
                        <div>
                          <h3 className=\"font-heading font-bold text-lg mb-3 flex items-center gap-2\">
                            <FileText className=\"w-5 h-5 text-brand\" />
                            Extracted Text
                          </h3>
                          <div className=\"border-2 border-border rounded-none p-4 bg-secondary max-h-48 overflow-y-auto\">
                            <p className=\"font-mono text-sm whitespace-pre-wrap\">
                              {extractedText}
                            </p>
                          </div>
                        </div>

                        {/* Action Buttons */}
                        <div className=\"grid grid-cols-2 gap-3\">
                          <Button
                            onClick={handleSummarize}
                            disabled={isProcessing}
                            variant=\"outline\"
                            className=\"border-2 border-border hover:bg-secondary\"
                          >
                            <FileText className=\"w-4 h-4 mr-2\" />
                            Summarize
                          </Button>
                          <Button
                            onClick={handleImportantPoints}
                            disabled={isProcessing}
                            variant=\"outline\"
                            className=\"border-2 border-border hover:bg-secondary\"
                          >
                            <ListChecks className=\"w-4 h-4 mr-2\" />
                            Important Points
                          </Button>
                        </div>

                        {/* Summary */}
                        {summary && (
                          <div>
                            <h3 className=\"font-heading font-bold text-lg mb-3 flex items-center gap-2\">
                              <FileText className=\"w-5 h-5 text-green-600\" />
                              Summary ({summary.summary_sentences} of {summary.original_sentences} sentences)
                            </h3>
                            <div className=\"border-2 border-green-600 rounded-none p-4 bg-green-50\">
                              <p className=\"font-body text-sm leading-relaxed\">
                                {summary.summary}
                              </p>
                            </div>
                          </div>
                        )}

                        {/* Important Points */}
                        {importantPoints && (
                          <div>
                            <h3 className=\"font-heading font-bold text-lg mb-3 flex items-center gap-2\">
                              <ListChecks className=\"w-5 h-5 text-blue-600\" />
                              Important Points ({importantPoints.total_points})
                            </h3>
                            <div className=\"border-2 border-blue-600 rounded-none p-4 bg-blue-50 space-y-2\">
                              {importantPoints.points.map((point, idx) => (
                                <div key={idx} className=\"flex gap-2\">
                                  <span className=\"font-heading font-bold text-blue-600\">•</span>
                                  <p className=\"font-body text-sm leading-relaxed flex-1\">
                                    {point}
                                  </p>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Q&A Section */}
                        <div>
                          <h3 className=\"font-heading font-bold text-lg mb-3 flex items-center gap-2\">
                            <MessageCircle className=\"w-5 h-5 text-purple-600\" />
                            Ask a Question
                          </h3>
                          <div className=\"flex gap-2\">
                            <Input
                              type=\"text\"
                              placeholder=\"e.g., What is the main topic?\"
                              value={question}
                              onChange={(e) => setQuestion(e.target.value)}
                              onKeyPress={(e) => e.key === 'Enter' && handleAskQuestion()}
                              className=\"flex-1 border-2 border-border\"
                            />
                            <Button
                              onClick={handleAskQuestion}
                              disabled={isProcessing || !question.trim()}
                              className=\"bg-purple-600 text-white hover:bg-purple-700\"
                            >
                              <MessageCircle className=\"w-4 h-4 mr-2\" />
                              Ask
                            </Button>
                          </div>
                        </div>

                        {/* Q&A Result */}
                        {qaResult && (
                          <div>
                            <h3 className=\"font-heading font-bold text-lg mb-3 flex items-center gap-2\">
                              <CheckCircle2 className=\"w-5 h-5 text-purple-600\" />
                              Answer (Confidence: {(qaResult.confidence * 100).toFixed(1)}%)
                            </h3>
                            <div className=\"border-2 border-purple-600 rounded-none p-4 bg-purple-50\">
                              <p className=\"font-body text-sm font-medium text-purple-900 mb-2\">
                                Q: {qaResult.question}
                              </p>
                              <p className=\"font-body text-sm leading-relaxed\">
                                A: {qaResult.answer}
                              </p>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};
"
