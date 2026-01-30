import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Clock, FileText, Image as ImageIcon, Download, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export const History = () => {
  const navigate = useNavigate();
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [user, setUser] = useState(null);

  useEffect(() => {
    const checkAuthAndFetchHistory = async () => {
      try {
        // Check authentication
        const authResponse = await axios.get(`${API}/auth/me`, {
          withCredentials: true
        });
        
        if (!authResponse.data) {
          navigate('/login');
          return;
        }
        
        setUser(authResponse.data);
        
        // Fetch history
        const historyResponse = await axios.get(`${API}/history`, {
          withCredentials: true
        });
        
        setHistory(historyResponse.data);
      } catch (error) {
        console.error('Error:', error);
        if (error.response?.status === 401) {
          toast.error('Please sign in to view history');
          navigate('/login');
        } else {
          toast.error('Failed to load history');
        }
      } finally {
        setLoading(false);
      }
    };

    checkAuthAndFetchHistory();
  }, [navigate]);

  const handleDownload = async (analysisId, format) => {
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

  if (loading) {
    return (
      <div className="min-h-screen pt-24 pb-20 px-6 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-16 h-16 animate-spin text-brand mx-auto mb-4" />
          <p className="font-body text-muted-foreground">Loading history...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen pt-24 pb-20 px-6">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-12"
        >
          <div className="flex items-center justify-between mb-4">
            <h1 className="font-heading font-bold text-4xl sm:text-5xl tracking-tight">
              Analysis History
            </h1>
            {user && (
              <div className="text-right">
                <p className="font-body text-sm text-muted-foreground">Signed in as</p>
                <p className="font-body font-medium">{user.name}</p>
              </div>
            )}
          </div>
          <p className="font-body text-lg text-muted-foreground">
            Your saved image and PDF analyses
          </p>
        </motion.div>

        {history.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="border-2 border-border rounded-none p-12 bg-card text-center"
          >
            <Clock className="w-16 h-16 text-muted-foreground/50 mx-auto mb-4" />
            <h3 className="font-heading font-bold text-2xl mb-2">No History Yet</h3>
            <p className="font-body text-muted-foreground mb-6">
              Start analyzing images and PDFs to build your history
            </p>
            <Button
              onClick={() => navigate('/app')}
              className="bg-brand text-white hover:bg-brand-dark"
            >
              Start Analyzing
            </Button>
          </motion.div>
        ) : (
          <div className="grid grid-cols-1 gap-6">
            {history.map((item, idx) => (
              <motion.div
                key={item.analysis_id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: idx * 0.1 }}
                className="border-2 border-border rounded-none p-6 bg-card hover:shadow-md transition-shadow"
                data-testid={`history-item-${idx}`}
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-start gap-4 flex-1">
                    <div className="flex-shrink-0 w-12 h-12 bg-brand/10 flex items-center justify-center">
                      {item.file_type === 'pdf' ? (
                        <FileText className="w-6 h-6 text-brand" />
                      ) : (
                        <ImageIcon className="w-6 h-6 text-brand" />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className="font-heading font-bold text-xl mb-1 truncate">
                        {item.filename}
                      </h3>
                      <p className="font-mono text-xs text-muted-foreground mb-2">
                        {new Date(item.timestamp).toLocaleString()} • {item.file_type.toUpperCase()}
                      </p>
                      <p className="font-body text-sm text-foreground line-clamp-3">
                        {item.analysis}
                      </p>
                    </div>
                  </div>
                  <div className="flex gap-2 ml-4">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleDownload(item.analysis_id, 'text')}
                      className="border-2 border-border hover:bg-secondary"
                      data-testid={`download-txt-${idx}`}
                    >
                      <Download className="w-4 h-4 mr-1" />
                      TXT
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleDownload(item.analysis_id, 'pdf')}
                      className="border-2 border-border hover:bg-secondary"
                      data-testid={`download-pdf-${idx}`}
                    >
                      <Download className="w-4 h-4 mr-1" />
                      PDF
                    </Button>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
