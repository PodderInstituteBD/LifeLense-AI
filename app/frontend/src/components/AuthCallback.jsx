"import { useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'sonner';
import { Loader2 } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export const AuthCallback = () => {
  const navigate = useNavigate();
  const hasProcessed = useRef(false);

  useEffect(() => {
    // Prevent double processing in React StrictMode
    if (hasProcessed.current) return;
    hasProcessed.current = true;

    const processAuth = async () => {
      try {
        // Extract session_id from URL fragment
        const hash = window.location.hash;
        const params = new URLSearchParams(hash.substring(1));
        const sessionId = params.get('session_id');

        if (!sessionId) {
          toast.error('Invalid authentication response');
          navigate('/');
          return;
        }

        // Exchange session_id for session_token
        const response = await fetch(`${API}/auth/session`, {
          method: 'POST',
          headers: {
            'X-Session-ID': sessionId,
            'Content-Type': 'application/json'
          },
          credentials: 'include'
        });

        if (!response.ok) {
          throw new Error('Authentication failed');
        }

        const userData = await response.json();
        
        // Clear the hash from URL
        window.history.replaceState(null, '', window.location.pathname);
        
        toast.success(`Welcome, ${userData.name}!`);
        
        // Navigate to app with user data
        navigate('/app', { state: { user: userData } });
        
      } catch (error) {
        console.error('Auth error:', error);
        toast.error('Authentication failed. Please try again.');
        navigate('/');
      }
    };

    processAuth();
  }, [navigate]);

  return (
    <div className=\"min-h-screen flex items-center justify-center\">
      <div className=\"text-center\">
        <Loader2 className=\"w-16 h-16 animate-spin text-brand mx-auto mb-4\" />
        <h2 className=\"font-heading font-bold text-2xl mb-2\">Authenticating...</h2>
        <p className=\"font-body text-muted-foreground\">Please wait while we log you in</p>
      </div>
    </div>
  );
};
"
