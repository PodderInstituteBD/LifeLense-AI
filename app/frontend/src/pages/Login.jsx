import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { motion } from 'framer-motion';
import { LogIn, Sparkles } from 'lucide-react';

export const Login = () => {
  const navigate = useNavigate();

  useEffect(() => {
    // Check if already logged in
    const checkAuth = async () => {
      try {
        const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/auth/me`, {
          credentials: 'include'
        });
        if (response.ok) {
          // Already logged in, redirect to app
          navigate('/app');
        }
      } catch (error) {
        // Not logged in, show login page
      }
    };
    checkAuth();
  }, [navigate]);

  const handleLogin = () => {
    const redirectUrl = window.location.origin + '/app';
    window.location.href = `https://auth.emergentagent.com/?redirect=${encodeURIComponent(redirectUrl)}`;
  };

  return (
    <div className="min-h-screen pt-24 pb-20 px-6">
      <div className="max-w-md mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center justify-center w-20 h-20 bg-brand text-white rounded-none mb-6 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
            <Sparkles className="w-10 h-10" />
          </div>
          <h1 className="font-heading font-bold text-4xl sm:text-5xl tracking-tight mb-4">
            Sign In to LifeLens AI
          </h1>
          <p className="font-body text-lg text-muted-foreground">
            Access your analysis history and saved insights
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="border-2 border-border rounded-none p-8 bg-card"
        >
          <Button
            onClick={handleLogin}
            className="w-full bg-brand text-white hover:bg-brand-dark h-14 px-8 rounded-none border-2 border-transparent hover:border-primary transition-all shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-none hover:translate-x-[2px] hover:translate-y-[2px] text-lg"
            data-testid="google-login-button"
          >
            <LogIn className="w-6 h-6 mr-3" />
            Sign in with Google
          </Button>

          <p className="font-body text-sm text-muted-foreground text-center mt-6">
            Secure authentication powered by Emergent
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="mt-8 text-center"
        >
          <p className="font-body text-sm text-muted-foreground mb-4">
            Don't want to sign in? You can still use the app!
          </p>
          <Button
            onClick={() => navigate('/app')}
            variant="outline"
            className="border-2 border-border hover:bg-secondary"
            data-testid="continue-guest-button"
          >
            Continue as Guest
          </Button>
        </motion.div>
      </div>
    </div>
  );
};
