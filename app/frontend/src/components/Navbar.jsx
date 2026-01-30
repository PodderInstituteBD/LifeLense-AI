"import { useState, useEffect } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { LogOut, User, History as HistoryIcon } from 'lucide-react';
import { toast } from 'sonner';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export const Navbar = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  
  const isActive = (path) => location.pathname === path;
  
  useEffect(() => {
    checkAuth();
  }, [location.pathname]);
  
  const checkAuth = async () => {
    try {
      const response = await axios.get(`${API}/auth/me`, {
        withCredentials: true
      });
      setUser(response.data);
    } catch (error) {
      setUser(null);
    } finally {
      setLoading(false);
    }
  };
  
  const handleLogout = async () => {
    try {
      await axios.post(`${API}/auth/logout`, {}, {
        withCredentials: true
      });
      setUser(null);
      toast.success('Logged out successfully');
      navigate('/');
    } catch (error) {
      console.error('Logout error:', error);
      toast.error('Failed to logout');
    }
  };
  
  return (
    <nav className=\"fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b-2 border-border\">
      <div className=\"max-w-7xl mx-auto px-6 py-4\">
        <div className=\"flex items-center justify-between\">
          <Link 
            to=\"/\" 
            className=\"font-heading font-bold text-2xl tracking-tight hover:text-brand transition-colors\"
            data-testid=\"navbar-logo\"
          >
            LifeLens AI
          </Link>
          
          <div className=\"flex items-center gap-6\">
            <Link 
              to=\"/\" 
              className={`font-body text-sm font-medium transition-colors hover:text-brand ${
                isActive('/') ? 'text-brand' : 'text-foreground'
              }`}
              data-testid=\"nav-home\"
            >
              Home
            </Link>
            <Link 
              to=\"/app\" 
              className={`font-body text-sm font-medium transition-colors hover:text-brand ${
                isActive('/app') ? 'text-brand' : 'text-foreground'
              }`}
              data-testid=\"nav-app\"
            >
              Analyze
            </Link>
            {user && (
              <Link 
                to=\"/history\" 
                className={`font-body text-sm font-medium transition-colors hover:text-brand flex items-center gap-1 ${
                  isActive('/history') ? 'text-brand' : 'text-foreground'
                }`}
                data-testid=\"nav-history\"
              >
                <HistoryIcon className=\"w-4 h-4\" />
                History
              </Link>
            )}
            <Link 
              to=\"/about\" 
              className={`font-body text-sm font-medium transition-colors hover:text-brand ${
                isActive('/about') ? 'text-brand' : 'text-foreground'
              }`}
              data-testid=\"nav-about\"
            >
              About
            </Link>
            
            {!loading && (
              <>
                {user ? (
                  <div className=\"flex items-center gap-3\">
                    <div className=\"flex items-center gap-2 px-3 py-1 bg-secondary rounded-none border-2 border-border\">
                      <User className=\"w-4 h-4 text-brand\" />
                      <span className=\"font-body text-sm font-medium\">{user.name}</span>
                    </div>
                    <Button
                      onClick={handleLogout}
                      size=\"sm\"
                      variant=\"outline\"
                      className=\"border-2 border-border hover:bg-secondary\"
                      data-testid=\"logout-button\"
                    >
                      <LogOut className=\"w-4 h-4 mr-1\" />
                      Logout
                    </Button>
                  </div>
                ) : (
                  <Link to=\"/login\">
                    <Button 
                      className=\"bg-primary text-primary-foreground hover:bg-primary/90 h-10 px-6 rounded-none border-2 border-transparent hover:border-brand transition-all shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-none hover:translate-x-[2px] hover:translate-y-[2px]\"
                      data-testid=\"nav-login-button\"
                    >
                      Sign In
                    </Button>
                  </Link>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};
"
