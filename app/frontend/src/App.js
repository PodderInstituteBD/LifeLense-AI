"import \"@/App.css\";
import { BrowserRouter, Routes, Route, useLocation } from \"react-router-dom\";
import { Toaster } from \"sonner\";
import { Navbar } from \"@/components/Navbar\";
import { AuthCallback } from \"@/components/AuthCallback\";
import { Landing } from \"@/pages/Landing\";
import { AppPage } from \"@/pages/AppPage\";
import { About } from \"@/pages/About\";
import { Login } from \"@/pages/Login\";
import { History } from \"@/pages/History\";

// Router wrapper to handle auth callback detection
function AppRouter() {
  const location = useLocation();
  
  // Check URL fragment for session_id synchronously during render
  // This prevents race conditions by processing session_id FIRST
  if (location.hash?.includes('session_id=')) {
    return <AuthCallback />;
  }
  
  return (
    <>
      <Navbar />
      <Routes>
        <Route path=\"/\" element={<Landing />} />
        <Route path=\"/app\" element={<AppPage />} />
        <Route path=\"/about\" element={<About />} />
        <Route path=\"/login\" element={<Login />} />
        <Route path=\"/history\" element={<History />} />
      </Routes>
    </>
  );
}

function App() {
  return (
    <div className=\"App\">
      <BrowserRouter>
        <AppRouter />
      </BrowserRouter>
      <Toaster position=\"top-right\" richColors />
    </div>
  );
}

export default App;
"
