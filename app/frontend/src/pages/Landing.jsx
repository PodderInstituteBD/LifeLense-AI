import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { motion } from 'framer-motion';
import { Eye, Zap, Shield, Upload } from 'lucide-react';

export const Landing = () => {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="pt-32 pb-20 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            {/* Left: Typography */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
            >
              <h1 className="font-heading font-bold text-5xl sm:text-6xl lg:text-7xl tracking-tight leading-none mb-6">
                Upload an image.
                <br />
                <span className="text-brand">Get instant AI insights.</span>
              </h1>
              <p className="font-body text-lg sm:text-xl text-muted-foreground leading-relaxed mb-8 max-w-xl">
                LifeLens AI analyzes your images with advanced AI technology, delivering professional insights in seconds. Perfect for creators, developers, and innovators.
              </p>
              <div className="flex flex-wrap gap-4">
                <Link to="/app">
                  <Button 
                    className="bg-primary text-primary-foreground hover:bg-primary/90 h-12 px-8 rounded-none border-2 border-transparent hover:border-brand transition-all shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-none hover:translate-x-[2px] hover:translate-y-[2px]"
                    data-testid="hero-cta-primary"
                  >
                    Start Analyzing
                  </Button>
                </Link>
                <Link to="/about">
                  <Button 
                    className="bg-white text-primary border-2 border-primary h-12 px-8 rounded-none hover:bg-secondary transition-all shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-none hover:translate-x-[2px] hover:translate-y-[2px]"
                    data-testid="hero-cta-secondary"
                  >
                    Learn More
                  </Button>
                </Link>
              </div>
            </motion.div>
            
            {/* Right: Hero Image */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="relative"
            >
              <div className="relative aspect-square rounded-none border-2 border-border overflow-hidden shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
                <img 
                  src="https://images.unsplash.com/photo-1769287429003-2a7e8ebee0d9?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NTY2NzV8MHwxfHNlYXJjaHwxfHxhYnN0cmFjdCUyMHRlY2hub2xvZ3klMjBsZW5zJTIwbGlnaHQlMjBiYWNrZ3JvdW5kfGVufDB8fHx8MTc2OTU3NDk2N3ww&ixlib=rb-4.1.0&q=85"
                  alt="AI Vision Technology"
                  className="w-full h-full object-cover"
                />
                <div className="absolute inset-0 bg-brand/10"></div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-6 bg-secondary">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="font-heading font-bold text-4xl sm:text-5xl tracking-tight mb-4">
              Powerful Features
            </h2>
            <p className="font-body text-lg text-muted-foreground max-w-2xl mx-auto">
              Everything you need to unlock insights from your images
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              {
                icon: Eye,
                title: "AI Vision",
                description: "Advanced AI models analyze every detail of your images with precision."
              },
              {
                icon: Zap,
                title: "Instant Results",
                description: "Get comprehensive insights in seconds, not minutes."
              },
              {
                icon: Shield,
                title: "Secure & Private",
                description: "Your images are processed securely and never stored permanently."
              },
              {
                icon: Upload,
                title: "Easy Upload",
                description: "Simple drag-and-drop interface that just works."
              }
            ].map((feature, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: idx * 0.1 }}
                viewport={{ once: true }}
                className="bg-card border-2 border-border rounded-none p-8 hover:shadow-md transition-all group"
                data-testid={`feature-card-${idx}`}
              >
                <feature.icon className="w-10 h-10 mb-4 text-brand group-hover:scale-110 transition-transform" />
                <h3 className="font-heading font-bold text-xl mb-3">{feature.title}</h3>
                <p className="font-body text-sm text-muted-foreground leading-relaxed">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-20 px-6">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="font-heading font-bold text-4xl sm:text-5xl tracking-tight mb-4">
              How It Works
            </h2>
            <p className="font-body text-lg text-muted-foreground">
              Three simple steps to AI-powered insights
            </p>
          </motion.div>

          <div className="space-y-12">
            {[
              {
                step: "01",
                title: "Upload Your Image",
                description: "Drag and drop or click to upload any image in JPEG, PNG, or WEBP format."
              },
              {
                step: "02",
                title: "AI Analyzes",
                description: "Our advanced AI model processes your image and identifies key elements, colors, composition, and mood."
              },
              {
                step: "03",
                title: "Get Insights",
                description: "Receive detailed, professional analysis instantly. Use insights for your projects, research, or creativity."
              }
            ].map((item, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: idx * 0.1 }}
                viewport={{ once: true }}
                className="flex gap-8 items-start"
                data-testid={`how-it-works-${idx}`}
              >
                <div className="flex-shrink-0">
                  <div className="w-20 h-20 bg-brand text-white font-heading font-bold text-2xl flex items-center justify-center border-2 border-primary shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
                    {item.step}
                  </div>
                </div>
                <div className="flex-1">
                  <h3 className="font-heading font-bold text-2xl mb-2">{item.title}</h3>
                  <p className="font-body text-muted-foreground leading-relaxed">
                    {item.description}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t-2 border-border py-12 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center gap-6">
            <div className="font-heading font-bold text-2xl tracking-tight">
              LifeLens AI
            </div>
            <div className="flex gap-8">
              <Link 
                to="/" 
                className="font-body text-sm text-muted-foreground hover:text-brand transition-colors"
                data-testid="footer-home"
              >
                Home
              </Link>
              <Link 
                to="/app" 
                className="font-body text-sm text-muted-foreground hover:text-brand transition-colors"
                data-testid="footer-app"
              >
                Analyze
              </Link>
              <Link 
                to="/about" 
                className="font-body text-sm text-muted-foreground hover:text-brand transition-colors"
                data-testid="footer-about"
              >
                About
              </Link>
            </div>
            <div className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
              © 2025 LifeLens AI
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};
