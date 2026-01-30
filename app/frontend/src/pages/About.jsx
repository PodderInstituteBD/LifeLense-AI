import { motion } from 'framer-motion';
import { Code, Database, Cpu, Sparkles } from 'lucide-react';

export const About = () => {
  return (
    <div className="min-h-screen pt-24 pb-20 px-6">
      <div className="max-w-4xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h1 className="font-heading font-bold text-4xl sm:text-5xl tracking-tight mb-4">
            About LifeLens AI
          </h1>
          <p className="font-body text-lg text-muted-foreground">
            AI-powered image analysis for the modern world
          </p>
        </motion.div>

        {/* Problem Statement */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="mb-16"
        >
          <div className="border-2 border-border rounded-none p-8 bg-card">
            <h2 className="font-heading font-bold text-3xl mb-6">The Problem</h2>
            <p className="font-body text-foreground leading-relaxed mb-4">
              In today's visual-first world, understanding and analyzing images is crucial for creators, developers, and businesses. However, most people lack the tools to extract meaningful insights from their images quickly and accurately.
            </p>
            <p className="font-body text-foreground leading-relaxed">
              Traditional image analysis requires expensive software, technical expertise, or time-consuming manual review. There's a need for an accessible, instant, and intelligent solution that anyone can use.
            </p>
          </div>
        </motion.section>

        {/* Solution */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="mb-16"
        >
          <div className="border-2 border-border rounded-none p-8 bg-card">
            <h2 className="font-heading font-bold text-3xl mb-6">Our Solution</h2>
            <p className="font-body text-foreground leading-relaxed mb-4">
              LifeLens AI leverages cutting-edge artificial intelligence to provide instant, detailed analysis of any image. Simply upload your photo, and our AI examines composition, colors, subjects, mood, and context to deliver professional-grade insights.
            </p>
            <p className="font-body text-foreground leading-relaxed mb-6">
              Whether you're a designer seeking feedback, a developer building visual applications, or a creator looking for inspiration, LifeLens AI democratizes image intelligence for everyone.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="border-2 border-border rounded-none p-6 bg-white">
                <Sparkles className="w-8 h-8 text-brand mb-3" />
                <h3 className="font-heading font-bold text-lg mb-2">AI-Powered</h3>
                <p className="font-body text-sm text-muted-foreground">
                  Advanced vision models trained on millions of images
                </p>
              </div>
              <div className="border-2 border-border rounded-none p-6 bg-white">
                <Code className="w-8 h-8 text-brand mb-3" />
                <h3 className="font-heading font-bold text-lg mb-2">Easy Integration</h3>
                <p className="font-body text-sm text-muted-foreground">
                  Simple API for developers to build upon
                </p>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Tech Stack */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="mb-16"
        >
          <div className="border-2 border-border rounded-none p-8 bg-card">
            <h2 className="font-heading font-bold text-3xl mb-6">Tech Stack</h2>
            <p className="font-body text-muted-foreground mb-6">
              Built with modern, production-ready technologies for speed, reliability, and scalability.
            </p>
            
            <div className="space-y-4">
              <div className="flex items-start gap-4 p-4 border-2 border-border bg-white">
                <Code className="w-6 h-6 text-brand flex-shrink-0 mt-1" />
                <div>
                  <h3 className="font-heading font-bold text-lg mb-1">Frontend</h3>
                  <p className="font-body text-sm text-muted-foreground">
                    React 19, Tailwind CSS, Framer Motion, Lucide Icons
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-4 p-4 border-2 border-border bg-white">
                <Cpu className="w-6 h-6 text-brand flex-shrink-0 mt-1" />
                <div>
                  <h3 className="font-heading font-bold text-lg mb-1">Backend</h3>
                  <p className="font-body text-sm text-muted-foreground">
                    FastAPI (Python), OpenAI GPT-5.2 Vision API, Tesseract OCR
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-4 p-4 border-2 border-border bg-white">
                <Database className="w-6 h-6 text-brand flex-shrink-0 mt-1" />
                <div>
                  <h3 className="font-heading font-bold text-lg mb-1">Database</h3>
                  <p className="font-body text-sm text-muted-foreground">
                    MongoDB for storing analysis history
                  </p>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Team Image */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.5 }}
        >
          <div className="border-2 border-border rounded-none overflow-hidden shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
            <img 
              src="https://images.pexels.com/photos/7970845/pexels-photo-7970845.jpeg"
              alt="Modern tech workspace"
              className="w-full h-64 object-cover"
            />
          </div>
        </motion.section>
      </div>
    </div>
  );
};
