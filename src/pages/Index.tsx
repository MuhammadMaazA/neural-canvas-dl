import { useState, useCallback, useEffect } from "react";
import Sidebar from "@/components/layout/Sidebar";
import TopBar from "@/components/layout/TopBar";
import HeroSection from "@/components/sections/HeroSection";
import AnalyzerSection from "@/components/analyzer/AnalyzerSection";
import GenerativeLabSection from "@/components/sections/GenerativeLabSection";
import Footer from "@/components/sections/Footer";
import { api, formatCNNPredictions } from "@/lib/api";

// Demo data - Starry Night
const DEMO_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg";

const DEMO_CNN_RESULTS = {
  artist: [
    { label: "Vincent van Gogh", confidence: 87 },
    { label: "Paul Gauguin", confidence: 6 },
    { label: "Claude Monet", confidence: 4 },
  ],
  style: [
    { label: "Post-Impressionism", confidence: 92 },
    { label: "Expressionism", confidence: 5 },
    { label: "Impressionism", confidence: 2 },
  ],
  genre: [
    { label: "Landscape", confidence: 78 },
    { label: "Cityscape", confidence: 15 },
    { label: "Abstract", confidence: 4 },
  ],
};

// Removed hardcoded LLM outputs - now using real API responses

const Index = () => {
  const [activeSection, setActiveSection] = useState("analyzer");
  const [devMode, setDevMode] = useState(false);
  
  // Analyzer state
  const [loadedImage, setLoadedImage] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [artistPredictions, setArtistPredictions] = useState<{ label: string; confidence: number }[]>([]);
  const [stylePredictions, setStylePredictions] = useState<{ label: string; confidence: number }[]>([]);
  const [genrePredictions, setGenrePredictions] = useState<{ label: string; confidence: number }[]>([]);
  
  // LLM state
  const [selectedModel, setSelectedModel] = useState<"scratch" | "distilgpt2" | "hosted">("distilgpt2");
  const [llmOutput, setLlmOutput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [llmExplanations, setLlmExplanations] = useState<{ model1?: string; model2?: string }>({});
  const [error, setError] = useState<string | null>(null);

  // Suggested prompt for diffusion
  const [suggestedPrompt, setSuggestedPrompt] = useState(
    "A landscape in the style of Van Gogh with vivid swirling skies and bold brushstrokes"
  );

  const handleImageLoad = useCallback(async (imageUrl: string, file?: File) => {
    setLoadedImage(imageUrl);
    if (file) {
      setUploadedFile(file);
    }
    setIsScanning(true);
    setIsGenerating(true);
    setLlmOutput("");
    setError(null);
    
    // Clear previous results
    setArtistPredictions([]);
    setStylePredictions([]);
    setGenrePredictions([]);
    setLlmExplanations({});

    try {
      // Always require a file for API calls
      if (!file) {
        setError('Please upload an image file');
        setIsScanning(false);
        setIsGenerating(false);
        return;
      }

      // Call full pipeline API - this gets CNN + LLM from both models
      console.log('Calling API with file:', file.name, file.size);
      const result = await api.fullPipeline(file);
      
      console.log('Full pipeline result:', result); // Debug log
      console.log('Result keys:', Object.keys(result));
      
      // Format CNN predictions
      if (!result.predictions) {
        throw new Error('No predictions in API response');
      }
      
      const formatted = formatCNNPredictions(result.predictions);
      setArtistPredictions(formatted.artist);
      setStylePredictions(formatted.style);
      setGenrePredictions(formatted.genre);
      
      // Store LLM explanations from both models
      const explanations: { model1?: string; model2?: string } = {};
      if (result.explanations && Array.isArray(result.explanations)) {
        console.log('Processing explanations:', result.explanations.length);
        result.explanations.forEach((exp: any) => {
          console.log('Explanation:', exp.model, 'length:', exp.explanation?.length);
          if (exp.model === 'model1' && exp.explanation) {
            explanations.model1 = exp.explanation;
          } else if (exp.model === 'model2' && exp.explanation) {
            explanations.model2 = exp.explanation;
          }
        });
        console.log('Final explanations:', explanations); // Debug log
      } else {
        console.error('No LLM explanations in response!', result);
        throw new Error('No LLM explanations received from API');
      }
      
      if (!explanations.model1 && !explanations.model2) {
        throw new Error('Both LLM models returned empty explanations');
      }
      
      setLlmExplanations(explanations);
      
      // Set output based on selected model
      if (selectedModel === 'scratch' && explanations.model1) {
        setLlmOutput(explanations.model1);
      } else if (selectedModel === 'distilgpt2' && explanations.model2) {
        setLlmOutput(explanations.model2);
      } else if (selectedModel === 'hosted' && explanations.model2) {
        setLlmOutput(explanations.model2);
      } else if (explanations.model1) {
        setLlmOutput(explanations.model1); // Fallback to model1
      } else if (explanations.model2) {
        setLlmOutput(explanations.model2); // Fallback to model2
      } else {
        throw new Error('No valid LLM explanation available');
      }
      
      setIsScanning(false);
      setIsGenerating(false);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Analysis failed';
      console.error('Pipeline error:', err);
      setError(errorMsg);
      setLlmOutput(`Error: ${errorMsg}. Please check the browser console for details.`);
      setIsScanning(false);
      setIsGenerating(false);
    }
  }, [selectedModel]);

  const handleLoadDemo = useCallback(async () => {
    // Fetch demo image and convert to File
    try {
      const response = await fetch(DEMO_IMAGE_URL);
      const blob = await response.blob();
      const file = new File([blob], 'demo-starry-night.jpg', { type: 'image/jpeg' });
      
      // Create object URL for display
      const imageUrl = URL.createObjectURL(file);
      await handleImageLoad(imageUrl, file);
    } catch (err) {
      setError('Failed to load demo image');
      console.error(err);
    }
  }, [handleImageLoad]);

  // Update LLM output when model changes (if already analyzed)
  useEffect(() => {
    if (Object.keys(llmExplanations).length > 0) {
      // Immediately switch to the selected model's explanation
      if (selectedModel === 'scratch' && llmExplanations.model1) {
        setLlmOutput(llmExplanations.model1);
      } else if (selectedModel === 'distilgpt2' && llmExplanations.model2) {
        setLlmOutput(llmExplanations.model2);
      } else if (selectedModel === 'hosted') {
        // For hosted, use model2 as fallback
        setLlmOutput(llmExplanations.model2 || llmExplanations.model1 || '');
      } else {
        setLlmOutput(llmExplanations.model1 || llmExplanations.model2 || '');
      }
    }
  }, [selectedModel, llmExplanations]);

  return (
    <div className="min-h-screen bg-background">
      <Sidebar activeSection={activeSection} onSectionChange={setActiveSection} />
      <TopBar />
      
      <main className="pl-16 pt-14 min-h-screen overflow-x-hidden scrollbar-thin">
        <HeroSection />
        
        {error && (
          <div className="max-w-7xl mx-auto px-6 mb-4">
            <div className="bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-3 rounded-lg text-sm">
              Error: {error}
            </div>
          </div>
        )}
        
        <AnalyzerSection
          loadedImage={loadedImage}
          isScanning={isScanning}
          onImageLoad={handleImageLoad}
          onLoadDemo={handleLoadDemo}
          artistPredictions={artistPredictions}
          stylePredictions={stylePredictions}
          genrePredictions={genrePredictions}
          selectedModel={selectedModel}
          onModelChange={setSelectedModel}
          llmOutput={llmOutput}
          isGenerating={isGenerating}
          devMode={devMode}
        />
        
        <GenerativeLabSection 
          suggestedPrompt={suggestedPrompt}
          devMode={devMode}
        />
        
        <Footer />
      </main>
    </div>
  );
};

export default Index;
