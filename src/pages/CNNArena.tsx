import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import { Upload, Bug, Sparkles, Image as ImageIcon, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import Sidebar from "@/components/layout/Sidebar";
import TopBar from "@/components/layout/TopBar";
import CNNOutputCard from "@/components/arena/CNNOutputCard";

const MOCK_CLASSIFICATIONS = {
  scratch: {
    predictions: [
      { label: "Impressionism", confidence: 0.34 },
      { label: "Abstract", confidence: 0.28 },
      { label: "Landscape", confidence: 0.21 },
      { label: "Portrait", confidence: 0.12 },
      { label: "Still Life", confidence: 0.05 },
    ],
    delay: 3000,
  },
  finetuned: {
    predictions: [
      { label: "Post-Impressionism", confidence: 0.87 },
      { label: "Impressionism", confidence: 0.08 },
      { label: "Expressionism", confidence: 0.03 },
      { label: "Fauvism", confidence: 0.01 },
      { label: "Symbolism", confidence: 0.01 },
    ],
    delay: 1200,
  },
};

const CNNArena = () => {
  const [activeSection, setActiveSection] = useState("cnn");
  const [devMode, setDevMode] = useState(false);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const [scratchResults, setScratchResults] = useState<typeof MOCK_CLASSIFICATIONS.scratch.predictions | null>(null);
  const [finetunedResults, setFinetunedResults] = useState<typeof MOCK_CLASSIFICATIONS.finetuned.predictions | null>(null);

  const [scratchLoading, setScratchLoading] = useState(false);
  const [finetunedLoading, setFinetunedLoading] = useState(false);

  const handleImageUpload = useCallback((file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      setUploadedImage(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      handleImageUpload(file);
    }
  }, [handleImageUpload]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleImageUpload(file);
    }
  }, [handleImageUpload]);

  const handleClassify = () => {
    if (!uploadedImage) return;

    setScratchResults(null);
    setFinetunedResults(null);

    setIsProcessing(true);
    setScratchLoading(true);
    setFinetunedLoading(true);

    // Fine-tuned finishes first
    setTimeout(() => {
      setFinetunedLoading(false);
      setFinetunedResults(MOCK_CLASSIFICATIONS.finetuned.predictions);
    }, MOCK_CLASSIFICATIONS.finetuned.delay);

    // Scratch finishes last
    setTimeout(() => {
      setScratchLoading(false);
      setScratchResults(MOCK_CLASSIFICATIONS.scratch.predictions);
      setIsProcessing(false);
    }, MOCK_CLASSIFICATIONS.scratch.delay);
  };

  const clearImage = () => {
    setUploadedImage(null);
    setScratchResults(null);
    setFinetunedResults(null);
  };

  return (
    <div className="min-h-screen bg-background">
      <Sidebar activeSection={activeSection} onSectionChange={setActiveSection} />
      <TopBar />

      <main className="pl-16 pt-14 h-[calc(100vh-0px)] overflow-hidden flex flex-col">
        {/* Hero gradient background */}
        <div className="absolute inset-0 pl-16 pt-14 pointer-events-none overflow-hidden">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-[120px]" />
          <div className="absolute top-20 right-1/4 w-80 h-80 bg-secondary/10 rounded-full blur-[100px]" />
        </div>

        <div className="relative flex-1 flex flex-col max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-4 w-full">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-3"
          >
            <h1 className="font-serif text-2xl md:text-3xl font-bold tracking-tight mb-1">
              <span className="text-foreground">The </span>
              <span className="bg-gradient-to-r from-primary via-secondary to-accent bg-clip-text text-transparent">
                CNN Arena
              </span>
            </h1>
            <p className="text-muted-foreground text-xs font-sans">
              One image. Two architectures. Observe classification accuracy.
            </p>
          </motion.div>

          {/* Main Grid: Image + 2 CNN Cards */}
          <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-4 min-h-0">
            {/* Image Upload Zone */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.1 }}
              className="glass-panel rounded-xl p-4 flex flex-col"
            >
              <div className="flex items-center gap-2 mb-3">
                <div className="w-8 h-8 rounded-lg bg-accent/20 flex items-center justify-center">
                  <ImageIcon className="w-4 h-4 text-accent" />
                </div>
                <div>
                  <h3 className="font-serif text-sm font-semibold text-foreground">Input Image</h3>
                  <p className="text-[10px] text-muted-foreground font-mono">Drop or select artwork</p>
                </div>
              </div>

              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                className={`flex-1 rounded-lg border-2 border-dashed transition-all duration-300 flex items-center justify-center relative overflow-hidden ${
                  isDragging
                    ? "border-accent bg-accent/10"
                    : uploadedImage
                    ? "border-border"
                    : "border-muted-foreground/30 hover:border-accent/50"
                }`}
              >
                {uploadedImage ? (
                  <>
                    <img
                      src={uploadedImage}
                      alt="Uploaded artwork"
                      className="w-full h-full object-contain"
                    />
                    <button
                      onClick={clearImage}
                      className="absolute top-2 right-2 p-1 rounded-full bg-background/80 hover:bg-background text-muted-foreground hover:text-foreground transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </>
                ) : (
                  <label className="flex flex-col items-center justify-center cursor-pointer p-4 text-center">
                    <Upload className="w-8 h-8 text-muted-foreground mb-2" />
                    <span className="text-xs text-muted-foreground">
                      Drop image here or click to browse
                    </span>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleFileSelect}
                      className="hidden"
                    />
                  </label>
                )}
              </div>
            </motion.div>

            {/* CNN Output Cards */}
            <CNNOutputCard
              title="The Novice"
              subtitle="CNN from Scratch"
              params="12M Params"
              statusLabel="Val Acc: 34%"
              icon={Bug}
              predictions={scratchResults}
              isLoading={scratchLoading}
              variant="scratch"
              devMode={devMode}
              metrics={{
                inferenceTime: "1.24s",
                accuracy: "34.2%",
                layers: "8",
              }}
            />

            <CNNOutputCard
              title="The Expert"
              subtitle="Fine-Tuned ResNet50"
              params="WikiArt Dataset"
              statusLabel="Val Acc: 87%"
              icon={Sparkles}
              predictions={finetunedResults}
              isLoading={finetunedLoading}
              variant="finetuned"
              devMode={devMode}
              metrics={{
                inferenceTime: "0.18s",
                accuracy: "87.4%",
                layers: "50",
              }}
            />
          </div>

          {/* Classify Button */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="flex justify-center mt-4"
          >
            <Button
              onClick={handleClassify}
              disabled={isProcessing || !uploadedImage}
              size="lg"
              className="px-8 text-sm font-semibold bg-gradient-to-r from-primary to-secondary text-primary-foreground rounded-xl"
            >
              <Upload className="w-4 h-4 mr-2" />
              Classify Artwork
            </Button>
          </motion.div>
        </div>
      </main>
    </div>
  );
};

export default CNNArena;
