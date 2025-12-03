import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "next-themes";
import Index from "./pages/Index";
import ModelArena from "./pages/ModelArena";
import CNNArena from "./pages/CNNArena";
import DiffusionLab from "./pages/DiffusionLab";
import ESRGANLab from "./pages/ESRGANLab";
import NSTLab from "./pages/NSTLab";
import NotFound from "./pages/NotFound";
const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem={false}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/model-arena" element={<ModelArena />} />
            <Route path="/cnn-arena" element={<CNNArena />} />
            <Route path="/diffusion-lab" element={<DiffusionLab />} />
            <Route path="/esrgan-lab" element={<ESRGANLab />} />
            <Route path="/nst-lab" element={<NSTLab />} />
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </ThemeProvider>
  </QueryClientProvider>
);

export default App;
