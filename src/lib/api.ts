/**
 * Neural Canvas API Client
 * Connects frontend to FastAPI backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface CNNPrediction {
  artist: string;
  artist_confidence: number;
  style: string;
  style_confidence: number;
  genre: string;
  genre_confidence: number;
}

export interface LLMExplanation {
  model: 'model1' | 'model2';
  explanation: string;
}

export interface FullResponse {
  predictions: CNNPrediction;
  explanations: LLMExplanation[];
}

export interface HealthCheck {
  status: string;
  models_loaded: {
    cnn: boolean;
    llm_model1: boolean;
    llm_model2: boolean;
  };
  device: string;
}

class NeuralCanvasAPI {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  /**
   * Health check - verify API is running and models are loaded
   */
  async healthCheck(): Promise<HealthCheck> {
    const response = await fetch(`${this.baseURL}/health`);
    if (!response.ok) {
      throw new Error('API health check failed');
    }
    return response.json();
  }

  /**
   * Classify image with CNN only
   */
  async classifyImage(imageFile: File, modelType: 'scratch' | 'finetuned' = 'scratch'): Promise<CNNPrediction> {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await fetch(`${this.baseURL}/classify?model_type=${modelType}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Classification failed' }));
      throw new Error(error.detail || 'Classification failed');
    }

    return response.json();
  }

  /**
   * Classify with BOTH CNN models for comparison
   */
  async classifyBoth(imageFile: File): Promise<{ scratch: CNNPrediction | null; finetuned: CNNPrediction | null }> {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await fetch(`${this.baseURL}/classify-both`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Classification failed' }));
      throw new Error(error.detail || 'Classification failed');
    }

    return response.json();
  }

  /**
   * Get LLM explanation for CNN predictions
   */
  async explainClassification(
    prediction: CNNPrediction,
    model: 'model1' | 'model2' | 'both' = 'both'
  ): Promise<LLMExplanation[]> {
    const response = await fetch(`${this.baseURL}/explain`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        artist: prediction.artist,
        style: prediction.style,
        genre: prediction.genre,
        artist_confidence: prediction.artist_confidence,
        style_confidence: prediction.style_confidence,
        genre_confidence: prediction.genre_confidence,
        model: model,
      }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Explanation failed' }));
      throw new Error(error.detail || 'Explanation failed');
    }

    return response.json();
  }

  /**
   * Generate text from LLM models (for Model Arena)
   */
  async generateText(prompt: string, model: 'model1' | 'model2' | 'both' = 'both'): Promise<LLMExplanation[]> {
    console.log('Calling API:', `${this.baseURL}/generate`);
    
    try {
      const response = await fetch(`${this.baseURL}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt,
          model,
          max_tokens: 200,
        }),
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: `Generation failed: ${response.status} ${response.statusText}` }));
        throw new Error(error.detail || `Generation failed: ${response.status}`);
      }

      return response.json();
    } catch (err) {
      if (err instanceof TypeError && err.message.includes('fetch')) {
        throw new Error(`Failed to connect to API at ${this.baseURL}. Is the backend server running?`);
      }
      throw err;
    }
  }

  /**
   * Full pipeline: Image → CNN → LLM
   * This is the main endpoint!
   */
  async fullPipeline(imageFile: File): Promise<FullResponse> {
    const formData = new FormData();
    formData.append('file', imageFile);

    console.log('Calling API:', `${this.baseURL}/full`);
    
    try {
      const response = await fetch(`${this.baseURL}/full`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: `Pipeline failed: ${response.status} ${response.statusText}` }));
        throw new Error(error.detail || `Pipeline failed: ${response.status}`);
      }

      return response.json();
    } catch (err) {
      if (err instanceof TypeError && err.message.includes('fetch')) {
        throw new Error(`Failed to connect to API at ${this.baseURL}. Is the backend server running?`);
      }
      throw err;
    }
  }
}

// Export singleton instance
export const api = new NeuralCanvasAPI();

// Helper to convert CNN prediction to format expected by components
export function formatCNNPredictions(prediction: CNNPrediction) {
  return {
    artist: [{ label: prediction.artist, confidence: prediction.artist_confidence }],
    style: [{ label: prediction.style, confidence: prediction.style_confidence }],
    genre: [{ label: prediction.genre, confidence: prediction.genre_confidence }],
  };
}

