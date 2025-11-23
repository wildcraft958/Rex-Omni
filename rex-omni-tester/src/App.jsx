import { useState } from 'react';
import './App.css';
import ImageUpload from './components/ImageUpload';
import ServiceSelector from './components/ServiceSelector';
import TaskSelector from './components/TaskSelector';
import ParameterInputs from './components/ParameterInputs';
import ResultsDisplay from './components/ResultsDisplay';
import ImagePreview from './components/ImagePreview';
import Header from './components/Header';

const API_ENDPOINTS = {
  llm: 'https://animeshraj958--rex-llm-service-rex-inference.modal.run',
  sam: 'https://animeshraj958--rex-vision-service-api-sam.modal.run',
  grounding: 'https://animeshraj958--rex-vision-service-api-grounding.modal.run',
};

function App() {
  const [imageData, setImageData] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [service, setService] = useState('llm');
  const [task, setTask] = useState('detection');
  const [categories, setCategories] = useState(['person', 'cup', 'laptop']);
  const [caption, setCaption] = useState('');
  const [keypointType, setKeypointType] = useState('person');
  const [visualPromptBoxes, setVisualPromptBoxes] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleImageUpload = (base64Data, preview) => {
    setImageData(base64Data);
    setImagePreview(preview);
    setResults(null);
    setError(null);
  };

  const handleTest = async () => {
    if (!imageData) {
      setError('Please upload an image first');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      let payload = { image: imageData };
      let url = API_ENDPOINTS[service];

      if (service === 'llm') {
        payload.task = task;

        if (task === 'keypoint') {
          payload.keypoint_type = keypointType;
        } else if (task === 'visual_prompting') {
          if (visualPromptBoxes) {
            try {
              payload.visual_prompt_boxes = JSON.parse(visualPromptBoxes);
            } catch (e) {
              throw new Error('Invalid visual prompt boxes format. Use: [[x1,y1,x2,y2]]');
            }
          }
          payload.categories = categories;
        } else if (task !== 'ocr_box' && task !== 'ocr_polygon') {
          payload.categories = categories;
        }
      } else if (service === 'sam') {
        payload.categories = categories;
      } else if (service === 'grounding') {
        payload.caption = caption;
      }

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <Header />

      <main className="container">
        <div className="grid">
          {/* Left Panel - Configuration */}
          <div className="config-panel">
            <ImageUpload onImageUpload={handleImageUpload} />

            <ServiceSelector
              service={service}
              onServiceChange={(newService) => {
                setService(newService);
                setResults(null);
                setError(null);
              }}
            />

            {service === 'llm' && (
              <TaskSelector
                task={task}
                onTaskChange={(newTask) => {
                  setTask(newTask);
                  setResults(null);
                  setError(null);
                }}
              />
            )}

            <ParameterInputs
              service={service}
              task={task}
              categories={categories}
              onCategoriesChange={setCategories}
              caption={caption}
              onCaptionChange={setCaption}
              keypointType={keypointType}
              onKeypointTypeChange={setKeypointType}
              visualPromptBoxes={visualPromptBoxes}
              onVisualPromptBoxesChange={setVisualPromptBoxes}
            />

            <button
              className="btn btn-primary btn-test"
              onClick={handleTest}
              disabled={loading || !imageData}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Processing...
                </>
              ) : (
                <>
                  <span>🚀</span>
                  Run Test
                </>
              )}
            </button>
          </div>

          {/* Right Panel - Results */}
          <div className="results-panel">
            {imagePreview && (
              <div className="image-preview-container glass-card">
                <ImagePreview imageSrc={imagePreview} results={results} />
              </div>
            )}

            <ResultsDisplay
              results={results}
              error={error}
              loading={loading}
            />
          </div>
        </div>
      </main>

      <footer className="footer">
        <p>
          Rex-Omni API Tester • Built with React + Vite •{' '}
          <a href="https://github.com/IDEA-Research/Rex-Omni" target="_blank" rel="noopener noreferrer">
            GitHub
          </a>
        </p>
      </footer>
    </div>
  );
}

export default App;
