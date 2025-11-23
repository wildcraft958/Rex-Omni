import { useState } from 'react';
import './ResultsDisplay.css';

function ResultsDisplay({ results, error, loading }) {
    const [activeTab, setActiveTab] = useState('formatted');

    if (loading) {
        return (
            <div className="glass-card results-container">
                <h3>⏳ Processing...</h3>
                <div className="loading-state">
                    <div className="spinner"></div>
                    <p>Running inference on Rex-Omni API...</p>
                    <p className="hint">First request may take 30-60s (cold start)</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="glass-card results-container">
                <h3>❌ Error</h3>
                <div className="error-state">
                    <p className="error-message">{error}</p>
                    <div className="error-hints">
                        <h4>Common Issues:</h4>
                        <ul>
                            <li>First request takes longer (model loading)</li>
                            <li>Check if image is uploaded</li>
                            <li>Verify categories are not empty</li>
                            <li>Service may be temporarily unavailable</li>
                        </ul>
                    </div>
                </div>
            </div>
        );
    }

    if (!results) {
        return (
            <div className="glass-card results-container">
                <h3>📊 Results</h3>
                <div className="empty-state">
                    <div className="empty-icon">🎯</div>
                    <p>Upload an image and run a test to see results here</p>
                </div>
            </div>
        );
    }

    const renderFormatted = () => {
        const { success, extracted_predictions, sam_results, annotations } = results;

        return (
            <div className="formatted-results">
                <div className="status-badge">
                    {success ? (
                        <span className="badge badge-success">✓ Success</span>
                    ) : (
                        <span className="badge badge-error">✗ Failed</span>
                    )}
                </div>

                {extracted_predictions && (
                    <div className="result-section">
                        <h4>🎯 Detected Objects</h4>
                        {Object.entries(extracted_predictions).map(([category, items]) => (
                            <div key={category} className="category-section">
                                <div className="category-header">
                                    <span className="category-name">{category}</span>
                                    <span className="badge">{items.length}</span>
                                </div>

                                <div className="items-list">
                                    {items.map((item, idx) => (
                                        <div key={idx} className="item-card">
                                            <div className="item-type">{item.type}</div>
                                            <div className="item-coords">
                                                {JSON.stringify(item.coords)}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {sam_results && sam_results.length > 0 && (
                    <div className="result-section">
                        <h4>✂️ SAM Segmentation</h4>
                        {sam_results.map((result, idx) => (
                            <div key={idx} className="sam-result-card">
                                <div className="sam-header">
                                    <span className="sam-category">{result.category}</span>
                                    <span className="badge">Score: {result.score.toFixed(4)}</span>
                                </div>
                                <div className="sam-details">
                                    <p><strong>Box:</strong> {JSON.stringify(result.box)}</p>
                                    <p><strong>Polygons:</strong> {result.polygons.length} polygon(s)</p>
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {annotations && annotations.length > 0 && (
                    <div className="result-section">
                        <h4>🔗 Phrase Grounding</h4>
                        {annotations.map((ann, idx) => (
                            <div key={idx} className="annotation-card">
                                <div className="annotation-phrase">"{ann.phrase}"</div>
                                <div className="annotation-details">
                                    <p><strong>Position:</strong> {ann.start_char}-{ann.end_char}</p>
                                    <p><strong>Boxes:</strong> {ann.boxes.length} match(es)</p>
                                    {ann.boxes.map((box, bidx) => (
                                        <div key={bidx} className="box-coords">
                                            Box {bidx + 1}: {JSON.stringify(box)}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="glass-card results-container">
            <div className="results-header">
                <h3>📊 Results</h3>

                <div className="tab-buttons">
                    <button
                        className={`tab-btn ${activeTab === 'formatted' ? 'active' : ''}`}
                        onClick={() => setActiveTab('formatted')}
                    >
                        Formatted
                    </button>
                    <button
                        className={`tab-btn ${activeTab === 'json' ? 'active' : ''}`}
                        onClick={() => setActiveTab('json')}
                    >
                        Raw JSON
                    </button>
                </div>
            </div>

            <div className="results-content">
                {activeTab === 'formatted' ? (
                    renderFormatted()
                ) : (
                    <pre className="code-block">
                        {JSON.stringify(results, null, 2)}
                    </pre>
                )}
            </div>
        </div>
    );
}

export default ResultsDisplay;
