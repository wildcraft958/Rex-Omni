import './ServiceSelector.css';

const SERVICES = [
    { id: 'llm', name: 'LLM Service', icon: '🤖', description: 'Object detection, pointing, keypoints, OCR, GUI' },
    { id: 'sam', name: 'SAM Segmentation', icon: '✂️', description: 'Detection + pixel-perfect masks' },
    { id: 'grounding', name: 'Phrase Grounding', icon: '🔗', description: 'Caption to region mapping' },
];

function ServiceSelector({ service, onServiceChange }) {
    return (
        <div className="glass-card">
            <h3>🎯 Select Service</h3>

            <div className="service-grid">
                {SERVICES.map((s) => (
                    <div
                        key={s.id}
                        className={`service-card ${service === s.id ? 'active' : ''}`}
                        onClick={() => onServiceChange(s.id)}
                    >
                        <div className="service-icon">{s.icon}</div>
                        <div className="service-info">
                            <h4>{s.name}</h4>
                            <p>{s.description}</p>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default ServiceSelector;
