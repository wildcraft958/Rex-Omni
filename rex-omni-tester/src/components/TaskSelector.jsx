import './TaskSelector.css';

const TASKS = [
    { id: 'detection', name: 'Object Detection', icon: '📦', description: 'Bounding boxes' },
    { id: 'pointing', name: 'Pointing', icon: '📍', description: 'Center points' },
    { id: 'visual_prompting', name: 'Visual Prompting', icon: '👁️', description: 'Find similar' },
    { id: 'keypoint', name: 'Keypoint Detection', icon: '🧍', description: 'Body keypoints' },
    { id: 'ocr_box', name: 'OCR - Boxes', icon: '📝', description: 'Word boxes' },
    { id: 'ocr_polygon', name: 'OCR - Polygons', icon: '📐', description: 'Text polygons' },
    { id: 'gui_grounding', name: 'GUI Grounding', icon: '🖥️', description: 'UI elements' },
    { id: 'gui_pointing', name: 'GUI Pointing', icon: '🖱️', description: 'UI centers' },
];

function TaskSelector({ task, onTaskChange }) {
    return (
        <div className="glass-card">
            <h3>⚙️ Select Task</h3>

            <div className="task-grid">
                {TASKS.map((t) => (
                    <div
                        key={t.id}
                        className={`task-chip ${task === t.id ? 'active' : ''}`}
                        onClick={() => onTaskChange(t.id)}
                        title={t.description}
                    >
                        <span className="task-icon">{t.icon}</span>
                        <span className="task-name">{t.name}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default TaskSelector;
