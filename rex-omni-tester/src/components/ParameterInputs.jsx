import { useState } from 'react';
import './ParameterInputs.css';

function ParameterInputs({
    service,
    task,
    categories,
    onCategoriesChange,
    caption,
    onCaptionChange,
    keypointType,
    onKeypointTypeChange,
    visualPromptBoxes,
    onVisualPromptBoxesChange,
}) {
    const [categoryInput, setCategoryInput] = useState('');

    const addCategory = () => {
        if (categoryInput.trim()) {
            onCategoriesChange([...categories, categoryInput.trim()]);
            setCategoryInput('');
        }
    };

    const removeCategory = (index) => {
        onCategoriesChange(categories.filter((_, i) => i !== index));
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            addCategory();
        }
    };

    const needsCategories = () => {
        if (service === 'sam') return true;
        if (service === 'llm') {
            return !['ocr_box', 'ocr_polygon', 'keypoint'].includes(task);
        }
        return false;
    };

    const needsCaption = () => {
        return service === 'grounding';
    };

    const needsKeypointType = () => {
        return service === 'llm' && task === 'keypoint';
    };

    const needsVisualPromptBoxes = () => {
        return service === 'llm' && task === 'visual_prompting';
    };

    return (
        <div className="glass-card">
            <h3>🎛️ Parameters</h3>

            {needsCategories() && (
                <div className="param-section">
                    <label className="label">Categories</label>

                    <div className="category-input-group">
                        <input
                            type="text"
                            className="input"
                            value={categoryInput}
                            onChange={(e) => setCategoryInput(e.target.value)}
                            onKeyPress={handleKeyPress}
                            placeholder="e.g., person, cup, laptop"
                        />
                        <button className="btn btn-secondary" onClick={addCategory}>
                            + Add
                        </button>
                    </div>

                    <div className="category-list">
                        {categories.map((cat, index) => (
                            <div key={index} className="category-tag">
                                <span>{cat}</span>
                                <button
                                    className="remove-btn"
                                    onClick={() => removeCategory(index)}
                                    aria-label="Remove category"
                                >
                                    ×
                                </button>
                            </div>
                        ))}
                    </div>

                    {categories.length === 0 && (
                        <p className="hint">Add at least one category</p>
                    )}
                </div>
            )}

            {needsCaption() && (
                <div className="param-section">
                    <label className="label">Caption</label>
                    <textarea
                        className="textarea"
                        value={caption}
                        onChange={(e) => onCaptionChange(e.target.value)}
                        placeholder="Describe the scene (e.g., A person sitting at a table with a laptop)"
                    />
                    <p className="hint">Natural language description of the image</p>
                </div>
            )}

            {needsKeypointType() && (
                <div className="param-section">
                    <label className="label">Keypoint Type</label>
                    <select
                        className="select"
                        value={keypointType}
                        onChange={(e) => onKeypointTypeChange(e.target.value)}
                    >
                        <option value="person">Person (17 COCO keypoints)</option>
                        <option value="hand">Hand keypoints</option>
                        <option value="animal">Animal keypoints</option>
                    </select>
                </div>
            )}

            {needsVisualPromptBoxes() && (
                <div className="param-section">
                    <label className="label">Visual Prompt Boxes (Optional)</label>
                    <textarea
                        className="textarea"
                        value={visualPromptBoxes}
                        onChange={(e) => onVisualPromptBoxesChange(e.target.value)}
                        placeholder="[[x1, y1, x2, y2]]"
                        rows="3"
                    />
                    <p className="hint">JSON array of reference boxes: [[x1,y1,x2,y2], ...]</p>
                </div>
            )}

            {!needsCategories() && !needsCaption() && !needsKeypointType() && !needsVisualPromptBoxes() && (
                <div className="param-section">
                    <p className="info-text">✨ No additional parameters needed for this task</p>
                </div>
            )}
        </div>
    );
}

export default ParameterInputs;
