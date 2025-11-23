import { useState } from 'react';
import './ImageUpload.css';

function ImageUpload({ onImageUpload }) {
    const [isDragging, setIsDragging] = useState(false);
    const [fileName, setFileName] = useState(null);

    const handleFile = (file) => {
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();

            reader.onload = (e) => {
                const base64 = e.target.result.split(',')[1];
                onImageUpload(base64, e.target.result);
                setFileName(file.name);
            };

            reader.readAsDataURL(file);
        } else {
            alert('Please upload a valid image file');
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);

        const file = e.dataTransfer.files[0];
        handleFile(file);
    };

    const handleFileInput = (e) => {
        const file = e.target.files[0];
        handleFile(file);
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => {
        setIsDragging(false);
    };

    return (
        <div className="glass-card">
            <h3>📷 Upload Image</h3>

            <div
                className={`upload-area ${isDragging ? 'drag-over' : ''}`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => document.getElementById('file-input').click()}
            >
                <input
                    id="file-input"
                    type="file"
                    accept="image/*"
                    onChange={handleFileInput}
                    style={{ display: 'none' }}
                />

                <div className="upload-icon">📁</div>

                {fileName ? (
                    <div className="file-info">
                        <p className="file-name">✅ {fileName}</p>
                        <p className="upload-hint">Click or drag to change</p>
                    </div>
                ) : (
                    <div>
                        <p className="upload-text">Drop image here or click to browse</p>
                        <p className="upload-hint">Supports JPG, PNG, WebP</p>
                    </div>
                )}
            </div>
        </div>
    );
}

export default ImageUpload;
