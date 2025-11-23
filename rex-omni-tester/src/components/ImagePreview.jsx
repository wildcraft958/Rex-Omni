import { useRef, useState, useEffect } from 'react';
import './ImagePreview.css';

function normalizeBox(box, naturalW, naturalH) {
    // Accept boxes in [x1,y1,x2,y2] or [x,y,w,h], absolute pixels or relative (0-1)
    if (!box || box.length < 4) return null;
    let [a, b, c, d] = box;

    // If the largest value is <= 1.0, we assume it's a relative coordinate system (0-1).
    // Otherwise, we treat it as absolute pixel coordinates.
    const isRelative = Math.max(a, b, c, d) <= 1.0;

    if (isRelative) {
        a = a * naturalW;
        b = b * naturalH;
        c = c * naturalW;
        d = d * naturalH;
    }

    // At this point, a,b,c,d are in the absolute pixel space of the natural image.
    // Now, we determine if it's [x1,y1,x2,y2] or [x,y,w,h]
    let x1, y1, x2, y2;

    // A common heuristic: if c and d are smaller than a and b, it's likely x,y,w,h.
    // Or if x2 (c) is much larger than the image width, it's likely width.
    // A simple check is if c > a. If so, it's likely x2.
    if (c > a && d > b) {
        // Assumed to be [x1, y1, x2, y2]
        x1 = a;
        y1 = b;
        x2 = c;
        y2 = d;
    } else {
        // Assumed to be [x, y, w, h]
        x1 = a;
        y1 = b;
        x2 = a + c;
        y2 = b + d;
    }

    // clamp
    x1 = Math.max(0, Math.min(naturalW, x1));
    y1 = Math.max(0, Math.min(naturalH, y1));
    x2 = Math.max(0, Math.min(naturalW, x2));
    y2 = Math.max(0, Math.min(naturalH, y2));

    return [x1, y1, x2, y2];
}

function ImagePreview({ imageSrc, results }) {
    const imgRef = useRef(null);
    const [naturalSize, setNaturalSize] = useState({ w: 0, h: 0 });

    useEffect(() => {
        setNaturalSize({ w: 0, h: 0 });
    }, [imageSrc]);

    const onImageLoad = () => {
        const img = imgRef.current;
        if (!img) return;
        setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
    };

    const getDisplayedSize = () => {
        const img = imgRef.current;
        if (!img) return { w: 0, h: 0 };
        return { w: img.clientWidth, h: img.clientHeight };
    };

    const gatherItems = () => {
        const items = [];
        if (!results) return items;

        const { extracted_predictions, sam_results, annotations } = results;

        if (extracted_predictions) {
            Object.entries(extracted_predictions).forEach(([category, predictions]) => {
                predictions.forEach((p) => {
                    if (p.type === 'box' && p.coords) {
                        items.push({ type: 'box', data: p.coords, label: category });
                    } else if (p.type === 'point' && p.coords) {
                        items.push({ type: 'point', data: p.coords, label: category });
                    } else if (p.type === 'polygon' && p.coords) {
                        items.push({ type: 'polygon', data: p.coords, label: category });
                    } else if (p.type === 'keypoint' && p.bbox && p.keypoints) {
                        items.push({ type: 'box', data: p.bbox, label: `${category} (bbox)` });
                        Object.entries(p.keypoints).forEach(([kp_name, kp_coords]) => {
                            if (kp_coords !== 'unvisible') {
                                items.push({ type: 'point', data: kp_coords, label: kp_name });
                            }
                        });
                    }
                });
            });
        }

        if (sam_results) {
            sam_results.forEach((r) => {
                if (r.box) items.push({ type: 'box', data: r.box, label: r.category || 'sam' });
                if (r.polygons) {
                    r.polygons.forEach(p => items.push({ type: 'polygon', data: p, label: r.category || 'sam' }));
                }
            });
        }

        if (annotations) {
            annotations.forEach((ann) => {
                if (ann.boxes && Array.isArray(ann.boxes)) {
                    ann.boxes.forEach((b) => items.push({ type: 'box', data: b, label: ann.phrase || 'ann' }));
                }
            });
        }

        return items;
    };

    const renderableItems = gatherItems();

    const displayed = getDisplayedSize();

    const scale = {
        x: naturalSize.w > 0 ? displayed.w / naturalSize.w : 1,
        y: naturalSize.h > 0 ? displayed.h / naturalSize.h : 1,
    };

    return (
        <div className="image-preview-wrap">
            <h3>Image Preview</h3>
            <div className="image-preview-stage">
                <img
                    ref={imgRef}
                    src={imageSrc}
                    alt="Preview"
                    className="image-preview"
                    onLoad={onImageLoad}
                />

                <div className="overlay">
                <svg className="overlay-svg" width={displayed.w} height={displayed.h}>
                    {renderableItems.map((item, i) => {
                        if (item.type === 'box') {
                            const norm = normalizeBox(item.data, naturalSize.w || 1, naturalSize.h || 1);
                            if (!norm) return null;
                            const [x1, y1, x2, y2] = norm;
                            const left = x1 * scale.x;
                            const top = y1 * scale.y;
                            const width = Math.max(0, (x2 - x1) * scale.x);
                            const height = Math.max(0, (y2 - y1) * scale.y);

                            return (
                                <g key={i}>
                                    <rect
                                        x={left}
                                        y={top}
                                        width={width}
                                        height={height}
                                        className="overlay-box"
                                    />
                                    <text x={left} y={top - 5} className="overlay-label">
                                        {item.label}
                                    </text>
                                </g>
                            );
                        } else if (item.type === 'point') {
                            const [x, y] = item.data;
                            const left = (x / (naturalSize.w || 1)) * displayed.w;
                            const top = (y / (naturalSize.h || 1)) * displayed.h;

                            return (
                                <g key={i}>
                                    <circle cx={left} cy={top} r="5" className="overlay-point" />
                                    <text x={left + 7} y={top + 5} className="overlay-label">
                                        {item.label}
                                    </text>
                                </g>
                            );
                        } else if (item.type === 'polygon') {
                            const points = item.data
                                .map(p => `${(p[0] / (naturalSize.w || 1)) * displayed.w},${(p[1] / (naturalSize.h || 1)) * displayed.h}`)
                                .join(' ');

                            return (
                                <g key={i}>
                                    <polygon points={points} className="overlay-polygon" />
                                    <text x={(item.data[0][0] / (naturalSize.w || 1)) * displayed.w} y={(item.data[0][1] / (naturalSize.h || 1)) * displayed.h - 5} className="overlay-label">
                                        {item.label}
                                    </text>
                                </g>
                            );
                        }
                        return null;
                    })}
                </svg>
                </div>
            </div>
        </div>
    );
}

export default ImagePreview;
