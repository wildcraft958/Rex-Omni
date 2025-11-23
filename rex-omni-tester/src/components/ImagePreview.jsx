import { useRef, useState, useEffect } from 'react';
import './ImagePreview.css';

function normalizeBox(box, naturalW, naturalH) {
    // Accept boxes in [x1,y1,x2,y2] or [x,y,w,h], absolute pixels or relative (0-1)
    if (!box || box.length < 4) return null;
    let [a, b, c, d] = box;

    const isRelative = Math.max(a, b, c, d) <= 1;

    if (isRelative) {
        a = a * naturalW;
        b = b * naturalH;
        c = c * naturalW;
        d = d * naturalH;
    }

    // Heuristic: if c > a and d > b and (c > naturalW || d > naturalH) it's likely x2,y2
    // Check if c represents width (w) by seeing if c <= naturalW && d <= naturalH and a + c <= naturalW
    let x1, y1, x2, y2;

    if (c > a && d > b && (a + c <= naturalW && b + d <= naturalH)) {
        // treat as x,y,w,h
        x1 = a;
        y1 = b;
        x2 = a + c;
        y2 = b + d;
    } else if (c > a && d > b) {
        // treat as x1,y1,x2,y2
        x1 = a;
        y1 = b;
        x2 = c;
        y2 = d;
    } else {
        // fallback: treat as x,y,w,h
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

    const gatherBoxes = () => {
        const boxes = [];
        if (!results) return boxes;

        const { extracted_predictions, sam_results, annotations } = results;

        if (extracted_predictions) {
            Object.entries(extracted_predictions).forEach(([category, items]) => {
                items.forEach((it) => {
                    if (it.coords && Array.isArray(it.coords) && it.coords.length >= 4) {
                        boxes.push({ box: it.coords, label: `${category}` });
                    }
                });
            });
        }

        if (sam_results) {
            sam_results.forEach((r) => {
                if (r.box) boxes.push({ box: r.box, label: r.category || 'sam' });
            });
        }

        if (annotations) {
            annotations.forEach((ann) => {
                if (ann.boxes && Array.isArray(ann.boxes)) {
                    ann.boxes.forEach((b) => boxes.push({ box: b, label: ann.phrase || 'ann' }));
                }
            });
        }

        return boxes;
    };

    const boxes = gatherBoxes();

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
                    {boxes.map((bobj, i) => {
                        const norm = normalizeBox(bobj.box, naturalSize.w || 1, naturalSize.h || 1);
                        if (!norm) return null;
                        const [x1, y1, x2, y2] = norm;
                        const left = x1 * scale.x;
                        const top = y1 * scale.y;
                        const width = Math.max(0, (x2 - x1) * scale.x);
                        const height = Math.max(0, (y2 - y1) * scale.y);

                        return (
                            <div
                                key={i}
                                className="overlay-box"
                                style={{ left: `${left}px`, top: `${top}px`, width: `${width}px`, height: `${height}px` }}
                                title={bobj.label}
                            >
                                <div className="overlay-label">{bobj.label}</div>
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
}

export default ImagePreview;
