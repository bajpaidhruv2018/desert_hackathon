import { useEffect, useState, useRef } from 'react';
import { useScrollReveal } from '../hooks/useScrollReveal';
import './Metrics.css';

const IOU_DATA = [
    { name: 'Sky', value: 98.73, color: '#87CEEB' },
    { name: 'Trees', value: 87.63, color: '#228B22' },
    { name: 'Dry Grass', value: 70.37, color: '#DAA520' },
    { name: 'Lush Bushes', value: 70.14, color: '#9ACD32' },
    { name: 'Landscape', value: 69.78, color: '#F4A460' },
    { name: 'Flowers', value: 64.22, color: '#FF69B4' },
    { name: 'Logs', value: 56.21, color: '#A0522D' },
    { name: 'Dry Bushes', value: 48.93, color: '#8B4513' },
    { name: 'Rocks', value: 47.84, color: '#696969' },
    { name: 'Ground Clutter', value: 39.98, color: '#808080' },
];

function AnimatedNumber({ target, duration = 2000, visible }) {
    const [value, setValue] = useState(0);
    const startTime = useRef(null);

    useEffect(() => {
        if (!visible) return;
        const animate = (timestamp) => {
            if (!startTime.current) startTime.current = timestamp;
            const elapsed = timestamp - startTime.current;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            setValue(eased * target);
            if (progress < 1) requestAnimationFrame(animate);
        };
        requestAnimationFrame(animate);
    }, [visible, target, duration]);

    return <span>{value.toFixed(value >= 10 ? 1 : 2)}</span>;
}

export default function Metrics() {
    const [ref, visible] = useScrollReveal(0.15);
    const [barsRef, barsVisible] = useScrollReveal(0.1);

    return (
        <section id="metrics" className="section metrics-section">
            <div className="container">
                <div className="reveal" ref={ref} style={{ opacity: visible ? 1 : 0, transform: visible ? 'none' : 'translateY(50px)', transition: 'all 0.8s ease' }}>
                    <span className="section-label">Performance</span>
                    <h2 className="section-title">Battle-Tested on Desert Terrain</h2>
                    <p className="section-subtitle">Our V3 model with hybrid CrossEntropy + Dice loss delivers strong results across all terrain types.</p>
                </div>

                {/* Big metrics cards */}
                <div className={`metrics__cards stagger-children ${visible ? 'visible' : ''}`}>
                    <div className="metrics__big-card glass-card">
                        <div className="metrics__big-value">
                            <AnimatedNumber target={87.78} visible={visible} />%
                        </div>
                        <div className="metrics__big-label">Pixel Accuracy</div>
                        <div className="metrics__big-desc">Percentage of correctly classified pixels across the whole validation set</div>
                    </div>
                    <div className="metrics__big-card glass-card">
                        <div className="metrics__big-value">
                            <AnimatedNumber target={65.38} visible={visible} />%
                        </div>
                        <div className="metrics__big-label">Mean IoU</div>
                        <div className="metrics__big-desc">Average Intersection over Union across all 10 terrain classes</div>
                    </div>
                    <div className="metrics__big-card glass-card">
                        <div className="metrics__big-value metrics__big-value--model">U-Net</div>
                        <div className="metrics__big-label">Architecture</div>
                        <div className="metrics__big-desc">ResNet-34 encoder pretrained on ImageNet, fine-tuned at 512&times;512</div>
                    </div>
                </div>

                {/* Per-class IoU bars */}
                <div className="metrics__bars-section" ref={barsRef}>
                    <h3 className="metrics__bars-title">Per-Class IoU Breakdown</h3>
                    <div className="metrics__bars">
                        {IOU_DATA.map((item, i) => (
                            <div key={item.name} className="metrics__bar-row" style={{ animationDelay: `${i * 0.08}s` }}>
                                <span className="metrics__bar-name">
                                    <span className="metrics__bar-dot" style={{ background: item.color }} />
                                    {item.name}
                                </span>
                                <div className="metrics__bar-track">
                                    <div
                                        className="metrics__bar-fill"
                                        style={{
                                            width: barsVisible ? `${item.value}%` : '0%',
                                            background: item.color,
                                            transitionDelay: `${i * 0.08}s`,
                                        }}
                                    />
                                </div>
                                <span className="metrics__bar-value">{item.value}%</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </section>
    );
}
