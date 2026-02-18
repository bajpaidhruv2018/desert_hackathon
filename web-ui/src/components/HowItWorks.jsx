import { useScrollReveal } from '../hooks/useScrollReveal';
import './HowItWorks.css';

const STEPS = [
    {
        icon: '📤',
        num: '01',
        title: 'Upload Terrain',
        desc: 'Drop any offroad terrain image — desert dunes, rocky trails, or vegetation-heavy paths.',
    },
    {
        icon: '🧠',
        num: '02',
        title: 'AI Processes',
        desc: 'Our U-Net model with ResNet-34 encoder analyzes every pixel at 512×512 resolution.',
    },
    {
        icon: '🗺️',
        num: '03',
        title: 'Terrain Map',
        desc: 'Get a color-coded segmentation map identifying 10 terrain types for safe navigation.',
    },
];

export default function HowItWorks() {
    const [ref, visible] = useScrollReveal(0.2);

    return (
        <section id="how-it-works" className="section how-it-works">
            <div className="container">
                <div className="reveal" ref={ref} style={{ opacity: visible ? 1 : 0, transform: visible ? 'none' : 'translateY(50px)', transition: 'all 0.8s ease' }}>
                    <span className="section-label">How It Works</span>
                    <h2 className="section-title">From Photo to Terrain Intelligence</h2>
                    <p className="section-subtitle">Three simple steps to map any desert terrain for autonomous navigation.</p>
                </div>

                <div className={`how-it-works__grid stagger-children ${visible ? 'visible' : ''}`}>
                    {STEPS.map((step, i) => (
                        <div key={i} className="how-it-works__card glass-card">
                            <div className="how-it-works__num">{step.num}</div>
                            <div className="how-it-works__icon">{step.icon}</div>
                            <h3 className="how-it-works__card-title">{step.title}</h3>
                            <p className="how-it-works__card-desc">{step.desc}</p>
                            {i < STEPS.length - 1 && <div className="how-it-works__arrow">→</div>}
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
}
