import { useScrollReveal } from '../hooks/useScrollReveal';
import './Legend.css';

const CLASSES = [
    { name: 'Trees', color: '#228B22', icon: '🌲', desc: 'Dense vegetation canopy' },
    { name: 'Lush Bushes', color: '#9ACD32', icon: '🌿', desc: 'Green shrubs and foliage' },
    { name: 'Dry Grass', color: '#DAA520', icon: '🌾', desc: 'Dried grassland terrain' },
    { name: 'Dry Bushes', color: '#8B4513', icon: '🥀', desc: 'Withered desert scrub' },
    { name: 'Ground Clutter', color: '#808080', icon: '🪨', desc: 'Mixed debris and small objects' },
    { name: 'Flowers', color: '#FF69B4', icon: '🌸', desc: 'Flowering desert plants' },
    { name: 'Logs', color: '#A0522D', icon: '🪵', desc: 'Fallen timber and branches' },
    { name: 'Rocks', color: '#696969', icon: '⛰️', desc: 'Rock formations and boulders' },
    { name: 'Landscape', color: '#F4A460', icon: '🏜️', desc: 'Open terrain and sand' },
    { name: 'Sky', color: '#87CEEB', icon: '☁️', desc: 'Sky and atmosphere' },
];

export default function Legend() {
    const [ref, visible] = useScrollReveal(0.15);

    return (
        <section id="legend" className="section legend-section">
            <div className="container">
                <div className="reveal" ref={ref} style={{ opacity: visible ? 1 : 0, transform: visible ? 'none' : 'translateY(50px)', transition: 'all 0.8s ease' }}>
                    <span className="section-label">Terrain Map</span>
                    <h2 className="section-title">10 Classes of Desert Intelligence</h2>
                    <p className="section-subtitle">Every pixel is classified into one of these terrain categories, each with a unique color for instant visual recognition.</p>
                </div>

                <div className={`legend__grid stagger-children ${visible ? 'visible' : ''}`}>
                    {CLASSES.map((cls) => (
                        <div key={cls.name} className="legend__card glass-card">
                            <div className="legend__color-bar" style={{ background: cls.color }} />
                            <div className="legend__icon">{cls.icon}</div>
                            <h4 className="legend__name">{cls.name}</h4>
                            <p className="legend__desc">{cls.desc}</p>
                            <div className="legend__swatch" style={{ background: cls.color }}>
                                {cls.color}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
}
