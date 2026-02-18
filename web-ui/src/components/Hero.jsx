import { useEffect, useRef } from 'react';
import './Hero.css';

export default function Hero() {
    const particlesRef = useRef(null);

    useEffect(() => {
        const canvas = particlesRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        let animationId;
        let particles = [];

        const resize = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        };
        resize();
        window.addEventListener('resize', resize);

        // Sand particles
        for (let i = 0; i < 60; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                size: Math.random() * 2 + 0.5,
                speedX: Math.random() * 0.8 + 0.2,
                speedY: Math.random() * 0.3 - 0.15,
                opacity: Math.random() * 0.4 + 0.1,
            });
        }

        const animate = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            particles.forEach(p => {
                p.x += p.speedX;
                p.y += p.speedY;
                if (p.x > canvas.width) { p.x = -5; p.y = Math.random() * canvas.height; }
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(212, 160, 62, ${p.opacity})`;
                ctx.fill();
            });
            animationId = requestAnimationFrame(animate);
        };
        animate();

        return () => {
            cancelAnimationFrame(animationId);
            window.removeEventListener('resize', resize);
        };
    }, []);

    return (
        <section id="hero" className="hero">
            <canvas ref={particlesRef} className="hero__particles" />
            <div className="hero__bg" />
            <div className="hero__overlay" />

            <div className="hero__content container">
                <div className="hero__badge">🚀 Startathon Desert Hackathon</div>
                <h1 className="hero__title">
                    Desert Navigator
                    <span className="hero__title-accent"> AI</span>
                </h1>
                <p className="hero__subtitle">
                    Intelligent terrain segmentation for autonomous offroad navigation.
                    Our AI maps 10 terrain types in real-time — from sand dunes to rocky
                    outcrops — keeping your vehicle on the safest path.
                </p>
                <div className="hero__actions">
                    <a href="#try-it" className="btn-glow">
                        <span>Try It Now</span>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14M12 5l7 7-7 7" /></svg>
                    </a>
                    <a href="#how-it-works" className="hero__learn-more">
                        Learn More ↓
                    </a>
                </div>

                <div className="hero__stats">
                    <div className="hero__stat">
                        <span className="hero__stat-value">87.78%</span>
                        <span className="hero__stat-label">Pixel Accuracy</span>
                    </div>
                    <div className="hero__stat-divider" />
                    <div className="hero__stat">
                        <span className="hero__stat-value">65.38%</span>
                        <span className="hero__stat-label">Mean IoU</span>
                    </div>
                    <div className="hero__stat-divider" />
                    <div className="hero__stat">
                        <span className="hero__stat-value">10</span>
                        <span className="hero__stat-label">Terrain Classes</span>
                    </div>
                    <div className="hero__stat-divider" />
                    <div className="hero__stat">
                        <span className="hero__stat-value">512px</span>
                        <span className="hero__stat-label">Resolution</span>
                    </div>
                </div>
            </div>

            <div className="hero__scroll-indicator">
                <div className="hero__scroll-mouse">
                    <div className="hero__scroll-wheel" />
                </div>
                <span>Scroll to explore</span>
            </div>
        </section>
    );
}
