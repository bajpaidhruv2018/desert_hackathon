import { useState, useEffect } from 'react';
import './Navbar.css';

const NAV_LINKS = [
    { id: 'hero', label: 'Home' },
    { id: 'how-it-works', label: 'How It Works' },
    { id: 'try-it', label: 'Try It' },
    { id: 'metrics', label: 'Performance' },
    { id: 'legend', label: 'Terrain Map' },
];

export default function Navbar() {
    const [scrolled, setScrolled] = useState(false);
    const [active, setActive] = useState('hero');

    useEffect(() => {
        const onScroll = () => {
            setScrolled(window.scrollY > 60);

            // Update active section
            for (const link of [...NAV_LINKS].reverse()) {
                const el = document.getElementById(link.id);
                if (el && el.getBoundingClientRect().top <= 120) {
                    setActive(link.id);
                    break;
                }
            }
        };
        window.addEventListener('scroll', onScroll, { passive: true });
        return () => window.removeEventListener('scroll', onScroll);
    }, []);

    return (
        <nav className={`navbar ${scrolled ? 'navbar--scrolled' : ''}`}>
            <div className="navbar__inner container">
                <a href="#hero" className="navbar__logo">
                    <span className="navbar__logo-icon">🏜️</span>
                    <span className="navbar__logo-text">DesertNav<span className="navbar__logo-ai">.AI</span></span>
                </a>
                <ul className="navbar__links">
                    {NAV_LINKS.map(link => (
                        <li key={link.id}>
                            <a
                                href={`#${link.id}`}
                                className={`navbar__link ${active === link.id ? 'navbar__link--active' : ''}`}
                            >
                                {link.label}
                            </a>
                        </li>
                    ))}
                </ul>
            </div>
        </nav>
    );
}
