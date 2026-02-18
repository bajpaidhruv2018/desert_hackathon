import './Footer.css';

export default function Footer() {
    return (
        <footer className="footer">
            <div className="desert-divider" />
            <div className="container footer__inner">
                <div className="footer__brand">
                    <span className="footer__logo">🏜️ DesertNav<span className="footer__logo-ai">.AI</span></span>
                    <p className="footer__tagline">Intelligent terrain mapping for autonomous offroad navigation.</p>
                </div>

                <div className="footer__team">
                    <h4 className="footer__heading">Team</h4>
                    <ul className="footer__list">
                        <li>Dhruv Bajpai</li>
                        <li>Samarth Shukla</li>
                        <li>Kshitij Trivedi</li>
                    </ul>
                </div>

                <div className="footer__tech">
                    <h4 className="footer__heading">Tech Stack</h4>
                    <ul className="footer__list">
                        <li>PyTorch + U-Net</li>
                        <li>ResNet-34 Encoder</li>
                        <li>React + Vite</li>
                    </ul>
                </div>
            </div>

            <div className="footer__bottom container">
                <span>Built for Startathon Desert Hackathon</span>
                <span>© 2025 DesertNav.AI</span>
            </div>
        </footer>
    );
}
