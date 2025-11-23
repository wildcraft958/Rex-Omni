import './Header.css';

function Header() {
    return (
        <header className="header">
            <div className="header-content">
                <div className="logo-section">
                    <div className="logo-icon">🤖</div>
                    <div>
                        <h1 className="logo-title">Rex-Omni Tester</h1>
                        <p className="logo-subtitle">Test all Rex-Omni API endpoints in real-time</p>
                    </div>
                </div>

                <div className="header-badges">
                    <span className="badge">LLM Service</span>
                    <span className="badge badge-success">SAM Vision</span>
                    <span className="badge badge-warning">Grounding</span>
                </div>
            </div>
        </header>
    );
}

export default Header;
