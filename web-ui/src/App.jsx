import Navbar from './components/Navbar';
import Hero from './components/Hero';
import HowItWorks from './components/HowItWorks';
import Upload from './components/Upload';
import Metrics from './components/Metrics';
import Legend from './components/Legend';
import Footer from './components/Footer';

function App() {
  return (
    <>
      <Navbar />
      <Hero />
      <div className="desert-divider" />
      <HowItWorks />
      <div className="desert-divider" />
      <Upload />
      <div className="desert-divider" />
      <Metrics />
      <div className="desert-divider" />
      <Legend />
      <Footer />
    </>
  );
}

export default App;
