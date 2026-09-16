import { useRef, useState, useEffect } from 'react';

function App() {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null);

  // 1. Setup: Make the canvas black (just like the MNIST images)
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  // 2. Mouse Tracking: Start drawing white lines when the mouse clicks
  const startDrawing = (e) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 20; // Thick enough to mimic a marker
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    setIsDrawing(true);
  };

  const draw = (e) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.lineTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    ctx.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  // 3. Clear button logic
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
  };

  // 4. The Bridge: Shrink the image and send it to Python
  const handlePredict = async () => {
    const canvas = canvasRef.current;
    
    // Create a tiny invisible 28x28 canvas to shrink our drawing down
    const smallCanvas = document.createElement('canvas');
    smallCanvas.width = 28;
    smallCanvas.height = 28;
    const smallCtx = smallCanvas.getContext('2d');
    
    // Draw the big image onto the tiny canvas
    smallCtx.drawImage(canvas, 0, 0, 28, 28);
    
    // Extract the raw pixel data
    const imgData = smallCtx.getImageData(0, 0, 28, 28);
    const pixels = [];
    
    // The data is in RGBA format (Red, Green, Blue, Alpha). 
    // We only need one color channel, so we jump by 4 to grab the Red value of each pixel.
    for (let i = 0; i < imgData.data.length; i += 4) {
      pixels.push(imgData.data[i] / 255.0); // Normalize to 0-1 just like in train.py
    }

    // Send the 784 pixels to FastAPI
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pixels: pixels })
      });
      const data = await response.json();
      setPrediction(data.prediction); // Save the Python answer to React state
    } catch (error) {
      console.error("Error connecting to API:", error);
    }
  };

  return (
    <div style={{ textAlign: 'center', marginTop: '50px', fontFamily: 'sans-serif' }}>
      <h1>Draw a Number (0-9)</h1>
      <p>Draw on the black square below, then click Predict!</p>
      
      {/* The actual drawing surface */}
      <canvas
        ref={canvasRef}
        width={280}
        height={280}
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseOut={stopDrawing}
        style={{ border: '2px solid #333', cursor: 'crosshair', borderRadius: '8px' }}
      />
      
      <br />
      
      <button onClick={clearCanvas} style={{ margin: '10px', padding: '10px 20px', fontSize: '16px', cursor: 'pointer' }}>
        Clear
      </button>
      <button onClick={handlePredict} style={{ margin: '10px', padding: '10px 20px', fontSize: '16px', cursor: 'pointer', backgroundColor: '#007BFF', color: 'white', border: 'none', borderRadius: '5px' }}>
        Predict
      </button>
      
      {/* Show the result ONLY if Python has sent an answer back */}
      {prediction !== null && (
        <div style={{ marginTop: '20px', padding: '15px', backgroundColor: '#e8f5e9', borderRadius: '5px', display: 'inline-block' }}>
          <h2 style={{color: 'blue'}}>Prediction: {prediction}</h2>
        </div>
      )}
    </div>
  );
}

export default App;