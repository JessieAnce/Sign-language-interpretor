const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const predictionText = document.getElementById("prediction-text");
const confidenceText = document.getElementById("confidence-text");
const startBtn = document.getElementById("start-btn");

async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;

    setInterval(captureFrame, 1000); // every 1 second
}

async function captureFrame() {
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL("image/jpeg");

    const response = await fetch("/predict", {
        method: "POST",
        body: JSON.stringify({ image: imageData }),
        headers: { "Content-Type": "application/json" }
    });

    const data = await response.json();
    predictionText.innerText = data.label;
    confidenceText.innerText = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
}

startBtn.addEventListener("click", startCamera);
