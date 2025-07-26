// UI logic for popup
const recordBtn = document.getElementById('record');
const statusEl = document.getElementById('status');
const outputEl = document.getElementById('output');
let recording = false;
let mediaRecorder;
let audioChunks = [];

function generateRequestId() {
  return Math.random().toString(36).slice(2) + Date.now();
}

function setRecordingUI(isRecording) {
  if (isRecording) {
    recordBtn.textContent = 'Stop Recording';
    recordBtn.style.background = 'linear-gradient(90deg, #fbeee6 0%, #f5c6aa 100%)';
    recordBtn.style.color = '#b94a00';
    statusEl.textContent = 'Recording...';
    outputEl.textContent = '';
  } else {
    recordBtn.textContent = 'Start Recording';
    recordBtn.style.background = '';
    recordBtn.style.color = '';
    statusEl.textContent = '';
  }
}

recordBtn.onclick = async () => {
  if (!recording) {
    audioChunks = [];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
      mediaRecorder.onstop = async () => {
        statusEl.textContent = 'Processing...';
        recordBtn.disabled = true;
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        const arrayBuffer = await audioBlob.arrayBuffer();
        const requestId = generateRequestId();
        chrome.runtime.sendMessage({
          type: 'INFER',
          audioBuffer: arrayBuffer,
          modelID: 'tiny_en',
          requestId
        }, (response) => {
          recordBtn.disabled = false;
          if (response && response.status === 'complete') {
            outputEl.textContent = response.output && response.output.text ? response.output.text : JSON.stringify(response.output, null, 2);
            statusEl.textContent = '';
          } else if (response && response.error) {
            outputEl.textContent = response.error;
            statusEl.textContent = '';
          } else if (response && response.status && response.message) {
            statusEl.textContent = response.message;
          } else {
            statusEl.textContent = 'Unknown response.';
          }
        });
      };
      mediaRecorder.start();
      setRecordingUI(true);
      recording = true;
    } catch (err) {
      statusEl.textContent = 'Microphone access denied.';
      outputEl.textContent = '';
    }
  } else {
    mediaRecorder.stop();
    setRecordingUI(false);
    recording = false;
  }
};
