
// Background service worker: orchestrates WASM inference via whisperWorker.js
let whisperWorker = null;
const pendingRequests = new Map();

function ensureWorker() {
  if (!whisperWorker) {
    whisperWorker = new Worker('whisperWorker.js', { type: 'module' });
    whisperWorker.onmessage = (e) => {
      const data = e.data;
      // Expect { status, output, error, requestId }
      if (data.requestId && pendingRequests.has(data.requestId)) {
        const { sender, sendResponse } = pendingRequests.get(data.requestId);
        sendResponse(data);
        pendingRequests.delete(data.requestId);
      }
    };
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message && message.type === 'INFER') {
    ensureWorker();
    const { audioBuffer, modelID, requestId } = message;
    // Forward to worker, include requestId
    pendingRequests.set(requestId, { sender, sendResponse });
    whisperWorker.postMessage({ audioBuffer, modelID, requestId }, [audioBuffer]);
    // Indicate async response
    return true;
  }
  return false;
});
