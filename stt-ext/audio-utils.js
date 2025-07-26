// Utility: Convert WebM/Blob audio to PCM 16kHz mono float32 (placeholder)

/**
 * Decodes audio (ArrayBuffer or Blob) to PCM Float32Array at 16kHz mono.
 * @param {ArrayBuffer|Blob} audioBufferOrBlob
 * @param {number} [targetSampleRate=16000]
 * @returns {Promise<Float32Array>}
 */
export async function decodeAudioToPCM(audioBufferOrBlob, targetSampleRate = 16000) {
  let arrayBuffer;
  if (audioBufferOrBlob instanceof Blob) {
    arrayBuffer = await audioBufferOrBlob.arrayBuffer();
  } else {
    arrayBuffer = audioBufferOrBlob;
  }
  // Decode audio data
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
  // If already mono, use channel 0; else average channels
  let channelData;
  if (audioBuffer.numberOfChannels === 1) {
    channelData = audioBuffer.getChannelData(0);
  } else {
    // Average all channels
    const length = audioBuffer.length;
    channelData = new Float32Array(length);
    for (let c = 0; c < audioBuffer.numberOfChannels; c++) {
      const data = audioBuffer.getChannelData(c);
      for (let i = 0; i < length; i++) {
        channelData[i] += data[i] / audioBuffer.numberOfChannels;
      }
    }
  }
  // Resample if needed
  if (audioBuffer.sampleRate !== targetSampleRate) {
    const offlineCtx = new OfflineAudioContext(1, Math.ceil(channelData.length * targetSampleRate / audioBuffer.sampleRate), targetSampleRate);
    const buffer = offlineCtx.createBuffer(1, channelData.length, audioBuffer.sampleRate);
    buffer.copyToChannel(channelData, 0);
    const source = offlineCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(offlineCtx.destination);
    source.start();
    const renderedBuffer = await offlineCtx.startRendering();
    return renderedBuffer.getChannelData(0);
  } else {
    return channelData;
  }
}
