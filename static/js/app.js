
let mediaRecorder;
let recordedChunks = [];
const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');
const btnDetect = document.getElementById('btnDetect');
const fileInput = document.getElementById('fileInput');
const resultBox = document.getElementById('resultBox');
const player = document.getElementById('player');
const downloadLink = document.getElementById('downloadLink');

function setResult(message, level) {
  resultBox.className = 'alert';
  if (level === 'high') resultBox.classList.add('alert-danger');
  else if (level === 'medium') resultBox.classList.add('alert-warning');
  else if (level === 'none') resultBox.classList.add('alert-success');
  else resultBox.classList.add('alert-info');
  resultBox.textContent = message;
}

btnStart?.addEventListener('click', async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.onstop = () => {
      const blob = new Blob(recordedChunks, { type: 'audio/webm' });
      const url = URL.createObjectURL(blob);
      player.src = url;
      player.classList.remove('d-none');
      downloadLink.href = url;
      downloadLink.classList.remove('d-none');
    };
    mediaRecorder.start();
    btnStart.disabled = true;
    btnStop.disabled = false;
    setResult('Recording... speak now!', null);
  } catch (e) {
    setResult('Microphone access denied or unavailable.', null);
    console.error(e);
  }
});

btnStop?.addEventListener('click', () => {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  btnStart.disabled = false;
  btnStop.disabled = true;
});

btnDetect?.addEventListener('click', async () => {
  setResult('Uploading and detecting...', null);
  const form = new FormData();
  if (recordedChunks.length > 0) {
    // Convert webm blob to file
    const blob = new Blob(recordedChunks, { type: 'audio/webm' });
    form.append('audio', blob, 'recording.webm');
  } else if (fileInput.files.length > 0) {
    form.append('audio', fileInput.files[0], fileInput.files[0].name);
  } else {
    setResult('Please record or choose an audio file first.', null);
    return;
  }

  try {
    const res = await fetch('/detect', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Detection failed');

    document.getElementById('svmProb').textContent = data.svm_prob.toFixed(3);
    document.getElementById('mlpProb').textContent = data.mlp_prob.toFixed(3);

    if (data.risk === 'High Risk') setResult('High Risk: both models predict positive.', 'high');
    else if (data.risk === 'Medium Risk') setResult('Medium Risk: one model predicts positive.', 'medium');
    else setResult('No Risk: neither model predicts positive.', 'none');
  } catch (e) {
    console.error(e);
    setResult('Error: ' + e.message, null);
  }
});
