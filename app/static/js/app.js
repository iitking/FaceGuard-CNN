/**
 * FaceGuard-CNN | Frontend Interactive Application Logic
 * Pure vanilla JavaScript with Web Audio API and MediaDevices.
 */

document.addEventListener("DOMContentLoaded", () => {
  // ==========================================
  // 1. DOM Elements
  // ==========================================
  const systemStatus = document.getElementById("systemStatus");

  // Tabs
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  // Upload Scanner Elements
  const dropZone = document.getElementById("dropZone");
  const dropZonePlaceholder = document.getElementById("dropZonePlaceholder");
  const fileInput = document.getElementById("fileInput");
  const previewWrapper = document.getElementById("previewWrapper");
  const previewImage = document.getElementById("previewImage");
  const laserScanner = document.getElementById("laserScanner");
  const autoCropToggle = document.getElementById("autoCropToggle");
  const clearImageBtn = document.getElementById("clearImageBtn");
  const sampleMaskBtn = document.getElementById("sampleMaskBtn");
  const sampleNoMaskBtn = document.getElementById("sampleNoMaskBtn");

  // Result Elements
  const statusBadge = document.getElementById("statusBadge");
  const resultPlaceholder = document.getElementById("resultPlaceholder");
  const resultDetails = document.getElementById("resultDetails");
  const statusHero = document.getElementById("statusHero");
  const heroIcon = document.getElementById("heroIcon");
  const heroLabel = document.getElementById("heroLabel");
  const heroSubtext = document.getElementById("heroSubtext");
  const heroScore = document.getElementById("heroScore");
  const probMaskText = document.getElementById("probMaskText");
  const probMaskBar = document.getElementById("probMaskBar");
  const probNoMaskText = document.getElementById("probNoMaskText");
  const probNoMaskBar = document.getElementById("probNoMaskBar");
  const faceBreakdownContainer = document.getElementById("faceBreakdownContainer");
  const faceCountBadge = document.getElementById("faceCountBadge");
  const faceList = document.getElementById("faceList");
  const annotatedImage = document.getElementById("annotatedImage");
  const downloadAnnotatedBtn = document.getElementById("downloadAnnotatedBtn");
  const latencyText = document.getElementById("latencyText");

  // Webcam Elements
  const startWebcamBtn = document.getElementById("startWebcamBtn");
  const stopWebcamBtn = document.getElementById("stopWebcamBtn");
  const captureWebcamSnapshotBtn = document.getElementById("captureWebcamSnapshotBtn");
  const webcamVideo = document.getElementById("webcamVideo");
  const webcamCanvas = document.getElementById("webcamCanvas");
  const webcamPlaceholder = document.getElementById("webcamPlaceholder");
  const cameraHud = document.getElementById("cameraHud");
  const hudStatusBanner = document.getElementById("hudStatusBanner");
  const hudFps = document.getElementById("hudFps");
  const scanIntervalSelect = document.getElementById("scanIntervalSelect");
  const audioAlertToggle = document.getElementById("audioAlertToggle");
  const countScans = document.getElementById("countScans");
  const countCompliant = document.getElementById("countCompliant");
  const countViolations = document.getElementById("countViolations");
  const livePreviewTarget = document.getElementById("livePreviewTarget");
  const liveDecisionBadge = document.getElementById("liveDecisionBadge");
  const liveComplianceBadge = document.getElementById("liveComplianceBadge");

  // Webcam State Variables
  let webcamStream = null;
  let webcamIntervalTimer = null;
  let isWebcamAnalyzing = false;
  let stats = { scans: 0, compliant: 0, violations: 0 };
  let lastFrameTime = performance.now();
  let frameCount = 0;
  let currentFps = 0;

  // ==========================================
  // 2. Audio Synthesizer (Web Audio API)
  // ==========================================
  let audioCtx = null;

  function initAudio() {
    if (!audioCtx) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (AudioContext) {
        audioCtx = new AudioContext();
      }
    }
  }

  function playAlertSound(isCompliant) {
    if (!audioAlertToggle || !audioAlertToggle.checked) return;
    try {
      initAudio();
      if (!audioCtx || audioCtx.state === "suspended") {
        audioCtx && audioCtx.resume();
      }
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.connect(gain);
      gain.connect(audioCtx.destination);

      if (isCompliant) {
        // High soft chime
        osc.type = "sine";
        osc.frequency.setValueAtTime(660, audioCtx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(880, audioCtx.currentTime + 0.15);
        gain.gain.setValueAtTime(0.12, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.25);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.25);
      } else {
        // Warning double beep
        osc.type = "sawtooth";
        osc.frequency.setValueAtTime(320, audioCtx.currentTime);
        osc.frequency.setValueAtTime(240, audioCtx.currentTime + 0.12);
        gain.gain.setValueAtTime(0.18, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.35);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.35);
      }
    } catch (e) {
      console.warn("Audio feedback error:", e);
    }
  }

  // ==========================================
  // 3. Health Check & Initialization
  // ==========================================
  async function checkSystemHealth() {
    try {
      const res = await fetch("/health");
      if (res.ok) {
        const data = await res.json();
        const dot = systemStatus.querySelector(".status-dot");
        const label = systemStatus.querySelector(".status-label");
        
        if (data.model_loaded) {
          dot.className = "status-dot active";
          label.textContent = "AI Ready | 98.3% Acc";
        } else {
          dot.className = "status-dot pulse";
          label.textContent = "Model Degraded";
        }
      }
    } catch (err) {
      console.warn("Health check unreachable:", err);
      const label = systemStatus.querySelector(".status-label");
      if (label) label.textContent = "Server Offline";
    }
  }
  checkSystemHealth();

  // ==========================================
  // 4. Tab Switching Logic
  // ==========================================
  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      tabButtons.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetTab = document.getElementById(targetId);
      if (targetTab) targetTab.classList.add("active");

      // Auto stop webcam if navigating away
      if (targetId !== "webcamTab" && webcamStream) {
        stopWebcam();
      }
    });
  });

  // ==========================================
  // 5. Image Upload & Prediction Pipeline
  // ==========================================
  function resetUpload() {
    fileInput.value = "";
    previewImage.src = "";
    previewWrapper.classList.add("hidden");
    dropZonePlaceholder.classList.remove("hidden");
    resultPlaceholder.classList.remove("hidden");
    resultDetails.classList.add("hidden");
    statusBadge.textContent = "Awaiting Input";
    statusBadge.className = "badge badge-neutral";
    laserScanner.classList.remove("scanning");
  }

  clearImageBtn.addEventListener("click", resetUpload);

  dropZone.addEventListener("click", (e) => {
    if (e.target.tagName !== "BUTTON" && previewWrapper.classList.contains("hidden")) {
      fileInput.click();
    }
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload a valid image file (.jpg, .png, .webp).");
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      previewImage.src = event.target.result;
      dropZonePlaceholder.classList.add("hidden");
      previewWrapper.classList.remove("hidden");
      executePrediction(file);
    };
    reader.readAsDataURL(file);
  }

  async function executePrediction(file) {
    statusBadge.textContent = "Analyzing Image...";
    statusBadge.className = "badge badge-accent";
    laserScanner.classList.add("scanning");

    const formData = new FormData();
    formData.append("file", file);
    const detectFaces = autoCropToggle ? autoCropToggle.checked : true;

    try {
      const response = await fetch(`/api/predict?detect_faces=${detectFaces}`, {
        method: "POST",
        body: formData
      });

      laserScanner.classList.remove("scanning");

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Inference failed" }));
        throw new Error(errorData.detail || `Server returned ${response.status}`);
      }

      const result = await response.json();
      renderPredictionResult(result);
    } catch (err) {
      laserScanner.classList.remove("scanning");
      statusBadge.textContent = "Error";
      statusBadge.className = "badge badge-danger";
      alert("Prediction Error: " + err.message);
    }
  }

  function renderPredictionResult(data) {
    resultPlaceholder.classList.add("hidden");
    resultDetails.classList.remove("hidden");

    const isWearing = data.is_wearing_mask;
    const confPct = (data.confidence * 100).toFixed(1) + "%";

    // Play feedback tone
    playAlertSound(isWearing);

    // Update Status Hero
    if (isWearing) {
      statusHero.className = "status-hero";
      heroIcon.textContent = "🛡️";
      heroLabel.textContent = "WITH MASK";
      heroSubtext.textContent = data.message || "Individual is safety compliant";
      statusBadge.textContent = "SAFE • COMPLIANT";
      statusBadge.className = "badge badge-success";
    } else {
      statusHero.className = "status-hero violation";
      heroIcon.textContent = "⚠️";
      heroLabel.textContent = "WITHOUT MASK";
      heroSubtext.textContent = data.message || "Face mask violation detected";
      statusBadge.textContent = "ALERT • NO MASK";
      statusBadge.className = "badge badge-danger";
    }

    heroScore.textContent = confPct;

    // Probabilities
    const probWithMask = (data.probabilities["With Mask"] * 100).toFixed(1);
    const probNoMask = (data.probabilities["Without Mask"] * 100).toFixed(1);

    probMaskText.textContent = `${probWithMask}%`;
    probMaskBar.style.width = `${probWithMask}%`;

    probNoMaskText.textContent = `${probNoMask}%`;
    probNoMaskBar.style.width = `${probNoMask}%`;

    // Detected Faces Breakdown
    faceCountBadge.textContent = `${data.face_count} Subject(s)`;
    faceList.innerHTML = "";

    if (data.faces && data.faces.length > 0) {
      data.faces.forEach((f) => {
        const item = document.createElement("div");
        item.className = "face-entry";
        const tag = f.is_wearing_mask 
          ? `<span class="badge badge-success">Mask (${(f.confidence * 100).toFixed(1)}%)</span>`
          : `<span class="badge badge-danger">No Mask (${(f.confidence * 100).toFixed(1)}%)</span>`;
        item.innerHTML = `<span>Face #${f.face_index}</span> ${tag}`;
        faceList.appendChild(item);
      });
    }

    // Annotated Image
    if (data.annotated_image) {
      annotatedImage.src = data.annotated_image;
      downloadAnnotatedBtn.href = data.annotated_image;
    } else {
      annotatedImage.src = previewImage.src;
      downloadAnnotatedBtn.href = previewImage.src;
    }

    // Diagnostics
    latencyText.textContent = `${data.inference_time_ms} ms`;
  }

  // ==========================================
  // 6. Built-in Sample Generator
  // ==========================================
  function createSyntheticSample(isWithMask) {
    const canvas = document.createElement("canvas");
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext("2d");

    // Face background
    ctx.fillStyle = "#fcd34d";
    ctx.beginPath();
    ctx.arc(64, 64, 52, 0, Math.PI * 2);
    ctx.fill();

    // Eyes
    ctx.fillStyle = "#1e293b";
    ctx.beginPath();
    ctx.arc(46, 52, 6, 0, Math.PI * 2);
    ctx.arc(82, 52, 6, 0, Math.PI * 2);
    ctx.fill();

    if (isWithMask) {
      // Surgical Mask (Cyan / Blue mask)
      ctx.fillStyle = "#38bdf8";
      ctx.beginPath();
      ctx.roundRect(34, 66, 60, 42, 8);
      ctx.fill();

      // Mask Straps
      ctx.strokeStyle = "#94a3b8";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(34, 76);
      ctx.lineTo(16, 68);
      ctx.moveTo(94, 76);
      ctx.lineTo(112, 68);
      ctx.stroke();
    } else {
      // Smile
      ctx.strokeStyle = "#dc2626";
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.arc(64, 76, 22, 0.1 * Math.PI, 0.9 * Math.PI, false);
      ctx.stroke();
    }

    canvas.toBlob((blob) => {
      const file = new File([blob], isWithMask ? "sample_mask.png" : "sample_nomask.png", { type: "image/png" });
      handleFileSelected(file);
    }, "image/png");
  }

  sampleMaskBtn.addEventListener("click", () => createSyntheticSample(true));
  sampleNoMaskBtn.addEventListener("click", () => createSyntheticSample(false));

  // ==========================================
  // 7. Live Webcam Processing
  // ==========================================
  async function startWebcam() {
    initAudio();
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user"
        },
        audio: false
      });

      webcamVideo.srcObject = webcamStream;
      webcamPlaceholder.classList.add("hidden");
      cameraHud.classList.remove("hidden");
      startWebcamBtn.classList.add("hidden");
      stopWebcamBtn.classList.remove("hidden");
      captureWebcamSnapshotBtn.classList.remove("hidden");

      // Start FPS counter and continuous frame inference loop
      startInferenceLoop();
    } catch (err) {
      console.error("Camera access denied or unavailable:", err);
      alert("Could not access camera. Please ensure camera permissions are allowed in your browser.");
    }
  }

  function stopWebcam() {
    if (webcamStream) {
      webcamStream.getTracks().forEach(track => track.stop());
      webcamStream = null;
    }
    if (webcamIntervalTimer) {
      clearInterval(webcamIntervalTimer);
      webcamIntervalTimer = null;
    }

    webcamVideo.srcObject = null;
    webcamPlaceholder.classList.remove("hidden");
    cameraHud.classList.add("hidden");
    startWebcamBtn.classList.remove("hidden");
    stopWebcamBtn.classList.add("hidden");
    captureWebcamSnapshotBtn.classList.add("hidden");
    hudStatusBanner.textContent = "Scanner Stopped";
  }

  startWebcamBtn.addEventListener("click", startWebcam);
  stopWebcamBtn.addEventListener("click", stopWebcam);

  function startInferenceLoop() {
    if (webcamIntervalTimer) clearInterval(webcamIntervalTimer);
    const intervalMs = parseInt(scanIntervalSelect.value, 10) || 800;

    webcamIntervalTimer = setInterval(() => {
      if (!isWebcamAnalyzing && webcamStream) {
        processWebcamFrame();
      }
    }, intervalMs);
  }

  scanIntervalSelect.addEventListener("change", startInferenceLoop);

  async function processWebcamFrame() {
    if (!webcamVideo.videoWidth || !webcamVideo.videoHeight) return;

    isWebcamAnalyzing = true;
    webcamCanvas.width = webcamVideo.videoWidth;
    webcamCanvas.height = webcamVideo.videoHeight;
    const ctx = webcamCanvas.getContext("2d");

    // Draw unmirrored frame
    ctx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);
    const base64Data = webcamCanvas.toDataURL("image/jpeg", 0.75);

    try {
      const startTime = performance.now();
      const res = await fetch("/api/predict-base64?detect_faces=true", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64Data })
      });

      const elapsed = Math.round(performance.now() - startTime);

      // FPS tracking
      frameCount++;
      const now = performance.now();
      if (now - lastFrameTime >= 1000) {
        currentFps = frameCount;
        frameCount = 0;
        lastFrameTime = now;
        hudFps.textContent = `${currentFps} FPS (${elapsed}ms)`;
      }

      if (res.ok) {
        const data = await res.json();
        updateWebcamTelemetry(data);
      }
    } catch (e) {
      console.warn("Webcam inference error:", e);
    } finally {
      isWebcamAnalyzing = false;
    }
  }

  function updateWebcamTelemetry(data) {
    stats.scans++;
    countScans.textContent = stats.scans;

    const isWearing = data.is_wearing_mask;
    if (isWearing) {
      stats.compliant++;
      countCompliant.textContent = stats.compliant;
      hudStatusBanner.textContent = `COMPLIANT • MASK DETECTED (${(data.confidence * 100).toFixed(0)}%)`;
      hudStatusBanner.style.color = "var(--success)";
      liveDecisionBadge.textContent = "SAFE • MASK DETECTED";
      liveDecisionBadge.style.color = "var(--success)";
      liveComplianceBadge.textContent = "Compliant";
      liveComplianceBadge.className = "badge badge-success";
    } else {
      stats.violations++;
      countViolations.textContent = stats.violations;
      hudStatusBanner.textContent = `ALERT • NO MASK DETECTED (${(data.confidence * 100).toFixed(0)}%)`;
      hudStatusBanner.style.color = "var(--danger)";
      liveDecisionBadge.textContent = "VIOLATION • NO MASK";
      liveDecisionBadge.style.color = "var(--danger)";
      liveComplianceBadge.textContent = "Violation Alert";
      liveComplianceBadge.className = "badge badge-danger";
    }

    // Play sound alert on state
    playAlertSound(isWearing);

    // Update live snapshot
    if (data.annotated_image) {
      livePreviewTarget.innerHTML = `<img src="${data.annotated_image}" alt="Live Frame">`;
    }
  }

  captureWebcamSnapshotBtn.addEventListener("click", () => {
    if (!webcamVideo.videoWidth) return;
    webcamCanvas.width = webcamVideo.videoWidth;
    webcamCanvas.height = webcamVideo.videoHeight;
    const ctx = webcamCanvas.getContext("2d");
    ctx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);
    
    webcamCanvas.toBlob((blob) => {
      const file = new File([blob], "webcam_snapshot.jpg", { type: "image/jpeg" });
      // Switch to Scanner Tab and display result
      const uploadTabBtn = document.querySelector('[data-tab="uploadTab"]');
      if (uploadTabBtn) uploadTabBtn.click();
      handleFileSelected(file);
    }, "image/jpeg");
  });
});
