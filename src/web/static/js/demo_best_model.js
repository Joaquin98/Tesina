async function getJSON(url, options = {}) {
  const res = await fetch(url, options);
  const data = await res.json();
  if (!res.ok || data.success === false) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data;
}

const ui = {
  fixedModelPill: document.getElementById('fixedModelPill'),
  healthPill: document.getElementById('healthPill'),
  videoSelect: document.getElementById('videoSelect'),
  confidenceInput: document.getElementById('confidenceInput'),
  analyzeBtn: document.getElementById('analyzeBtn'),
  statusText: document.getElementById('statusText'),
  summaryBox: document.getElementById('summaryBox'),
  timeline: document.getElementById('timeline'),
  detectionsBody: document.getElementById('detectionsBody'),
  mainVideo: document.getElementById('mainVideo'),
  videoInfo: document.getElementById('videoInfo'),
  mergeGapInput: document.getElementById('mergeGapInput'),
  remergeBtn: document.getElementById('remergeBtn'),
  timelineTrack: document.getElementById('timelineTrack'),
  timelineInfo: document.getElementById('timelineInfo'),
  mergedChips: document.getElementById('mergedChips'),
  exportZipBtn: document.getElementById('exportZipBtn'),
  exportStatus: document.getElementById('exportStatus'),
};

const state = {
  analysisId: null,
  videoPath: null,
  videoUrl: null,
  rawDetections: [],
  mergedDetections: [],
  videoDuration: 0,
};

const TARGET_PLAY_TYPES = new Set(['line', 'scrum']);

function filterTargetDetections(detections = []) {
  return detections.filter(d => TARGET_PLAY_TYPES.has((d.play_type || '').toLowerCase()));
}

function setStatus(text, cls = '') {
  ui.statusText.className = `status ${cls}`.trim();
  ui.statusText.textContent = text;
}

function formatVideoDuration(duration) {
  const totalSeconds = Math.max(0, Math.round(Number(duration) || 0));
  if (!totalSeconds) return 'no disponible';
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return hours > 0
    ? `${hours}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
    : `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

function renderVideos(videos) {
  ui.videoSelect.innerHTML = '';
  if (!videos.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No hay videos disponibles';
    ui.videoSelect.appendChild(opt);
    ui.analyzeBtn.disabled = true;
    return;
  }

  videos.forEach(v => {
    const opt = document.createElement('option');
    opt.value = v.path;
    opt.textContent = `${v.name} · Duración: ${formatVideoDuration(v.duration)}`;
    ui.videoSelect.appendChild(opt);
  });
  ui.analyzeBtn.disabled = false;

  if (videos[0]) {
    state.videoPath = videos[0].path;
    attachSelectedVideo(videos[0].path);
  }
}

function renderSummary(results, detections = state.rawDetections) {
  const summary = results.summary || {};
  const duration = Number(results?.metadata?.duration || 0);
  state.videoDuration = duration;

  const playsByType = detections.reduce((acc, d) => {
    const key = (d.play_type || '').toLowerCase();
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
  const averageConfidence = detections.length
    ? detections.reduce((acc, d) => acc + Number(d.confidence || 0), 0) / detections.length
    : 0;

  ui.summaryBox.innerHTML = `
    <div><strong>Total detecciones (line/scrum):</strong> ${detections.length}</div>
    <div><strong>Bloques fusionados:</strong> ${state.mergedDetections.length}</div>
    <div><strong>Confianza promedio:</strong> ${(averageConfidence * 100).toFixed(1)}%</div>
    <div><strong>Modo:</strong> ${results.metadata?.analysis_mode || 'segments'}</div>
    <div><strong>Duración video:</strong> ${duration ? `${duration.toFixed(1)}s` : 'n/d'}</div>
    <div><strong>Distribución:</strong> ${JSON.stringify(playsByType)}</div>
    <div class="tiny">Nota: la clase "other" está oculta en esta demo.</div>
  `;
}

function renderTimeline(detections) {
  if (!detections || detections.length === 0) {
    ui.timeline.textContent = 'Sin detecciones de line/scrum para este video.';
    return;
  }

  ui.timeline.innerHTML = detections
    .slice(0, 25)
    .map((d, i) => `
      <div class="item">
        <strong>#${i + 1} ${d.play_type}</strong>
        — ${(d.confidence * 100).toFixed(1)}%
        — [${(d.start_time || 0).toFixed(1)}s → ${(d.end_time || 0).toFixed(1)}s]
      </div>
    `)
    .join('');
}

function mergeContiguousDetections(detections, gapSeconds = 0.2) {
  if (!detections || detections.length === 0) return [];

  const sorted = detections
    .map((d, idx) => ({ ...d, _sourceIndex: idx }))
    .sort((a, b) => (a.start_time || 0) - (b.start_time || 0));

  const merged = [];
  let current = null;

  for (const d of sorted) {
    const candidate = {
      play_type: d.play_type || 'other',
      start_time: Number(d.start_time || 0),
      end_time: Number(d.end_time || 0),
      confidence: Number(d.confidence || 0),
      source_indices: [d._sourceIndex],
    };

    if (!current) {
      current = candidate;
      continue;
    }

    const sameType = candidate.play_type === current.play_type;
    const contiguous = candidate.start_time <= current.end_time + gapSeconds;

    if (sameType && contiguous) {
      current.end_time = Math.max(current.end_time, candidate.end_time);
      current.confidence = Math.max(current.confidence, candidate.confidence);
      current.source_indices.push(...candidate.source_indices);
    } else {
      current.duration = current.end_time - current.start_time;
      merged.push(current);
      current = candidate;
    }
  }

  if (current) {
    current.duration = current.end_time - current.start_time;
    merged.push(current);
  }

  return merged;
}

function getPlaybackDuration() {
  const fromVideo = Number(ui.mainVideo?.duration || 0);
  if (Number.isFinite(fromVideo) && fromVideo > 0) return fromVideo;
  if (state.videoDuration > 0) return state.videoDuration;
  const maxDetectionEnd = state.rawDetections.reduce((acc, d) => Math.max(acc, Number(d.end_time || 0)), 0);
  return maxDetectionEnd;
}

function renderMergedTimeline() {
  const merged = state.mergedDetections || [];
  const duration = getPlaybackDuration();

  ui.timelineTrack.innerHTML = '';
  ui.mergedChips.innerHTML = '';

  if (!merged.length || duration <= 0) {
    ui.timelineInfo.textContent = 'Sin bloques fusionados aún.';
    ui.exportZipBtn.disabled = true;
    return;
  }

  merged.forEach((d, idx) => {
    const start = Number(d.start_time || 0);
    const end = Number(d.end_time || 0);
    const left = Math.max(0, (start / duration) * 100);
    const width = Math.max(0.8, ((end - start) / duration) * 100);

    const seg = document.createElement('div');
    seg.className = `timeline-seg ${d.play_type || 'other'}`;
    seg.style.left = `${left}%`;
    seg.style.width = `${Math.min(width, 100 - left)}%`;
    seg.title = `#${idx + 1} ${d.play_type} [${start.toFixed(1)}s-${end.toFixed(1)}s]`;
    seg.addEventListener('click', () => seekToSegment(idx));
    ui.timelineTrack.appendChild(seg);

    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'seg-chip';
    chip.textContent = `#${idx + 1} ${d.play_type} (${start.toFixed(0)}-${end.toFixed(0)}s)`;
    chip.addEventListener('click', () => seekToSegment(idx));
    ui.mergedChips.appendChild(chip);
  });

  ui.timelineInfo.textContent = `Bloques fusionados: ${merged.length} | Duración referencia: ${duration.toFixed(1)}s`;
  ui.exportZipBtn.disabled = !state.analysisId;
}

function seekToSegment(index) {
  const seg = state.mergedDetections[index];
  if (!seg || !ui.mainVideo) return;

  ui.mainVideo.currentTime = Number(seg.start_time || 0);
  ui.mainVideo.play().catch(() => {});
}

function recomputeMergedDetections() {
  const gap = Number(ui.mergeGapInput.value || 0.2);
  state.mergedDetections = mergeContiguousDetections(state.rawDetections, gap);
  renderMergedTimeline();

  if (state.rawDetections.length) {
    renderSummary({
      metadata: { duration: state.videoDuration, analysis_mode: 'segments' },
      summary: {},
      statistics: {},
    }, state.rawDetections);
  } else {
    renderSummary({
      metadata: { duration: state.videoDuration, analysis_mode: 'segments' },
      summary: {},
      statistics: {},
    }, []);
  }
}

function attachSelectedVideo(videoPath) {
  state.videoPath = videoPath;
  if (!videoPath) {
    ui.mainVideo.removeAttribute('src');
    ui.videoInfo.textContent = 'Seleccioná un video para previsualizarlo.';
    return;
  }

  const src = `/api/demo/video?video_path=${encodeURIComponent(videoPath)}`;
  state.videoUrl = src;
  ui.mainVideo.src = src;
  ui.videoInfo.textContent = 'Video listo. Podés reproducirlo y saltar desde los bloques.';
}

function renderTable(detections) {
  if (!detections || detections.length === 0) {
    ui.detectionsBody.innerHTML = '<tr><td colspan="5" class="muted">Sin detecciones.</td></tr>';
    return;
  }

  ui.detectionsBody.innerHTML = detections
    .map((d, i) => `
      <tr>
        <td>${i + 1}</td>
        <td>${d.play_type}</td>
        <td>${(d.confidence * 100).toFixed(1)}%</td>
        <td>${(d.start_time || 0).toFixed(2)}</td>
        <td>${(d.end_time || 0).toFixed(2)}</td>
      </tr>
    `)
    .join('');
}

async function init() {
  try {
    setStatus('Verificando modelo...');
    const health = await getJSON('/api/demo/health');
    ui.fixedModelPill.textContent = `Modelo fijo: ${health.fixed_model}`;
    ui.healthPill.textContent = `Estado: listo`;
    ui.healthPill.classList.add('ok');

    setStatus('Cargando biblioteca de videos...');
    const videosResp = await getJSON('/api/demo/videos');
    renderVideos(videosResp.videos || []);
    ui.videoSelect.addEventListener('change', () => {
      attachSelectedVideo(ui.videoSelect.value);
    });
    setStatus('Listo para análisis en vivo.', 'ok');
  } catch (err) {
    console.error(err);
    ui.healthPill.textContent = 'Estado: error';
    ui.healthPill.classList.add('error');
    setStatus(`Error inicial: ${err.message}`, 'error');
  }
}

ui.analyzeBtn.addEventListener('click', async () => {
  const videoPath = ui.videoSelect.value;
  const confidence = Number(ui.confidenceInput.value || 0.5);

  if (!videoPath) {
    setStatus('Seleccioná un video primero.', 'warn');
    return;
  }

  try {
    ui.analyzeBtn.disabled = true;
    setStatus('Analizando... (puede tardar unos segundos)', 'warn');

    const payload = {
      video_path: videoPath,
      confidence_threshold: confidence,
      segment_duration: 7,
    };

    const resp = await getJSON('/api/demo/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const results = resp.results || {};
  const detections = results.detections || [];
  const targetDetections = filterTargetDetections(detections);
    state.analysisId = resp.analysis_id || null;
  state.rawDetections = targetDetections;
    state.videoDuration = Number(results?.metadata?.duration || 0);

    if (resp.video_url) {
      state.videoUrl = resp.video_url;
      ui.mainVideo.src = resp.video_url;
    }

    recomputeMergedDetections();

  renderSummary(results, targetDetections);
  renderTimeline(targetDetections);
  renderTable(targetDetections);
    ui.exportStatus.textContent = 'Listo para exportar ZIP.';
  setStatus(`Análisis completado. Detecciones line/scrum: ${targetDetections.length}`, 'ok');
  } catch (err) {
    console.error(err);
    setStatus(`Error en análisis: ${err.message}`, 'error');
  } finally {
    ui.analyzeBtn.disabled = false;
  }
});

ui.remergeBtn.addEventListener('click', () => {
  recomputeMergedDetections();
  setStatus(`Bloques recalculados: ${state.mergedDetections.length}`, 'ok');
});

ui.exportZipBtn.addEventListener('click', async () => {
  if (!state.analysisId) {
    setStatus('Primero ejecutá un análisis.', 'warn');
    return;
  }

  try {
    ui.exportZipBtn.disabled = true;
    ui.exportStatus.textContent = 'Generando ZIP...';

    const mergeGap = Number(ui.mergeGapInput.value || 0.2);
    const res = await fetch('/api/demo/export-clips', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        analysis_id: state.analysisId,
        merge_gap_seconds: mergeGap,
      }),
    });

    if (!res.ok) {
      let message = `HTTP ${res.status}`;
      try {
        const err = await res.json();
        message = err.error || message;
      } catch (_) {
        // ignore parse error
      }
      throw new Error(message);
    }

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `demo_merged_clips_${state.analysisId}.zip`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);

    ui.exportStatus.textContent = `ZIP listo: ${state.mergedDetections.length} bloque(s) fusionado(s).`;
    setStatus('Exportación completada.', 'ok');
  } catch (err) {
    console.error(err);
    ui.exportStatus.textContent = `Error: ${err.message}`;
    setStatus(`Error en exportación: ${err.message}`, 'error');
  } finally {
    ui.exportZipBtn.disabled = false;
  }
});

init();
