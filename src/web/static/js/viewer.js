/**
 * viewer.js — Timeline viewer for a single rugby play detection result.
 *
 * Depends on:
 *   - RESULT_ID   (injected by Flask as a global const in viewer.html)
 *   - /api/result/<id>              → { success, results }
 *   - /api/video-for-result/<id>    → video file (HTTP range supported)
 */

'use strict';

/* ═══════════════════════════════════════════════════════════════════
   Colour palette — must match CSS variables in style.css
   ═══════════════════════════════════════════════════════════════════ */
const TYPE_COLORS = {
    scrum: '#ef4444',
    line:  '#10b981',
    other: '#3b82f6',
};

/* ═══════════════════════════════════════════════════════════════════
   State
   ═══════════════════════════════════════════════════════════════════ */
let allDetections  = [];   // full list from API
let videoDuration  = 0;    // seconds (from video metadata)
let activeFilters  = new Set(['scrum', 'line', 'other']);
let minConf        = 0.70;
let animFrameId    = null;
let videoSourceMode = 'source';
let modelRoiController = null;
let isSamplingComparison = false;

/* ═══════════════════════════════════════════════════════════════════
   Helpers
   ═══════════════════════════════════════════════════════════════════ */
function formatTime(sec) {
    if (!isFinite(sec) || sec < 0) return '0:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${String(s).padStart(2, '0')}`;
}

function pct(value) {
    return (value * 100).toFixed(0) + '%';
}

function playTypeLabel(type) {
    return type === 'other' ? 'Otro' : type.charAt(0).toUpperCase() + type.slice(1);
}

function friendlyViewerModelName(name) {
    if (name === 'best_model/best_model.keras' || /InceptionV3-GRU/i.test(name || '')) {
        return 'ResNet-50 + BiGRU';
    }
    return name || '—';
}

function samplingComparisonMarkup(comparison) {
    if (!comparison?.rows?.length) return '';
    const escape = value => String(value ?? '').replace(/[&<>"']/g, ch =>
        ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    const metric = (hits, total) => Number.isFinite(hits) && total > 0
        ? `${(100 * hits / total).toLocaleString('es-AR', {minimumFractionDigits:1, maximumFractionDigits:1})} % (${hits}/${total})`
        : '—';
    const rows = comparison.rows.map(row => {
        const cells = ['line', 'scrum', 'total'].map(type => {
            const precision = row.precision?.[type] || {};
            const recall = row.recall?.[type] || {};
            return `<td>${metric(precision.correct, precision.candidates)}</td><td>${metric(recall.detected, recall.events)}</td>`;
        }).join('');
        const seconds = Math.round(Number(row.runtime_s));
        const time = Number.isFinite(seconds)
            ? `${Math.floor(seconds / 3600)} h ${Math.floor(seconds / 60) % 60} min ${seconds % 60} s`
            : '—';
        const current = row.result_id === RESULT_ID;
        return `<tr${current ? ' class="sampling-current"' : ''}><th scope="row"><a href="/viewer?result_id=${encodeURIComponent(row.result_id)}"${current ? ' aria-current="page"' : ''}>${escape(row.label)}</a>${current ? ' · viendo ahora' : ''}</th><td>${escape(row.candidates)}</td>${cells}<td>${time}<br><small>${row.runtime_kind === 'measured' ? 'Medido · histórico' : 'Estimado · desde video'}</small></td></tr>`;
    }).join('');
    return `<h3>${escape(comparison.title)}</h3><p>${escape(comparison.note)}</p><div class="sampling-table-scroll"><table><caption>Mismo partido y modelo; cambia el muestreo de YOLO y ResNet. Las fracciones muestran aciertos/candidatos o jugadas detectadas/inventario.</caption><thead><tr><th scope="col">Versión · abrir detecciones</th><th scope="col">Candidatos</th><th scope="col">Precisión line</th><th scope="col">Sensibilidad line</th><th scope="col">Precisión scrum</th><th scope="col">Sensibilidad scrum</th><th scope="col">Precisión total</th><th scope="col">Sensibilidad total</th><th scope="col">Procesamiento completo</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function renderSamplingComparison(results) {
    const comparison = results.sampling_comparison;
    const markup = samplingComparisonMarkup(comparison);
    if (!markup) return;
    const container = document.querySelector('.viewer-container');
    const section = document.createElement('section');
    section.className = 'sampling-comparison';
    section.setAttribute('aria-label', 'Comparación de muestreo');
    section.innerHTML = markup + reviewWorkloadMarkup(comparison.review_rows, comparison.active_review_id);
    container.classList.add('has-sampling-comparison');
    container.insertBefore(section, container.querySelector('.timeline-section'));
}

function reviewWorkloadMarkup(rows, activeId) {
    if (!rows?.length) return '';
    const escape = value => String(value ?? '').replace(/[&<>"']/g, ch =>
        ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    const duration = value => {
        if (!Number.isFinite(value)) return '—';
        const seconds = Math.round(value);
        return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
    };
    const body = rows.map(row => `<tr${row.id === activeId ? ' class="sampling-current"' : ''}><th scope="row">${escape(row.label)}</th><td>${duration(row.review_video_s)}</td><td>${duration(row.correct_candidate_union_s)}</td><td>${duration(row.extra_review_outside_correct_s)}</td><td>${escape(row.detected_unique_events)}/26</td></tr>`).join('');
    return `<h3 style="margin-top:1.5rem">Tiempo de video a revisar</h3><p>Duraciones a velocidad 1×, sin contar dos veces los intervalos superpuestos. Se recorta desde el inicio del partido. Sin margen adicional, clics, pausas ni tiempo de decisión: no es una medición de trabajo humano.</p><div class="sampling-table-scroll sampling-review-table"><table><caption>Formato minutos:segundos. Extra = tiempo de candidatos incorrectos o dudosos fuera de clips acertados. Las variantes dejan distintas jugadas sin detectar.</caption><thead><tr><th scope="col">Muestreo · uno de cada N</th><th scope="col">Todos los candidatos</th><th scope="col">Clips acertados</th><th scope="col">Revisión extra</th><th scope="col">Jugadas únicas detectadas</th></tr></thead><tbody>${body}</tbody></table></div><p>Revisar sólo los candidatos no garantiza encontrar todas las jugadas. Para buscar las omitidas también hay que recorrer el video sin marcar. Los clips acertados incluyen repeticiones y rearmados; su duración no equivale al tiempo exacto de ejecución de la jugada.</p>`;
}

function getFiltered() {
    return allDetections.filter(d =>
        activeFilters.has(d.play_type) && d.confidence >= minConf
    );
}

/* ═══════════════════════════════════════════════════════════════════
   Canvas timeline
   ═══════════════════════════════════════════════════════════════════ */
const canvas = document.getElementById('timeline-canvas');
const ctx    = canvas ? canvas.getContext('2d') : null;

// Polyfill for CanvasRenderingContext2D.roundRect (Chrome < 99, older browsers)
if (ctx && !ctx.roundRect) {
    CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {
        const radius = Math.min(r, w / 2, h / 2);
        this.beginPath();
        this.moveTo(x + radius, y);
        this.lineTo(x + w - radius, y);
        this.arcTo(x + w, y, x + w, y + radius, radius);
        this.lineTo(x + w, y + h - radius);
        this.arcTo(x + w, y + h, x + w - radius, y + h, radius);
        this.lineTo(x + radius, y + h);
        this.arcTo(x, y + h, x, y + h - radius, radius);
        this.lineTo(x, y + radius);
        this.arcTo(x, y, x + radius, y, radius);
        this.closePath();
    };
}

function resizeCanvas() {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    canvas.width  = Math.round(rect.width  * window.devicePixelRatio);
    canvas.height = Math.round(rect.height * window.devicePixelRatio);
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    drawTimeline();
}

/**
 * Draw the timeline bar onto the canvas.
 * @param {Array}  filtered  – filtered detections to draw
 * @param {number} currentT  – current video time in seconds (for scrubber)
 */
function drawTimeline(filtered, currentT) {
    if (!ctx) return;

    // Logical size (CSS pixels)
    const W = canvas.offsetWidth  || canvas.width;
    const H = canvas.offsetHeight || canvas.height;

    ctx.clearRect(0, 0, W, H);

    // Background rail
    ctx.fillStyle = 'rgba(0,0,0,0.06)';
    const railY = 8, railH = H - 16;
    ctx.beginPath();
    ctx.roundRect(0, railY, W, railH, 5);
    ctx.fill();

    if (videoDuration <= 0) return;

    // Detection segments
    const dets = filtered !== undefined ? filtered : getFiltered();
    dets.forEach(d => {
        const x = (d.start_time / videoDuration) * W;
        const w = Math.max(2, ((d.end_time - d.start_time) / videoDuration) * W);
        ctx.fillStyle = TYPE_COLORS[d.play_type] || '#aaa';
        ctx.globalAlpha = 0.5 + d.confidence * 0.45;  // opacity encodes confidence
        ctx.beginPath();
        ctx.roundRect(x, railY, w, railH, 3);
        ctx.fill();
    });
    ctx.globalAlpha = 1;

    // Scrubber line
    const t = currentT !== undefined ? currentT : (document.getElementById('video-player')?.currentTime || 0);
    const sx = (t / videoDuration) * W;
    ctx.fillStyle = '#1e293b';
    ctx.beginPath();
    ctx.roundRect(sx - 1.5, railY - 3, 3, railH + 6, 2);
    ctx.fill();
    canvas.setAttribute('aria-valuemax', String(Math.max(0, Math.round(videoDuration))));
    canvas.setAttribute('aria-valuenow', String(Math.max(0, Math.round(t))));
    canvas.setAttribute('aria-valuetext', formatTime(t));
}

/* ═══════════════════════════════════════════════════════════════════
   Detection cards
   ═══════════════════════════════════════════════════════════════════ */
function renderCards(filtered) {
    const container = document.getElementById('detection-cards');
    if (!container) return;

    document.getElementById('detection-count').textContent = filtered.length;

    if (filtered.length === 0) {
        container.innerHTML = '<div class="no-detections">No hay jugadas que coincidan con los filtros.</div>';
        return;
    }

    container.innerHTML = filtered.map((d, i) => {
        const color  = TYPE_COLORS[d.play_type] || '#aaa';
        const confPct = (d.confidence * 100).toFixed(1);
        const allProbs = d.probabilities
            ? Object.entries(d.probabilities)
                  .sort((a, b) => b[1] - a[1])
                  .map(([k, v]) => `${playTypeLabel(k)}: ${(v * 100).toFixed(1)}%`)
                  .join(' · ')
            : '';

        return `
        <div class="detection-card ${d.play_type}" data-index="${i}">
            <div class="card-badge">
                <span class="badge ${d.play_type}">${playTypeLabel(d.play_type).toUpperCase()}</span>
            </div>
            <div class="card-info">
                <div class="card-time">${formatTime(d.start_time)} – ${formatTime(d.end_time)} <small style="font-weight:400;color:var(--text-secondary)">(${d.duration.toFixed(1)}s)</small></div>
                <div class="card-meta">${allProbs}</div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill ${d.play_type}" style="width:${confPct}%"></div>
                    </div>
                    <span class="conf-value">${confPct}%</span>
                </div>
            </div>
            <div class="card-actions">
                <button class="jump-btn" data-time="${d.start_time}" aria-label="Ir a ${formatTime(d.start_time)}">▶ Ir</button>
            </div>
        </div>`;
    }).join('');

    // Jump buttons
    container.querySelectorAll('.jump-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const video = document.getElementById('video-player');
            const t     = parseFloat(btn.dataset.time);
            if (video && isFinite(t)) {
                video.currentTime = t;
                video.play();
            }
        });
    });

    // Click on card → seek
    container.querySelectorAll('.detection-card').forEach(card => {
        card.addEventListener('click', e => {
            if (e.target.classList.contains('jump-btn')) return;
            const idx = parseInt(card.dataset.index, 10);
            const d   = filtered[idx];
            if (!d) return;
            const video = document.getElementById('video-player');
            if (video) {
                video.currentTime = d.start_time;
                video.play();
                card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        });
    });
}

/* ═══════════════════════════════════════════════════════════════════
   Statistics panel
   ═══════════════════════════════════════════════════════════════════ */
function renderStats(results) {
    const container = document.getElementById('stats-content');
    if (!container) return;

    const stats     = results.statistics || {};
    const byType    = stats.by_type || {};
    const summary   = results.summary || {};
    const meta      = results.metadata || {};
    const total     = summary.total_detections ?? allDetections.length;
    const avgConf   = stats.avg_confidence != null
                          ? (stats.avg_confidence * 100).toFixed(1) + '%'
                          : '—';
    const dur       = meta.duration != null
                          ? formatTime(meta.duration)
                          : '—';

    const typeCards = Object.entries(byType)
        .filter(([type]) => type !== 'other')
        .map(([type, info]) => `
        <div class="stat-card ${type}">
            <div class="stat-value">${info.count ?? 0}</div>
            <div class="stat-label">${playTypeLabel(type)}</div>
        </div>
    `).join('');

    container.innerHTML = `
        <div class="stat-card">
            <div class="stat-value">${total}</div>
            <div class="stat-label">Jugadas detectadas</div>
        </div>
        ${typeCards}
        <div class="stat-card">
            <div class="stat-value">${avgConf}</div>
            <div class="stat-label">Confianza media</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${dur}</div>
            <div class="stat-label">Duración del video</div>
        </div>
        <div class="stat-card">
            <div class="stat-value stat-model-name">${friendlyViewerModelName(results.model_name)}</div>
            <div class="stat-label">Modelo</div>
        </div>
    `;
}

/* ═══════════════════════════════════════════════════════════════════
   Render (canvas + cards) on filter change
   ═══════════════════════════════════════════════════════════════════ */
function render() {
    const filtered = getFiltered();
    renderCards(filtered);
    drawTimeline(filtered);
}

/* ═══════════════════════════════════════════════════════════════════
   Video animation loop  (keeps scrubber in sync)
   ═══════════════════════════════════════════════════════════════════ */
function startAnimLoop(video) {
    function frame() {
        drawTimeline(getFiltered(), video.currentTime);
        animFrameId = requestAnimationFrame(frame);
    }
    if (animFrameId) cancelAnimationFrame(animFrameId);
    animFrameId = requestAnimationFrame(frame);
}

/* ═══════════════════════════════════════════════════════════════════
   Timeline tooltip
   ═══════════════════════════════════════════════════════════════════ */
function setupTimelineTooltip() {
    if (!canvas) return;

    // Create tooltip element
    const tip = document.createElement('div');
    tip.className = 'timeline-tooltip';
    canvas.parentElement.style.position = 'relative';
    canvas.parentElement.appendChild(tip);

    canvas.addEventListener('mousemove', e => {
        const ratio = e.offsetX / canvas.offsetWidth;
        const t     = ratio * videoDuration;
        tip.textContent = formatTime(t);
        tip.style.left  = e.offsetX + 'px';
        tip.style.display = 'block';
    });

    canvas.addEventListener('mouseleave', () => {
        tip.style.display = 'none';
    });
}

/* ═══════════════════════════════════════════════════════════════════
   Main init
   ═══════════════════════════════════════════════════════════════════ */
async function setVideoSource(video, mode) {
    const previousTime = Number.isFinite(video.currentTime) ? video.currentTime : 0;
    videoSourceMode = mode;
    modelRoiController?.setAvailable(mode === 'source');
    const status = document.getElementById('video-source-status');
    const sourceUrl = mode === 'preprocessed'
        ? `/api/preprocessed-video/${RESULT_ID}`
        : `/api/video-for-result/${RESULT_ID}`;

    if (status) {
        status.textContent = mode === 'preprocessed'
            ? (isSamplingComparison ? 'Preparando recorte espacial 224×224...' : 'Preparando video 224×224...')
            : 'Video original';
    }
    if (mode === 'preprocessed') {
        try {
            const probe = await fetch(sourceUrl, { headers: { Range: 'bytes=0-0' } });
            if (!probe.ok) {
                let message = 'No se pudo generar el video usado por la inferencia.';
                try {
                    const payload = await probe.json();
                    if (payload.error) message = payload.error;
                } catch (_) { /* keep the readable fallback */ }
                if (status) status.textContent = message;
                return;
            }
        } catch (err) {
            if (status) status.textContent = `No se pudo preparar el video: ${err.message}`;
            return;
        }
    }
    video.pause();
    video.src = sourceUrl;
    video.load();
    video.addEventListener('loadedmetadata', () => {
        if (previousTime > 0 && isFinite(video.duration)) {
            video.currentTime = Math.min(previousTime, video.duration);
        }
        if (status && mode === 'preprocessed') {
            status.textContent = isSamplingComparison
                ? 'Recorte espacial: este video no representa los fotogramas omitidos ni la interpolación'
                : 'Video preprocesado usado por la inferencia';
        }
    }, {once: true});
}

async function init() {
    /* 1. Fetch result data */
    let results;
    try {
        const resp = await fetch(`/api/result/${RESULT_ID}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        if (!data.success) throw new Error(data.error || 'API error');
        results = data.results;
    } catch (err) {
        console.error('No se pudo cargar el resultado:', err);
        document.querySelector('.viewer-container').innerHTML =
            `<div style="padding:2rem;color:var(--danger-color)">
                ⚠️ No se pudo cargar el resultado <strong>${RESULT_ID}</strong>: ${err.message}
             </div>`;
        return;
    }

    allDetections = results.detections || [];
    isSamplingComparison = Boolean(results.sampling_comparison);
    const savedThreshold = Number(results.metadata?.confidence_threshold ?? 0.70);
    minConf = Number.isFinite(savedThreshold) ? Math.min(1, Math.max(0, savedThreshold)) : 0.70;
    const minConfidenceSlider = document.getElementById('min-confidence');
    minConfidenceSlider.value = String(minConf);
    document.getElementById('min-conf-value').textContent = minConf.toFixed(2);

    /* 2. Video player */
    const video = document.getElementById('video-player');
    const sourceMode = document.getElementById('video-source-mode');
    if (isSamplingComparison && sourceMode) {
        sourceMode.querySelector('option[value="preprocessed"]').textContent = 'Recorte espacial (224×224; sin mostrar el muestreo temporal)';
    }
    modelRoiController = window.ModelRoiOverlay?.attach({
        video,
        stage: document.getElementById('viewer-roi-stage'),
        overlay: document.getElementById('viewer-model-roi'),
        toggle: document.getElementById('show-model-roi'),
        inputMode: /highres|alta resoluci[oó]n/i.test(results.model_name || '')
            ? 'source_equivalent_roi'
            : 'rgb_224',
    });
    setVideoSource(video, 'source');

    video.addEventListener('loadedmetadata', () => {
        videoDuration = video.duration;
        document.getElementById('duration').textContent = formatTime(videoDuration);
        const status = document.getElementById('video-source-status');
        if (status && videoSourceMode === 'source') status.textContent = 'Video original';
        resizeCanvas();  // now we know the canvas is visible, set pixel ratio
        render();
    });

    video.addEventListener('error', () => {
        const status = document.getElementById('video-source-status');
        if (status && videoSourceMode === 'preprocessed') {
            status.textContent = 'No se pudo generar el video preprocesado';
        }
    });

    sourceMode.addEventListener('change', () => {
        setVideoSource(video, sourceMode.value);
    });

    video.addEventListener('timeupdate', () => {
        const seek = document.getElementById('video-seek');
        document.getElementById('current-time').textContent = formatTime(video.currentTime);
        if (videoDuration > 0) {
            seek.value = (video.currentTime / videoDuration) * 100;
        }
    });

    video.addEventListener('play',  () => startAnimLoop(video));
    video.addEventListener('pause', () => { if (animFrameId) cancelAnimationFrame(animFrameId); });
    video.addEventListener('ended', () => { if (animFrameId) cancelAnimationFrame(animFrameId); });

    /* 3. Video controls */
    document.getElementById('play-pause').addEventListener('click', () => {
        video.paused ? video.play() : video.pause();
    });

    document.getElementById('video-seek').addEventListener('input', e => {
        if (videoDuration > 0) {
            video.currentTime = (parseFloat(e.target.value) / 100) * videoDuration;
            drawTimeline(getFiltered(), video.currentTime);
        }
    });

    document.getElementById('fullscreen').addEventListener('click', () => {
        const stage = document.getElementById('viewer-roi-stage');
        if (stage.requestFullscreen) stage.requestFullscreen();
        else if (stage.webkitRequestFullscreen) stage.webkitRequestFullscreen();
    });

    /* 4. Filter checkboxes */
    document.querySelectorAll('.filter-checkbox').forEach(cb => {
        cb.addEventListener('change', () => {
            if (cb.checked) activeFilters.add(cb.dataset.type);
            else            activeFilters.delete(cb.dataset.type);
            render();
        });
    });

    /* 5. Min-confidence slider */
    const slider = document.getElementById('min-confidence');
    slider.addEventListener('input', e => {
        minConf = parseFloat(e.target.value);
        document.getElementById('min-conf-value').textContent = minConf.toFixed(2);
        render();
    });

    /* 6. Canvas click → seek */
    if (canvas) {
        canvas.addEventListener('click', e => {
            if (videoDuration <= 0) return;
            const ratio = e.offsetX / canvas.offsetWidth;
            video.currentTime = ratio * videoDuration;
            drawTimeline(getFiltered(), video.currentTime);
        });
        canvas.addEventListener('keydown', event => {
            if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key) || videoDuration <= 0) return;
            event.preventDefault();
            if (event.key === 'Home') video.currentTime = 0;
            else if (event.key === 'End') video.currentTime = videoDuration;
            else video.currentTime = Math.min(
                videoDuration,
                Math.max(0, video.currentTime + (event.key === 'ArrowRight' ? 5 : -5)),
            );
            drawTimeline(getFiltered(), video.currentTime);
        });
    }

    /* 7. Resize observer to keep canvas sharp */
    if (window.ResizeObserver && canvas) {
        new ResizeObserver(resizeCanvas).observe(canvas);
    } else {
        window.addEventListener('resize', resizeCanvas);
    }

    /* 8. Tooltip */
    setupTimelineTooltip();

    /* 9. Initial render */
    render();
    renderStats(results);
    renderSamplingComparison(results);
}

document.addEventListener('DOMContentLoaded', init);
