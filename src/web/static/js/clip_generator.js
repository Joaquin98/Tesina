const MAX_VISIBLE_CLIP_GENERATOR_VIDEOS = 36;

const clipGeneratorState = {
    videos: [],
    selectedVideo: null,
    session: null,
    currentIndex: 0,
    saving: false,
    analysisComparison: null,
    analysisComparisons: [],
    comparisonVideoPath: '',
    comparisonLoading: false,
    comparisonError: '',
};

document.addEventListener('DOMContentLoaded', async () => {
    bindClipGeneratorEvents();
    await loadClipGeneratorVideos();

    const sessionId = new URLSearchParams(window.location.search).get('session_id');
    if (sessionId) {
        await loadClipGeneratorSession(sessionId);
    }
});

function bindClipGeneratorEvents() {
    document.getElementById('clip-generator-search')?.addEventListener('input', renderClipGeneratorVideos);
    document.getElementById('clip-generator-source-filter')?.addEventListener('change', renderClipGeneratorVideos);
    document.getElementById('clip-generator-label-filter')?.addEventListener('change', renderClipGeneratorVideos);
    document.getElementById('clip-generator-video-list')?.addEventListener('click', event => {
        const button = event.target.closest('[data-video-path]');
        if (button) selectClipGeneratorVideo(button.dataset.videoPath);
    });
    document.getElementById('clip-generator-start')?.addEventListener('click', startClipGeneratorSession);
    document.querySelectorAll('.clip-label-button').forEach(button => {
        button.addEventListener('click', () => saveClipLabel(button.dataset.label));
    });
    document.getElementById('clip-generator-replay')?.addEventListener('click', playCurrentClip);
    document.getElementById('clip-generator-fullscreen')?.addEventListener('click', () => {
        const video = document.getElementById('clip-generator-video');
        if (video?.requestFullscreen) video.requestFullscreen();
        else if (video?.webkitRequestFullscreen) video.webkitRequestFullscreen();
    });
    document.getElementById('clip-generator-prev')?.addEventListener('click', () => moveClipGenerator(-1));
    document.getElementById('clip-generator-next')?.addEventListener('click', moveToNextPendingClip);
    document.getElementById('clip-generator-export')?.addEventListener('click', exportLabeledClips);
    document.getElementById('clip-generator-analysis-select')?.addEventListener('change', event => {
        const selectedResultId = event.currentTarget.value;
        clipGeneratorState.analysisComparison = clipGeneratorState.analysisComparisons.find(
            item => item.result_id === selectedResultId
        ) || null;
        renderClipGeneratorComparison();
    });

    const video = document.getElementById('clip-generator-video');
    video?.addEventListener('timeupdate', stopAtCurrentClipEnd);
    video?.addEventListener('ended', () => setClipVideoMessage('Terminó el fragmento. Elegí una clase o volvé a reproducirlo.'));
    ['clip-generator-prediction-timeline', 'clip-generator-label-timeline'].forEach(id => {
        document.getElementById(id)?.addEventListener('click', jumpToClipFromTimeline);
    });
    window.addEventListener('resize', renderClipGeneratorComparison);
    document.addEventListener('keydown', handleClipGeneratorShortcut);
}

async function loadClipGeneratorVideos() {
    const list = document.getElementById('clip-generator-video-list');
    try {
        const response = await fetch('/api/videos', { cache: 'no-store' });
        const data = await response.json();
        if (!data.success) throw new Error(data.error || 'No se pudieron cargar los videos');
        clipGeneratorState.videos = data.videos || [];
        renderClipGeneratorVideos();
    } catch (error) {
        list.innerHTML = `<div class="error">${escapeClipGeneratorHtml(error.message)}</div>`;
        document.getElementById('clip-generator-video-count').textContent = 'Error al cargar';
    }
}

function renderClipGeneratorVideos() {
    const list = document.getElementById('clip-generator-video-list');
    const query = (document.getElementById('clip-generator-search')?.value || '').trim().toLowerCase();
    const sourceFilter = document.getElementById('clip-generator-source-filter')?.value || 'all';
    const labelFilter = document.getElementById('clip-generator-label-filter')?.value || 'all';
    const filtered = clipGeneratorState.videos.filter(video => {
        const text = `${video.name || ''} ${video.path || ''}`.toLowerCase();
        const review = video.clip_generator;
        const reviewed = Number(review?.reviewed) || 0;
        const status = review?.status || 'unlabeled';
        const matchesSource = sourceFilter === 'all'
            || (sourceFilter === 'local'
                && !text.includes('youtube')
                && !text.includes('web_downloads')
                && !text.includes('final_tests'))
            || text.includes(sourceFilter);
        const matchesLabel = labelFilter === 'all'
            || (labelFilter === 'labeled' && reviewed > 0)
            || (labelFilter === 'complete' && status === 'complete')
            || (labelFilter === 'in_progress' && status === 'in_progress')
            || (labelFilter === 'unlabeled' && reviewed <= 0);
        return matchesSource && matchesLabel && (!query || text.includes(query));
    });
    const visible = filtered.slice(0, MAX_VISIBLE_CLIP_GENERATOR_VIDEOS);
    const labeledTotal = clipGeneratorState.videos.filter(video => Number(video.clip_generator?.reviewed) > 0).length;
    const visibleCountText = filtered.length > visible.length
        ? `Mostrando ${visible.length} de ${filtered.length}`
        : `${filtered.length} videos`;
    document.getElementById('clip-generator-video-count').textContent = `${visibleCountText} · ${labeledTotal} con etiquetas`;

    if (!visible.length) {
        list.innerHTML = '<div class="loading">No hay videos que coincidan con la búsqueda.</div>';
        return;
    }

    list.innerHTML = visible.map(video => {
        const normalizedName = String(video.name || '').replace(/\\/g, '/');
        const parts = normalizedName.split('/');
        const filename = parts.pop() || normalizedName;
        const folder = parts.join(' / ');
        const isSelected = clipGeneratorState.selectedVideo?.path === video.path;
        const clipCount = Math.floor((Number(video.duration) || 0) / 7);
        const review = video.clip_generator;
        const reviewed = Number(review?.reviewed) || 0;
        const total = Number(review?.total) || clipCount;
        const status = review?.status || 'unlabeled';
        const badgeText = status === 'complete'
            ? 'Etiquetado completo'
            : (status === 'in_progress' ? `${reviewed}/${total} etiquetados` : 'Sin etiquetas');
        const labelSummary = reviewed > 0
            ? `<span class="clip-generator-video-label-summary">${reviewed}/${total} revisados · ${Number(review?.line) || 0} line · ${Number(review?.scrum) || 0} scrum</span>`
            : '';
        return `
            <button type="button" class="clip-generator-video-option ${isSelected ? 'active' : ''}"
                data-video-path="${escapeClipGeneratorHtml(video.path)}" aria-pressed="${isSelected}">
                <span class="clip-generator-video-heading">
                    <span class="clip-generator-video-name">${escapeClipGeneratorHtml(filename)}</span>
                    <span class="clip-generator-video-label-badge ${escapeClipGeneratorHtml(status)}">${escapeClipGeneratorHtml(badgeText)}</span>
                </span>
                <span class="clip-generator-video-folder">${escapeClipGeneratorHtml(folder || 'Biblioteca local')}</span>
                <span class="clip-generator-video-meta">${formatClipGeneratorTime(video.duration)} · ${clipCount} clips</span>
                ${labelSummary}
            </button>
        `;
    }).join('');
}

function selectClipGeneratorVideo(path) {
    clipGeneratorState.selectedVideo = clipGeneratorState.videos.find(video => video.path === path) || null;
    renderClipGeneratorVideos();
    const startButton = document.getElementById('clip-generator-start');
    const selection = document.getElementById('clip-generator-selection');
    startButton.disabled = !clipGeneratorState.selectedVideo;
    if (!clipGeneratorState.selectedVideo) {
        selection.textContent = 'Todavía no seleccionaste un video.';
        return;
    }
    const video = clipGeneratorState.selectedVideo;
    const clipCount = Math.floor((Number(video.duration) || 0) / 7);
    const tailSeconds = Math.max(0, (Number(video.duration) || 0) - (clipCount * 7));
    selection.innerHTML = `
        <strong>${escapeClipGeneratorHtml(video.name)}</strong>
        <span>${formatClipGeneratorTime(video.duration)} · ${clipCount} bloques de 7 s${tailSeconds >= 1 ? ` · ${Math.floor(tailSeconds)} s finales sin incluir` : ''}</span>
        ${Number(video.clip_generator?.reviewed) > 0 ? `
            <span class="clip-generator-selection-labels">${Number(video.clip_generator.reviewed)} de ${Number(video.clip_generator.total) || clipCount} ya etiquetados · ${Number(video.clip_generator.line) || 0} line · ${Number(video.clip_generator.scrum) || 0} scrum</span>
            <a class="clip-generator-existing-session-link" href="${escapeClipGeneratorHtml(video.clip_generator.clip_generator_url || '#')}">Abrir sesión etiquetada</a>
        ` : ''}
    `;
}

async function startClipGeneratorSession() {
    if (!clipGeneratorState.selectedVideo) return;
    const button = document.getElementById('clip-generator-start');
    const message = document.getElementById('clip-generator-start-message');
    const datasetSplit = document.querySelector('input[name="dataset-split"]:checked')?.value || 'train';
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = 'Preparando clips...';
    message.textContent = 'Creando la secuencia de bloques de 7 segundos.';

    try {
        const response = await fetch('/api/clip-generator/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                video_path: clipGeneratorState.selectedVideo.path,
                dataset_split: datasetSplit,
            }),
        });
        const data = await response.json();
        if (!data.success) throw new Error(data.error || 'No se pudo iniciar la sesión');
        setClipGeneratorSession(data.session);
        const url = new URL(window.location.href);
        url.searchParams.set('session_id', data.session.session_id);
        window.history.replaceState({}, '', url);
        message.textContent = '';
    } catch (error) {
        message.textContent = error.message;
        message.style.color = 'var(--danger-color)';
    } finally {
        button.disabled = !clipGeneratorState.selectedVideo;
        button.textContent = originalText;
    }
}

async function loadClipGeneratorSession(sessionId) {
    try {
        const response = await fetch(`/api/review/session/${encodeURIComponent(sessionId)}`, { cache: 'no-store' });
        const data = await response.json();
        if (!data.success) throw new Error(data.error || 'No se pudo cargar la sesión');
        if (data.session.review_mode !== 'clip_generator') throw new Error('La sesión no pertenece al generador de clips');
        setClipGeneratorSession(data.session);
    } catch (error) {
        const message = document.getElementById('clip-generator-start-message');
        message.textContent = error.message;
        message.style.color = 'var(--danger-color)';
    }
}

function setClipGeneratorSession(session) {
    clipGeneratorState.session = session;
    syncClipGeneratorVideoSummary(session);
    const firstPending = session.candidates.findIndex(candidate => !candidate.final_label);
    clipGeneratorState.currentIndex = firstPending >= 0 ? firstPending : Math.max(0, session.candidates.length - 1);
    document.getElementById('clip-generator-workspace').hidden = false;

    const video = document.getElementById('clip-generator-video');
    if (video.dataset.sessionId !== session.session_id) {
        video.dataset.sessionId = session.session_id;
        video.src = session.video_url;
        video.load();
    }
    renderClipGeneratorSession();
    if (clipGeneratorState.comparisonVideoPath !== session.video_path) {
        loadClipGeneratorComparison(session.video_path);
    }
    document.getElementById('clip-generator-workspace').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function syncClipGeneratorVideoSummary(session) {
    const video = clipGeneratorState.videos.find(item => item.path === session?.video_path);
    if (!video) return;
    const summary = session.summary || {};
    const reviewed = Number(summary.reviewed) || 0;
    const total = Number(summary.total_candidates) || session.candidates?.length || 0;
    const pendingValue = Number(summary.pending);
    const pending = Number.isFinite(pendingValue) ? pendingValue : Math.max(0, total - reviewed);
    const counts = summary.final_counts || {};
    video.clip_generator = {
        session_id: session.session_id,
        dataset_split: session.dataset_split || 'train',
        updated_at: session.updated_at,
        status: total > 0 && pending <= 0 ? 'complete' : (reviewed > 0 ? 'in_progress' : 'empty'),
        reviewed,
        total,
        pending,
        line: Number(counts.line) || 0,
        scrum: Number(counts.scrum) || 0,
        clip_generator_url: `/clip-generator?session_id=${encodeURIComponent(session.session_id || '')}`,
    };
    if (clipGeneratorState.selectedVideo?.path === video.path) clipGeneratorState.selectedVideo = video;
    renderClipGeneratorVideos();
}

function renderClipGeneratorSession() {
    const session = clipGeneratorState.session;
    if (!session?.candidates?.length) return;
    const candidate = session.candidates[clipGeneratorState.currentIndex];
    const summary = session.summary || {};
    const reviewed = Number(summary.reviewed) || 0;
    const total = session.candidates.length;
    const percent = total ? Math.round((reviewed / total) * 100) : 0;

    document.getElementById('clip-generator-current-title').textContent = `Clip ${clipGeneratorState.currentIndex + 1} de ${total}`;
    document.getElementById('clip-generator-current-time').textContent = `${formatClipGeneratorTime(candidate.start_time)}–${formatClipGeneratorTime(candidate.end_time)} · ${candidate.duration.toFixed(1)} s`;
    document.getElementById('clip-generator-progress-count').textContent = `${reviewed} de ${total} etiquetados`;
    document.getElementById('clip-generator-progress-percent').textContent = `${percent}%`;
    document.getElementById('clip-generator-track-fill').style.width = `${percent}%`;
    document.getElementById('clip-generator-track-cursor').style.left = `${total > 1 ? (clipGeneratorState.currentIndex / (total - 1)) * 100 : 0}%`;

    const splitBadge = document.getElementById('clip-generator-split-badge');
    splitBadge.textContent = session.dataset_split === 'test' ? 'Prueba' : 'Entrenamiento';
    splitBadge.className = `training-split-badge ${session.dataset_split === 'test' ? 'test' : 'train'}`;

    document.querySelectorAll('.clip-label-button').forEach(button => {
        button.classList.toggle('selected', button.dataset.label === candidate.final_label);
        button.disabled = clipGeneratorState.saving;
    });
    const currentLabel = document.getElementById('clip-generator-current-label');
    currentLabel.textContent = candidate.final_label ? `Etiqueta guardada: ${candidate.final_label}` : 'Sin etiqueta';
    currentLabel.className = `clip-generator-current-label ${candidate.final_label || ''}`;

    document.getElementById('clip-generator-prev').disabled = clipGeneratorState.currentIndex <= 0 || clipGeneratorState.saving;
    document.getElementById('clip-generator-next').disabled = clipGeneratorState.saving || reviewed >= total;
    document.getElementById('clip-generator-export').disabled = clipGeneratorState.saving || reviewed === 0;
    renderClipGeneratorComparison();

    if (reviewed >= total) {
        setClipVideoMessage('Etiquetado completo. Ya podés guardar todos los clips.', 'success');
    } else if (!clipGeneratorState.saving) {
        playCurrentClip();
    }
}

async function loadClipGeneratorComparison(videoPath) {
    clipGeneratorState.comparisonVideoPath = videoPath;
    clipGeneratorState.analysisComparison = null;
    clipGeneratorState.analysisComparisons = [];
    clipGeneratorState.comparisonError = '';
    clipGeneratorState.comparisonLoading = true;
    renderClipGeneratorComparison();

    try {
        const response = await fetch(`/api/clip-generator/analysis?include_all=1&video_path=${encodeURIComponent(videoPath)}`, { cache: 'no-store' });
        const data = await response.json();
        if (!response.ok || !data.success) throw new Error(data.error || 'No se pudo buscar el análisis previo');
        if (clipGeneratorState.comparisonVideoPath !== videoPath) return;
        const comparisons = Array.isArray(data.analyses)
            ? data.analyses
            : (data.available && data.analysis ? [data.analysis] : []);
        clipGeneratorState.analysisComparisons = comparisons;
        clipGeneratorState.analysisComparison = comparisons[0] || null;
    } catch (error) {
        if (clipGeneratorState.comparisonVideoPath !== videoPath) return;
        clipGeneratorState.comparisonError = error.message;
    } finally {
        if (clipGeneratorState.comparisonVideoPath === videoPath) {
            clipGeneratorState.comparisonLoading = false;
            renderClipGeneratorComparison();
        }
    }
}

function renderClipGeneratorComparison() {
    const session = clipGeneratorState.session;
    const container = document.getElementById('clip-generator-comparison');
    if (!container || !session?.candidates?.length) return;

    const meta = document.getElementById('clip-generator-comparison-meta');
    const empty = document.getElementById('clip-generator-comparison-empty');
    const content = document.getElementById('clip-generator-comparison-content');
    const predictionRow = document.getElementById('clip-generator-prediction-row');
    const link = document.getElementById('clip-generator-analysis-link');
    const picker = document.getElementById('clip-generator-analysis-picker');
    const select = document.getElementById('clip-generator-analysis-select');
    const comparison = clipGeneratorState.analysisComparison;
    const comparisons = clipGeneratorState.analysisComparisons;
    const reviewedCount = session.candidates.filter(candidate => candidate.final_label).length;
    const score = document.getElementById('clip-generator-comparison-score');

    content.hidden = false;
    drawClipGeneratorTimeline(
        document.getElementById('clip-generator-label-timeline'),
        session.candidates.map(candidate => ({ label: candidate.final_label || 'pending', confidence: 1 })),
        'Tus etiquetas manuales'
    );

    select.innerHTML = comparisons.map(item => {
        const counts = item.detection_counts || {};
        const details = `${Number(counts.line) || 0} line · ${Number(counts.scrum) || 0} scrum`;
        return `<option value="${escapeClipGeneratorHtml(item.result_id)}">${escapeClipGeneratorHtml(friendlyClipGeneratorModelName(item.model_name))} — ${details}</option>`;
    }).join('');
    if (comparison) select.value = comparison.result_id;
    picker.hidden = comparisons.length < 2;

    if (clipGeneratorState.comparisonLoading) {
        meta.textContent = 'Buscando un análisis previo para este video...';
        empty.textContent = 'Buscando predicciones guardadas. Tus etiquetas ya están visibles en el timeline.';
        empty.hidden = false;
        predictionRow.hidden = true;
        link.hidden = true;
        picker.hidden = true;
        score.textContent = `${reviewedCount} de ${session.candidates.length} clips etiquetados`;
        return;
    }

    if (!comparison) {
        meta.textContent = clipGeneratorState.comparisonError || 'Tus etiquetas manuales se actualizan después de cada selección.';
        empty.textContent = clipGeneratorState.comparisonError
            ? 'No se pudieron cargar las predicciones, pero podés continuar etiquetando.'
            : 'Este video todavía no tiene predicciones guardadas. Cuando lo analices, aparecerán encima de tus etiquetas.';
        empty.hidden = false;
        predictionRow.hidden = true;
        link.hidden = true;
        score.textContent = `${reviewedCount} de ${session.candidates.length} clips etiquetados`;
        return;
    }

    meta.textContent = comparisons.length > 1
        ? `${comparisons.length} predicciones guardadas para este video. Elegí una para comparar.`
        : `${friendlyClipGeneratorModelName(comparison.model_name)} · resultado ${comparison.result_id}`;
    empty.hidden = true;
    predictionRow.hidden = false;
    link.hidden = false;
    link.href = comparison.analysis_url || '#';

    const predictionsByBlock = new Map((comparison.predictions || []).map(item => [Number(item.block_index), item]));
    drawClipGeneratorTimeline(
        document.getElementById('clip-generator-prediction-timeline'),
        session.candidates.map((candidate, index) => {
            const prediction = predictionsByBlock.get(Number(candidate.block_index ?? index));
            return {
                label: prediction?.predicted_label || 'other',
                confidence: Number(prediction?.confidence) || 0,
            };
        }),
        'Predicciones del modelo'
    );
    const reviewed = session.candidates.filter(candidate => candidate.final_label);
    const matches = reviewed.filter((candidate, index) => {
        const blockIndex = Number(candidate.block_index ?? session.candidates.indexOf(candidate));
        return predictionsByBlock.get(blockIndex)?.predicted_label === candidate.final_label;
    }).length;
    const currentCandidate = currentClipGeneratorCandidate();
    const currentBlock = Number(currentCandidate?.block_index ?? clipGeneratorState.currentIndex);
    const currentPrediction = predictionsByBlock.get(currentBlock);
    const predictionText = currentPrediction
        ? `${String(currentPrediction.predicted_label).toUpperCase()} ${Math.round((Number(currentPrediction.confidence) || 0) * 100)}%`
        : 'sin predicción';
    const currentReal = currentCandidate?.final_label ? String(currentCandidate.final_label).toUpperCase() : 'pendiente';
    const agreementText = reviewed.length
        ? `${matches} de ${reviewed.length} coincidencias (${Math.round((matches / reviewed.length) * 100)}%)`
        : 'todavía no hay etiquetas para comparar';
    score.textContent = `Clip actual: ${predictionText} · real: ${currentReal} · ${agreementText}`;
}

function drawClipGeneratorTimeline(canvas, items, accessibleName) {
    if (!canvas || !items.length) return;
    const width = Math.max(1, Math.round(canvas.getBoundingClientRect().width));
    const height = 32;
    const pixelRatio = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = Math.round(width * pixelRatio);
    canvas.height = Math.round(height * pixelRatio);
    const context = canvas.getContext('2d');
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, width, height);

    const colors = {
        line: '#10b981',
        scrum: '#ef4444',
        other: '#3b82f6',
        pending: '#cbd5e1',
    };
    const segmentWidth = width / items.length;
    items.forEach((item, index) => {
        const confidence = Math.max(0, Math.min(1, Number(item.confidence) || 0));
        context.globalAlpha = item.label === 'pending' ? 1 : 0.45 + (confidence * 0.55);
        context.fillStyle = colors[item.label] || colors.other;
        const left = index * segmentWidth;
        const right = (index + 1) * segmentWidth;
        context.fillRect(left, 0, Math.max(1, right - left), height);
        if (segmentWidth >= 4) {
            context.globalAlpha = 0.24;
            context.fillStyle = '#ffffff';
            context.fillRect(Math.floor(right) - 1, 0, 1, height);
        }
    });
    context.globalAlpha = 1;
    const cursorLeft = clipGeneratorState.currentIndex * segmentWidth;
    context.strokeStyle = '#111827';
    context.lineWidth = 2;
    context.strokeRect(Math.min(width - 2, cursorLeft + 1), 1, Math.max(1, Math.min(segmentWidth - 2, width - cursorLeft - 2)), height - 2);
    canvas.setAttribute('aria-label', `${accessibleName}. ${items.length} bloques. Clip actual ${clipGeneratorState.currentIndex + 1}.`);
}

function jumpToClipFromTimeline(event) {
    const candidates = clipGeneratorState.session?.candidates || [];
    if (!candidates.length) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    if (!bounds.width) return;
    const ratio = Math.max(0, Math.min(0.999999, (event.clientX - bounds.left) / bounds.width));
    clipGeneratorState.currentIndex = Math.floor(ratio * candidates.length);
    renderClipGeneratorSession();
}

function currentClipGeneratorCandidate() {
    return clipGeneratorState.session?.candidates?.[clipGeneratorState.currentIndex] || null;
}

function playCurrentClip() {
    const candidate = currentClipGeneratorCandidate();
    const video = document.getElementById('clip-generator-video');
    if (!candidate || !video) return;
    const startPlayback = () => {
        setClipVideoMessage('');
        video.currentTime = candidate.start_time;
        video.play().catch(() => setClipVideoMessage('Usá reproducir para ver este fragmento.'));
    };
    if (video.readyState >= 1) startPlayback();
    else video.addEventListener('loadedmetadata', startPlayback, { once: true });
}

function stopAtCurrentClipEnd() {
    const candidate = currentClipGeneratorCandidate();
    const video = document.getElementById('clip-generator-video');
    if (!candidate || !video || video.paused) return;
    if (video.currentTime >= candidate.end_time - 0.04) {
        video.pause();
        video.currentTime = candidate.end_time;
        setClipVideoMessage('Terminó el fragmento. Elegí una clase o volvé a reproducirlo.');
    }
}

async function saveClipLabel(label) {
    const session = clipGeneratorState.session;
    if (!session || clipGeneratorState.saving) return;
    clipGeneratorState.saving = true;
    let saveError = '';
    renderClipGeneratorSession();
    setClipVideoMessage(`Guardando ${label}...`);

    try {
        const response = await fetch(`/api/review/session/${encodeURIComponent(session.session_id)}/decision`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                candidate_index: clipGeneratorState.currentIndex,
                final_label: label,
            }),
        });
        const data = await response.json();
        if (!data.success) throw new Error(data.error || 'No se pudo guardar la etiqueta');
        clipGeneratorState.session = data.session;
        syncClipGeneratorVideoSummary(data.session);
        const nextPending = findNextPendingClip(clipGeneratorState.currentIndex + 1);
        if (nextPending !== null) clipGeneratorState.currentIndex = nextPending;
    } catch (error) {
        saveError = error.message;
    } finally {
        clipGeneratorState.saving = false;
        renderClipGeneratorSession();
        if (saveError) setClipVideoMessage(saveError, 'error');
    }
}

function findNextPendingClip(startIndex) {
    const candidates = clipGeneratorState.session?.candidates || [];
    for (let index = startIndex; index < candidates.length; index += 1) {
        if (!candidates[index].final_label) return index;
    }
    for (let index = 0; index < startIndex; index += 1) {
        if (!candidates[index].final_label) return index;
    }
    return null;
}

function moveClipGenerator(delta) {
    const candidates = clipGeneratorState.session?.candidates || [];
    const nextIndex = clipGeneratorState.currentIndex + delta;
    if (nextIndex < 0 || nextIndex >= candidates.length) return;
    clipGeneratorState.currentIndex = nextIndex;
    renderClipGeneratorSession();
}

function moveToNextPendingClip() {
    const nextPending = findNextPendingClip(clipGeneratorState.currentIndex + 1);
    if (nextPending === null) return;
    clipGeneratorState.currentIndex = nextPending;
    renderClipGeneratorSession();
}

async function exportLabeledClips() {
    const session = clipGeneratorState.session;
    if (!session || clipGeneratorState.saving) return;
    const button = document.getElementById('clip-generator-export');
    const message = document.getElementById('clip-generator-export-message');
    const originalText = button.textContent;
    clipGeneratorState.saving = true;
    button.disabled = true;
    button.textContent = 'Generando clips...';
    message.textContent = 'El recorte puede tardar según la cantidad de fragmentos etiquetados.';

    try {
        const response = await fetch(`/api/review/session/${encodeURIComponent(session.session_id)}/export`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });
        const data = await response.json();
        if (!data.success) throw new Error(data.error || 'No se pudieron generar los clips');
        clipGeneratorState.session = data.session;
        syncClipGeneratorVideoSummary(data.session);
        const destination = session.dataset_split === 'test' ? 'test' : 'training';
        message.style.color = 'var(--success-color)';
        message.textContent = `${data.exports.exported_count} clips guardados en ${destination}.`;
    } catch (error) {
        message.style.color = 'var(--danger-color)';
        message.textContent = error.message;
    } finally {
        clipGeneratorState.saving = false;
        button.textContent = originalText;
        renderClipGeneratorSession();
    }
}

function handleClipGeneratorShortcut(event) {
    if (!clipGeneratorState.session || event.ctrlKey || event.metaKey || event.altKey) return;
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;
    const key = event.key.toLowerCase();
    if (key === '1' || key === 'l') saveClipLabel('line');
    else if (key === '2' || key === 's') saveClipLabel('scrum');
    else if (key === '3' || key === 'o') saveClipLabel('other');
    else if (key === 'arrowleft') moveClipGenerator(-1);
    else if (key === 'arrowright') moveToNextPendingClip();
    else if (key === ' ') {
        event.preventDefault();
        playCurrentClip();
    }
}

function formatClipGeneratorTime(seconds) {
    const totalSeconds = Math.max(0, Math.floor(Number(seconds) || 0));
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const secs = totalSeconds % 60;
    if (hours) return `${hours}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    return `${minutes}:${String(secs).padStart(2, '0')}`;
}

function setClipVideoMessage(message, state = '') {
    const element = document.getElementById('clip-generator-video-message');
    element.textContent = message;
    element.className = `review-help ${state}`.trim();
}

function friendlyClipGeneratorModelName(modelName) {
    const name = String(modelName || 'Modelo analizado');
    if (name.includes('best_model/best_model.keras') || /^InceptionV3-GRU \(best_model\/best_model\.keras\)$/i.test(name)) {
        return 'ResNet-50 + BiGRU';
    }
    return name;
}

function escapeClipGeneratorHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
