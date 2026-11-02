let reviewSession = null;
let reviewSessionId = null;
let currentCandidateIndex = null;
let reviewModelRoiController = null;

document.addEventListener('DOMContentLoaded', () => {
    initializeReview();
});

async function initializeReview() {
    setupReviewEventListeners();

    const bootstrap = window.REVIEW_BOOTSTRAP || {};
    if (bootstrap.sessionId) {
        reviewSessionId = bootstrap.sessionId;
        await loadReviewSession(reviewSessionId);
        return;
    }

    if (bootstrap.resultId) {
        await startReviewFromResult(bootstrap.resultId);
        return;
    }

    setReviewControlsEnabled(false);
    showReviewStatus('No hay una sesión de revisión activa. Iniciá la revisión desde el resultado de un análisis.', true);
}

function setReviewControlsEnabled(enabled) {
    ['review-accept', 'review-line', 'review-scrum', 'review-other', 'review-no-event', 'review-prev', 'review-next', 'review-export']
        .forEach(id => {
            const control = document.getElementById(id);
            if (control) control.disabled = !enabled;
        });
}

function setupReviewEventListeners() {
    const sourceVideo = document.getElementById('review-source-video');
    reviewModelRoiController = window.ModelRoiOverlay?.attach({
        video: sourceVideo,
        stage: document.getElementById('review-roi-stage'),
        overlay: document.getElementById('review-model-roi'),
        toggle: document.getElementById('show-review-model-roi'),
    });

    document.getElementById('review-accept')?.addEventListener('click', () => applyDecision(getCurrentCandidate()?.predicted_label));
    document.getElementById('review-line')?.addEventListener('click', () => applyDecision('line'));
    document.getElementById('review-scrum')?.addEventListener('click', () => applyDecision('scrum'));
    document.getElementById('review-other')?.addEventListener('click', () => applyDecision('other'));
    document.getElementById('review-no-event')?.addEventListener('click', () => applyDecision('no_event'));
    document.getElementById('review-prev')?.addEventListener('click', () => moveCandidate(-1));
    document.getElementById('review-next')?.addEventListener('click', () => moveCandidate(1));
    document.getElementById('review-export')?.addEventListener('click', exportReviewedClips);
    document.getElementById('show-cropped-preview')?.addEventListener('change', renderCurrentCandidate);
    document.getElementById('review-source-fullscreen')?.addEventListener('click', () => {
        const stage = document.getElementById('review-roi-stage');
        if (stage?.requestFullscreen) stage.requestFullscreen();
        else if (stage?.webkitRequestFullscreen) stage.webkitRequestFullscreen();
    });

    window.addEventListener('keydown', (event) => {
        const tagName = document.activeElement?.tagName?.toLowerCase();
        if (tagName === 'input' || tagName === 'textarea') {
            return;
        }

        const key = event.key.toLowerCase();
        const isBlindManual = reviewSession?.review_mode === 'clip_generator';
        const isFullMatch = reviewSession?.review_mode === 'full_match';
        if (key === 'a' && !isBlindManual) {
            event.preventDefault();
            applyDecision(getCurrentCandidate()?.predicted_label);
        } else if (key === 'l') {
            event.preventDefault();
            applyDecision('line');
        } else if (key === 's') {
            event.preventDefault();
            applyDecision('scrum');
        } else if (key === 'o') {
            event.preventDefault();
            applyDecision('other');
        } else if (key === 'n' && isFullMatch) {
            event.preventDefault();
            applyDecision('no_event');
        } else if (event.key === 'ArrowRight') {
            event.preventDefault();
            moveCandidate(1);
        } else if (event.key === 'ArrowLeft') {
            event.preventDefault();
            moveCandidate(-1);
        }
    });
}

function showReviewStatus(message, isError = false) {
    const status = document.getElementById('review-status');
    status.textContent = message;
    status.style.color = isError ? 'var(--danger-color)' : 'var(--text-secondary)';
}

async function startReviewFromResult(resultId) {
    showReviewStatus('Preparing review session...');
    try {
        const response = await fetch('/api/review/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ result_id: resultId })
        });
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || 'Failed to create review session');
        }

        reviewSessionId = data.session.session_id;
        if (window.history?.replaceState) {
            window.history.replaceState({}, document.title, `/review?session_id=${reviewSessionId}`);
        }
        await loadReviewSession(reviewSessionId);
    } catch (error) {
        showReviewStatus(error.message, true);
    }
}

async function loadReviewSession(sessionId) {
    showReviewStatus('Loading review session...');
    try {
        const response = await fetch(`/api/review/session/${sessionId}`);
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || 'Failed to load review session');
        }

        reviewSessionId = sessionId;
        reviewSession = data.session;
        const firstPending = reviewSession.candidates.findIndex(candidate => !candidate.final_label);
        if (currentCandidateIndex === null) {
            currentCandidateIndex = firstPending >= 0 ? firstPending : 0;
        } else {
            currentCandidateIndex = Math.min(currentCandidateIndex, reviewSession.candidates.length - 1);
        }
        renderReviewSession();
    } catch (error) {
        showReviewStatus(error.message, true);
    }
}

function renderReviewSession() {
    if (!reviewSession) {
        return;
    }

    const isNonOverlapping = reviewSession.review_mode === 'non_overlapping' || reviewSession.review_mode === 'full_match';
    const isFullMatch = reviewSession.review_mode === 'full_match';
    const isBlindManual = reviewSession.review_mode === 'clip_generator';
    const introText = document.getElementById('review-intro-text');
    if (introText) {
        introText.textContent = isBlindManual
            ? 'Etiquetá cada bloque de forma ciega para construir la referencia temporal del benchmark de partidos.'
            : 'Revisá las detecciones, corregí su clase y prepará fragmentos para futuros entrenamientos.';
    }
    const helpText = document.getElementById('review-help-text');
    if (helpText) {
        helpText.textContent = isBlindManual
            ? 'Benchmark ciego de tres clases: revisá cada bloque consecutivo de 7 segundos y marcá line, scrum u other. Las predicciones del modelo permanecen ocultas.'
            : isFullMatch
            ? 'Modo partido completo: revisá todos los bloques consecutivos de 7 segundos. Usá no_event para pausas, repeticiones, público o ausencia de jugada.'
            : isNonOverlapping
                ? 'Bloques fijos consecutivos sin solapamiento: 0–7 s, 7–14 s, 14–21 s y así sucesivamente. Las probabilidades se promedian dentro de cada bloque.'
            : 'Las ventanas superpuestas originales se mantienen para revisar evidencia temporal con solapamiento.';
    }
    showReviewStatus(`${reviewSession.review_mode_label || (isNonOverlapping ? 'Bloques sin solapamiento' : 'Ventanas superpuestas')} · ${reviewSession.video_name} · ${reviewSession.model_name || 'modelo seleccionado'}`);
    renderSummary();
    renderQueue();
    renderExportedClips();

    const noEventButton = document.getElementById('review-no-event');
    if (noEventButton) {
        noEventButton.style.display = isFullMatch ? '' : 'none';
    }
    const acceptButton = document.getElementById('review-accept');
    if (acceptButton) {
        acceptButton.style.display = isBlindManual ? 'none' : '';
    }
    const probabilitiesSection = document.getElementById('review-probabilities-section');
    if (probabilitiesSection) {
        probabilitiesSection.style.display = isBlindManual ? 'none' : '';
    }
    const shortcutsHelp = document.getElementById('review-shortcuts-help');
    if (shortcutsHelp) {
        shortcutsHelp.textContent = isBlindManual
            ? 'Atajos: L line, S scrum, O other, flechas para navegar.'
            : isFullMatch
                ? 'Atajos: A aceptar, L line, S scrum, O other, N no_event, flechas para navegar.'
                : 'Atajos: A aceptar, L line, S scrum, O other, flechas para navegar.';
    }

    document.getElementById('review-empty').style.display = reviewSession.candidates.length ? 'none' : 'block';
    document.getElementById('review-main').style.display = reviewSession.candidates.length ? 'block' : 'none';
    document.getElementById('review-summary').style.display = 'grid';
    setReviewControlsEnabled(reviewSession.candidates.length > 0);

    if (!reviewSession.candidates.length) {
        document.getElementById('review-empty').textContent = 'Este resultado no contiene detecciones de scrum o line para revisar.';
        return;
    }

    renderCurrentCandidate();
}

function renderSummary() {
    const summary = reviewSession.summary || {};
    const finalCounts = summary.final_counts || {};
    const summaryEl = document.getElementById('review-summary');
    summaryEl.innerHTML = `
        <div class="review-summary-card">
            <div class="label">Candidatas</div>
            <div class="value">${summary.total_candidates || 0}</div>
        </div>
        <div class="review-summary-card">
            <div class="label">Revisadas</div>
            <div class="value">${summary.reviewed || 0}</div>
        </div>
        <div class="review-summary-card">
            <div class="label">Pendientes</div>
            <div class="value">${summary.pending || 0}</div>
        </div>
        <div class="review-summary-card">
            <div class="label">Aceptadas</div>
            <div class="value">${summary.accepted || 0}</div>
        </div>
        <div class="review-summary-card">
            <div class="label">Corregidas</div>
            <div class="value">${summary.relabeled || 0}</div>
        </div>
        <div class="review-summary-card">
            <div class="label">Etiquetas finales</div>
            <div class="value">L ${finalCounts.line || 0} · S ${finalCounts.scrum || 0} · O ${finalCounts.other || 0}</div>
        </div>
    `;
}

function renderQueue() {
    const queueEl = document.getElementById('review-queue');
    const isBlindManual = reviewSession?.review_mode === 'clip_generator';
    const isNonOverlapping = reviewSession?.review_mode === 'non_overlapping' || isBlindManual;
    queueEl.innerHTML = reviewSession.candidates.map((candidate, index) => {
        const isActive = index === currentCandidateIndex;
        const done = Boolean(candidate.final_label);
        const mergedCount = candidate.merged_detection_count || 1;
        const sourceText = isNonOverlapping
            ? `${mergedCount} ventanas fuente promediadas`
            : `${mergedCount} ventanas agrupadas`;
        return `
            <button class="review-queue-item ${isActive ? 'active' : ''} ${done ? 'done' : ''}" data-candidate-index="${index}">
                <div class="queue-title">
                    <span>${isBlindManual ? 'Bloque' : `<span class="badge ${candidate.predicted_label}">${candidate.predicted_label.toUpperCase()}</span>`} #${index + 1}</span>
                    <span>${done ? `<span class="badge ${candidate.final_label}">${candidate.final_label.toUpperCase()}</span>` : 'Pending'}</span>
                </div>
                <div class="queue-meta">${formatTime(candidate.start_time)} – ${formatTime(candidate.end_time)}${isBlindManual ? '' : ` · ${(candidate.confidence * 100).toFixed(1)}%`}${mergedCount > 1 ? ` · ${sourceText}` : ''}</div>
            </button>
        `;
    }).join('');

    queueEl.querySelectorAll('.review-queue-item').forEach(button => {
        button.addEventListener('click', () => {
            currentCandidateIndex = parseInt(button.dataset.candidateIndex, 10);
            renderCurrentCandidate();
            renderQueue();
        });
    });
}

function getCurrentCandidate() {
    if (!reviewSession || currentCandidateIndex === null) {
        return null;
    }
    return reviewSession.candidates[currentCandidateIndex] || null;
}

function renderCurrentCandidate() {
    const candidate = getCurrentCandidate();
    if (!candidate) {
        return;
    }

    const isBlindManual = reviewSession?.review_mode === 'clip_generator';
    document.getElementById('review-current-title').textContent = `${isBlindManual ? 'Bloque' : 'Detección'} ${currentCandidateIndex + 1}`;
    document.getElementById('review-current-subtitle').textContent = isBlindManual
        ? `${formatTime(candidate.start_time)} – ${formatTime(candidate.end_time)} · evaluación ciega de tres clases`
        : `${formatTime(candidate.start_time)} – ${formatTime(candidate.end_time)} · predicted ${candidate.predicted_label.toUpperCase()} at ${(candidate.confidence * 100).toFixed(1)}% confidence`;

    const metaGrid = document.getElementById('review-meta-grid');
    const isNonOverlapping = reviewSession?.review_mode === 'non_overlapping' || isBlindManual;
    metaGrid.innerHTML = `
        <div class="review-meta-card">
            <strong>${isBlindManual ? 'Protocolo' : 'Predicted label'}</strong>
            ${isBlindManual ? '<span>Ciego · line / scrum / other</span>' : `<span class="badge ${candidate.predicted_label}">${candidate.predicted_label.toUpperCase()}</span>`}
        </div>
        <div class="review-meta-card">
            <strong>Current decision</strong>
            ${candidate.final_label ? `<span class="badge ${candidate.final_label}">${candidate.final_label.toUpperCase()}</span>` : '<span>Pending</span>'}
        </div>
        <div class="review-meta-card">
            <strong>${isNonOverlapping ? 'Ventanas fuente promediadas' : 'Ventanas agrupadas'}</strong>
            <span>${candidate.merged_detection_count || 1} corte(s) fuente</span>
        </div>
        <div class="review-meta-card">
            <strong>Time window</strong>
            <span>${formatTime(candidate.start_time)} – ${formatTime(candidate.end_time)} (${candidate.duration.toFixed(1)}s)</span>
        </div>
        <div class="review-meta-card">
            <strong>Reviewed at</strong>
            <span>${candidate.reviewed_at || 'Not reviewed yet'}</span>
        </div>
    `;

    renderProbabilities(isBlindManual ? {} : (candidate.probabilities || {}));
    renderSourceVideo(candidate);
    renderCroppedPreview(candidate);
}

function renderProbabilities(probabilities) {
    const container = document.getElementById('review-probabilities');
    const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
    if (!entries.length) {
        container.innerHTML = '<div class="review-empty">No per-class probabilities available for this detection.</div>';
        return;
    }

    container.innerHTML = entries.map(([label, probability]) => `
        <div class="review-prob-row">
            <span>${label}</span>
            <div class="review-prob-bar"><div class="review-prob-fill" style="width:${Math.max(0, Math.min(100, probability * 100))}%;"></div></div>
            <strong>${(probability * 100).toFixed(1)}%</strong>
        </div>
    `).join('');
}

function renderSourceVideo(candidate) {
    const sourceVideo = document.getElementById('review-source-video');
    const videoUrl = reviewSession.video_url;
    if (!videoUrl) {
        return;
    }
    if (!sourceVideo.src || !sourceVideo.src.endsWith(videoUrl)) {
        sourceVideo.src = videoUrl;
        sourceVideo.load();
        sourceVideo.onloadedmetadata = () => {
            playBoundedSegment(sourceVideo, candidate.start_time, candidate.end_time);
        };
    } else if (sourceVideo.readyState >= 1) {
        playBoundedSegment(sourceVideo, candidate.start_time, candidate.end_time);
    }
}

function playBoundedSegment(videoElement, startTime, endTime) {
    if (videoElement._boundedHandler) {
        videoElement.removeEventListener('timeupdate', videoElement._boundedHandler);
    }

    videoElement.currentTime = startTime;
    const stopAtEnd = () => {
        if (videoElement.currentTime >= endTime) {
            videoElement.pause();
            videoElement.currentTime = startTime;
        }
    };
    videoElement._boundedHandler = stopAtEnd;
    videoElement.addEventListener('timeupdate', stopAtEnd);
    videoElement.play().catch(() => {});
}

function renderCroppedPreview(candidate) {
    const showPreview = document.getElementById('show-cropped-preview').checked;
    const previewBox = document.getElementById('cropped-preview-box');
    const previewVideo = document.getElementById('review-cropped-video');
    const grid = document.getElementById('review-player-grid');

    if (!showPreview) {
        previewBox.style.display = 'none';
        grid.classList.remove('with-preview');
        previewVideo.pause();
        previewVideo.removeAttribute('src');
        previewVideo.load();
        return;
    }

    previewBox.style.display = 'grid';
    grid.classList.add('with-preview');
    previewVideo.src = `/api/review/session/${reviewSessionId}/cropped-preview/${currentCandidateIndex}`;
    previewVideo.load();
}

async function applyDecision(finalLabel) {
    const candidate = getCurrentCandidate();
    if (!candidate || !finalLabel) {
        return;
    }

    try {
        const response = await fetch(`/api/review/session/${reviewSessionId}/decision`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                candidate_index: currentCandidateIndex,
                final_label: finalLabel
            })
        });
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || 'Failed to save review decision');
        }

        reviewSession = data.session;
        const exportMessageEl = document.getElementById('review-export-message');
        if (exportMessageEl) {
            if (data.auto_exported) {
                exportMessageEl.style.color = 'var(--text-secondary)';
                exportMessageEl.textContent = `All items reviewed. Auto-synced to training dataset: ${reviewSession.exports?.central_manifest_path || reviewSession.exports?.manifest_path || 'manifest updated'}`;
            } else if (data.dataset_synced) {
                exportMessageEl.style.color = 'var(--text-secondary)';
                exportMessageEl.textContent = 'Decision saved. Training dataset already synced for this session.';
            } else {
                exportMessageEl.style.color = 'var(--warn-color, #f59e0b)';
                exportMessageEl.textContent = 'Decision saved locally. Use Export Reviewed Clips to sync reviewed items to the training dataset before the queue is complete.';
            }

            if (data.sync_warning) {
                exportMessageEl.style.color = 'var(--danger-color)';
                exportMessageEl.textContent = `Decision saved, but auto-sync failed: ${data.sync_warning}`;
            }
        }

        const nextPending = findNextPendingCandidate(currentCandidateIndex + 1);
        currentCandidateIndex = nextPending !== null ? nextPending : currentCandidateIndex;
        renderReviewSession();
    } catch (error) {
        alert(`Could not save decision: ${error.message}`);
    }
}

function moveCandidate(delta) {
    if (!reviewSession?.candidates?.length) {
        return;
    }
    const nextIndex = currentCandidateIndex + delta;
    if (nextIndex < 0 || nextIndex >= reviewSession.candidates.length) {
        return;
    }
    currentCandidateIndex = nextIndex;
    renderCurrentCandidate();
    renderQueue();
}

function findNextPendingCandidate(startIndex) {
    if (!reviewSession?.candidates?.length) {
        return null;
    }

    for (let index = startIndex; index < reviewSession.candidates.length; index += 1) {
        if (!reviewSession.candidates[index].final_label) {
            return index;
        }
    }
    for (let index = 0; index < startIndex; index += 1) {
        if (!reviewSession.candidates[index].final_label) {
            return index;
        }
    }
    return null;
}

async function exportReviewedClips() {
    if (!reviewSessionId) {
        return;
    }

    const button = document.getElementById('review-export');
    const message = document.getElementById('review-export-message');
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = 'Exporting...';
    message.textContent = '';

    try {
        const response = await fetch(`/api/review/session/${reviewSessionId}/export`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || 'Export failed');
        }

        reviewSession = data.session;
        renderReviewSession();
        message.style.color = 'var(--text-secondary)';
        message.textContent = `Se exportaron ${data.exports.exported_count} fragmentos al conjunto revisado.`;
    } catch (error) {
        message.textContent = error.message;
        message.style.color = 'var(--danger-color)';
    } finally {
        button.disabled = !reviewSession?.candidates?.length;
        button.textContent = originalText;
    }
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function renderExportedClips() {
    const panel = document.getElementById('review-exported-clips');
    const list = document.getElementById('review-exported-clip-list');
    const video = document.getElementById('review-exported-video');
    if (!panel || !list || !video || !reviewSession) {
        return;
    }

    const exported = (reviewSession.candidates || [])
        .map((candidate, index) => ({ candidate, index }))
        .filter(({ candidate }) => Boolean(candidate.exported_clip_path));

    if (!exported.length) {
        panel.style.display = 'none';
        list.innerHTML = '';
        video.pause();
        video.removeAttribute('src');
        video.style.display = 'none';
        video.load();
        return;
    }

    panel.style.display = 'grid';
    list.innerHTML = exported.map(({ candidate, index }) => {
        const label = (candidate.final_label || candidate.predicted_label || 'other').toUpperCase();
        const confidence = Number(candidate.confidence || 0);
        return `
            <div class="review-exported-clip-item">
                <button type="button" class="btn btn-secondary review-exported-clip-button" data-candidate-index="${index}">
                    Ver ${formatTime(candidate.start_time)}–${formatTime(candidate.end_time)} · ${label} · ${(confidence * 100).toFixed(1)}%
                </button>
                <a href="/api/review/session/${reviewSessionId}/exported-clip/${index}" target="_blank" rel="noopener">Abrir</a>
            </div>
        `;
    }).join('');

    list.querySelectorAll('.review-exported-clip-button').forEach(button => {
        button.addEventListener('click', () => {
            const candidateIndex = Number(button.dataset.candidateIndex);
            video.src = `/api/review/session/${reviewSessionId}/exported-clip/${candidateIndex}`;
            video.style.display = 'block';
            video.load();
        });
    });

    // Select the first exported clip without autoplay so the browser does not
    // interrupt the review page with an autoplay prompt.
    const firstIndex = exported[0].index;
    video.src = `/api/review/session/${reviewSessionId}/exported-clip/${firstIndex}`;
    video.style.display = 'block';
    video.load();
}
