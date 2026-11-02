const trainingClipsState = {
    clips: [],
    page: 1,
    pages: 1,
    selectedFilename: null,
    searchTimer: null,
};

const TRAINING_SPLIT_LABELS = {
    train: 'Entrenamiento',
    validation: 'Validación',
    test: 'Test',
    unassigned: 'No seleccionado',
};

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('training-clip-fullscreen')?.addEventListener('click', () => {
        const stage = document.getElementById('training-clip-roi-stage');
        if (stage?.requestFullscreen) stage.requestFullscreen();
        else if (stage?.webkitRequestFullscreen) stage.webkitRequestFullscreen();
    });
    document.getElementById('training-clips-refresh')?.addEventListener('click', loadTrainingClips);
    document.getElementById('training-clips-label')?.addEventListener('change', resetAndLoadTrainingClips);
    document.getElementById('training-clips-split')?.addEventListener('change', resetAndLoadTrainingClips);
    document.getElementById('training-clips-search')?.addEventListener('input', () => {
        window.clearTimeout(trainingClipsState.searchTimer);
        trainingClipsState.searchTimer = window.setTimeout(resetAndLoadTrainingClips, 250);
    });
    document.getElementById('training-clips-prev')?.addEventListener('click', () => changeTrainingClipsPage(-1));
    document.getElementById('training-clips-next')?.addEventListener('click', () => changeTrainingClipsPage(1));
    document.getElementById('training-clips-grid')?.addEventListener('click', (event) => {
        const button = event.target.closest('[data-training-clip-index]');
        if (!button) return;
        selectTrainingClip(Number(button.dataset.trainingClipIndex));
    });
    loadTrainingClips();
});

function resetAndLoadTrainingClips() {
    trainingClipsState.page = 1;
    loadTrainingClips();
}

function changeTrainingClipsPage(delta) {
    const nextPage = Math.max(1, Math.min(trainingClipsState.pages, trainingClipsState.page + delta));
    if (nextPage === trainingClipsState.page) return;
    trainingClipsState.page = nextPage;
    loadTrainingClips();
}

async function loadTrainingClips() {
    const grid = document.getElementById('training-clips-grid');
    grid.innerHTML = '<div class="loading">Cargando clips...</div>';

    const params = new URLSearchParams({
        q: document.getElementById('training-clips-search')?.value || '',
        label: document.getElementById('training-clips-label')?.value || 'all',
        split: document.getElementById('training-clips-split')?.value || 'train',
        page: String(trainingClipsState.page),
        page_size: '24',
    });

    try {
        const response = await fetch(`/api/training-clips?${params}`, { cache: 'no-store' });
        const data = await response.json();
        if (!response.ok || !data.success) throw new Error(data.error || 'No se pudieron cargar los clips');

        trainingClipsState.clips = data.clips || [];
        trainingClipsState.page = data.pagination.page;
        trainingClipsState.pages = data.pagination.pages;
        renderTrainingClipsSummary(data.summary || {});
        renderTrainingClips(data.pagination);
    } catch (error) {
        grid.innerHTML = `<div class="dataset-empty">${escapeTrainingClipsHtml(error.message)}</div>`;
        document.getElementById('training-clips-result-count').textContent = 'Error al cargar';
    }
}

function renderTrainingClipsSummary(summary) {
    document.getElementById('training-clips-collection').textContent = formatTrainingClipsNumber(summary.collection_clips);
    document.getElementById('training-clips-train-examples').textContent = formatTrainingClipsNumber(summary.training_examples);
    document.getElementById('training-clips-train-unique').textContent = `${formatTrainingClipsNumber(summary.unique_training_clips)} clips únicos`;
    document.getElementById('training-clips-test').textContent = formatTrainingClipsNumber(summary.test_clips);
    document.getElementById('training-clips-validation').textContent = formatTrainingClipsNumber(summary.validation_clips);
    document.getElementById('training-clips-balanced').textContent = formatTrainingClipsNumber(summary.balanced_training_examples);
    document.getElementById('training-clips-oversampled').textContent = formatTrainingClipsNumber(summary.oversampled_examples);
}

function renderTrainingClips(pagination) {
    const grid = document.getElementById('training-clips-grid');
    document.getElementById('training-clips-result-count').textContent = `${formatTrainingClipsNumber(pagination.total)} clips`;
    document.getElementById('training-clips-page').textContent = `Página ${pagination.page} de ${pagination.pages}`;
    document.getElementById('training-clips-prev').disabled = pagination.page <= 1;
    document.getElementById('training-clips-next').disabled = pagination.page >= pagination.pages;

    if (!trainingClipsState.clips.length) {
        grid.innerHTML = '<div class="dataset-empty">No hay clips que coincidan con los filtros.</div>';
        return;
    }

    grid.innerHTML = trainingClipsState.clips.map((clip, index) => `
        <article class="training-clip-card ${clip.filename === trainingClipsState.selectedFilename ? 'active' : ''}">
            <div class="training-clip-card-topline">
                <span class="badge ${escapeTrainingClipsHtml(clip.label)}">${escapeTrainingClipsHtml(clip.label.toUpperCase())}</span>
                <span class="training-split-badge ${escapeTrainingClipsHtml(clip.split)}">${TRAINING_SPLIT_LABELS[clip.split] || clip.split}</span>
            </div>
            <h3 title="${escapeTrainingClipsHtml(clip.filename)}">${escapeTrainingClipsHtml(clip.filename)}</h3>
            <div class="training-clip-card-meta">
                <span>${formatTrainingClipBytes(clip.size_bytes)}</span>
                ${clip.split === 'train' ? `<span>${clip.training_occurrences} usos en tabla balanceada</span>` : '<span>Reservado</span>'}
            </div>
            <button class="btn btn-secondary" type="button" data-training-clip-index="${index}">Ver clip</button>
        </article>
    `).join('');

    const selectedIndex = trainingClipsState.clips.findIndex(clip => clip.filename === trainingClipsState.selectedFilename);
    selectTrainingClip(selectedIndex >= 0 ? selectedIndex : 0);
}

function selectTrainingClip(index) {
    const clip = trainingClipsState.clips[index];
    if (!clip) return;
    trainingClipsState.selectedFilename = clip.filename;
    document.querySelectorAll('.training-clip-card').forEach((card, cardIndex) => {
        card.classList.toggle('active', cardIndex === index);
    });

    const player = document.getElementById('training-clip-player');
    if (player.dataset.filename !== clip.filename) {
        player.dataset.filename = clip.filename;
        player.src = `/api/training-clips/video/${encodeURIComponent(clip.filename)}`;
        player.load();
    }
    document.getElementById('training-clip-roi-help').textContent = 'Se muestra la entrada 224×224 completa utilizada por el modelo.';

    const label = document.getElementById('training-clip-player-label');
    label.className = `badge ${clip.label}`;
    label.textContent = clip.label.toUpperCase();
    document.getElementById('training-clip-player-split').textContent = TRAINING_SPLIT_LABELS[clip.split] || clip.split;
    document.getElementById('training-clip-player-name').textContent = clip.filename;
    const usage = clip.split === 'train'
        ? clip.training_occurrences > 0
            ? `Incluido ${clip.training_occurrences} veces en la tabla de entrenamiento balanceada.`
            : 'Pertenece a entrenamiento; no está seleccionado en la tabla balanceada.'
        : clip.split === 'validation'
            ? 'Reservado para validar y seleccionar el modelo.'
            : 'Reservado para la evaluación final del modelo.';
    document.getElementById('training-clip-player-detail').textContent = `Entrada 224×224 · ${formatTrainingClipBytes(clip.size_bytes)} · ${usage}`;
}

function formatTrainingClipsNumber(value) {
    return Number(value || 0).toLocaleString('es-AR');
}

function formatTrainingClipBytes(bytes) {
    const value = Number(bytes || 0);
    if (value < 1024 * 1024) return `${(value / 1024).toFixed(0)} KB`;
    return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function escapeTrainingClipsHtml(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}
