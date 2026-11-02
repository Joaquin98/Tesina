// Compare page JavaScript
// Mirrors the API calls from main.js exactly.

let selectedModels = [];   // array of model name strings (max 4)
let selectedVideo = null;
let availableModels = [];
let availableVideos = [];
let availableProcessedVideos = [];
const MAX_VISIBLE_VIDEOS = 36;
const CORE_COMPARISON_MODELS = [
    'best_model/best_model.keras',
    'rgb_yolo_quantile_highres_roi',
];
const RECOMMENDED_COMPARISON_MODELS = new Set([
    ...CORE_COMPARISON_MODELS,
    'rgb_yolo_historical_primera_calibrated',
    'player_geometry_fusion_calibrated',
    'light_live/resnet50__lstm__frames20/model.keras',
]);
let originalComparisonResults = {};
let currentComparisonResults = {};
let manualComparison = null;
let comparisonDuration = 0;
let initialComparisonThresholds = { confidence: 0.70, line: 0.95 };
let thresholdUpdateTimer = null;
const COMPARISON_HISTORY_DB = 'rugby-comparison-history';
const COMPARISON_HISTORY_STORE = 'comparisons';
let comparisonHistoryDbPromise = null;
let comparisonHistoryRecordsById = new Map();
let lastModelInfoTrigger = null;

const MODEL_INFO_DESCRIPTIONS = {
    no_play_tuned_macro_ensemble: 'Ensemble por votación suave entrenado con fragmentos sin jugada revisados. Su combinación fue ajustada para maximizar el F1 macro entre line, other y scrum.',
    no_play_tuned_accuracy_ensemble: 'Ensemble de dos modelos por votación suave, ajustado para obtener la mejor exactitud global con negativos sin jugada curados manualmente.',
    ensemble_top5: 'Promedia las probabilidades de los cinco mejores CNN-RNN de la búsqueda de hiperparámetros para reducir la variabilidad de un único modelo.',
    ensemble_top3: 'Promedia las probabilidades de los tres mejores CNN-RNN de la búsqueda de hiperparámetros, con menor costo que la variante Top-5.',
    'light_live/resnet50__lstm__frames20/model.keras': 'Variante liviana para detección en vivo. Resume cada ventana con 20 cuadros, extrae rasgos con ResNet-50 y modela la secuencia con una LSTM.',
    'best_model/best_model.keras': 'Modelo principal histórico: extrae rasgos visuales con ResNet-50 y usa una BiGRU para reconocer la evolución temporal de cada ventana de video.',
    'sequence_model_gru_big.keras': 'Variante experimental con rasgos ResNet-50 y una GRU de mayor capacidad para modelar el orden temporal de los cuadros.',
    'sequence_model_gru_small.keras': 'Variante compacta con rasgos ResNet-50 y una GRU pequeña; reduce costo a cambio de menor capacidad temporal.',
    'sequence_model_lstm.keras': 'Variante experimental con rasgos ResNet-50 y una LSTM como módulo temporal.',
    two_stream_precision_exact_line1_scrum7: 'Flujo orientado a revisión de alta precisión. Combina apariencia y movimiento, y ordena los clips por percentiles para revisar primero los candidatos más confiables.',
    two_stream_full_classifier: 'Clasificador completo de tres clases que combina descriptores espaciales de una CNN con estadísticas de flujo óptico.',
};

document.addEventListener('DOMContentLoaded', () => {
    ensureComparisonInterface();
    initModelInfoModal();
    initComparisonHistory();
    loadModels();
    loadVideos();
    loadProcessedVideos();
    setupSliders();
    setupTabs();
    const modelFilter = document.getElementById('compare-model-filter');
    if (modelFilter) modelFilter.value = 'all';
    document.getElementById('compare-model-search')?.addEventListener('input', () => renderModels());
    document.getElementById('compare-model-filter')?.addEventListener('change', () => renderModels());
    document.getElementById('models-comparison-grid')?.addEventListener('click', event => {
        const infoButton = event.target.closest('[data-model-info]');
        if (infoButton) {
            openModelInfo(infoButton.dataset.modelInfo, infoButton);
            return;
        }
        const selectButton = event.target.closest('.model-card-select');
        const card = selectButton?.closest('.model-card[data-model]');
        if (card) toggleModel(card.dataset.model);
    });
    document.getElementById('models-comparison-grid')?.addEventListener('contextmenu', event => {
        const card = event.target.closest('.model-card[data-model]');
        if (!card) return;
        event.preventDefault();
        openModelInfo(card.dataset.model, card.querySelector('.model-card-select'));
    });
    document.getElementById('models-comparison-grid')?.addEventListener('keydown', event => {
        if (event.key !== 'ContextMenu' && !(event.shiftKey && event.key === 'F10')) return;
        const card = event.target.closest('.model-card[data-model]');
        if (!card) return;
        event.preventDefault();
        openModelInfo(card.dataset.model, event.target);
    });
    document.getElementById('compare-video-search')?.addEventListener('input', renderVideoCollections);
    document.getElementById('compare-video-source-filter')?.addEventListener('change', renderVideoCollections);

    document.getElementById('run-comparison')?.addEventListener('click', runComparison);
    document.getElementById('select-core-comparison')?.addEventListener('click', selectCoreComparison);
    document.getElementById('compare-wide-toggle')?.addEventListener('click', toggleWideComparison);
    document.getElementById('compare-threshold-reset')?.addEventListener('click', resetComparisonThresholds);
    document.getElementById('compare-show-manual-other')?.addEventListener('change', renderComparisonTimelines);
    document.getElementById('compare-results-confidence')?.addEventListener('input', scheduleThresholdUpdate);
    document.getElementById('compare-results-line-confidence')?.addEventListener('input', scheduleThresholdUpdate);

    const comparisonVideo = document.getElementById('compare-video');
    comparisonVideo?.addEventListener('loadedmetadata', () => {
        const savedVariants = Object.values(currentComparisonResults);
        comparisonDuration = savedVariants.length && savedVariants.every(result => result.metadata?.saved_decisions_locked)
            ? getComparisonDuration(currentComparisonResults)
            : Number(comparisonVideo.duration) || comparisonDuration;
        renderComparisonTimelines();
        updateComparisonPlaybackState();
    });
    comparisonVideo?.addEventListener('timeupdate', updateComparisonPlaybackState);
});

function ensureComparisonInterface() {
    if (!document.getElementById('compare-history-section')) {
        const intro = document.querySelector('.container > .section');
        intro?.insertAdjacentHTML('afterend', `
            <section id="compare-history-section" class="section compare-history-section" aria-labelledby="compare-history-title">
                <div class="compare-history-heading">
                    <div><span class="compare-featured-kicker">Resultados guardados</span><h3 id="compare-history-title">Historial de comparaciones</h3><p>Volvé a abrir una comparación con sus modelos, video, umbrales y timelines sin procesarla otra vez.</p></div>
                    <button id="refresh-comparison-history" class="btn btn-secondary" type="button">Actualizar</button>
                </div>
                <p id="compare-history-status" class="compare-history-status" aria-live="polite"></p>
                <div id="compare-history-list" class="compare-history-list"><div class="loading">Cargando historial...</div></div>
            </section>`);
    }

    const modelGrid = document.getElementById('models-comparison-grid');
    if (modelGrid && !document.getElementById('select-core-comparison')) {
        modelGrid.insertAdjacentHTML('beforebegin', `
            <div class="compare-featured-preset">
                <div>
                    <span class="compare-featured-kicker">Comparación sugerida por el experimento</span>
                    <strong>ResNet-50 + BiGRU vs. RGB + YOLO con ROI de alta resolución</strong>
                    <small>Selecciona la rama histórica ganadora y el ensemble high-res para analizarlos sobre el mismo video.</small>
                </div>
                <button id="select-core-comparison" class="btn btn-secondary" type="button">Elegir estos dos</button>
            </div>`);
    }

    const results = document.getElementById('comparison-results');
    if (results && !document.getElementById('compare-video')) {
        results.classList.add('comparison-results-section');
        results.innerHTML = `
            <div class="compare-results-heading">
                <div><span class="compare-featured-kicker">Revisión sincronizada</span><h3>Video y timelines de la comparación</h3><p id="compare-results-subtitle">Los dos modelos se muestran sobre la misma escala temporal.</p></div>
                <button id="compare-wide-toggle" class="btn btn-secondary" type="button" aria-pressed="false">Vista amplia</button>
            </div>
            <div class="compare-review-layout">
                <div class="compare-video-panel">
                    <video id="compare-video" controls controlslist="nofullscreen" preload="metadata" playsinline>Tu navegador no puede reproducir este video.</video>
                    <div class="compare-video-context"><strong id="compare-video-title">Video comparado</strong><span id="compare-current-time">0:00 / 0:00</span><span id="compare-active-events" class="compare-active-events">Sin jugadas activas</span></div>
                </div>
                <aside class="compare-result-thresholds" aria-labelledby="compare-result-threshold-title">
                    <div class="compare-result-threshold-heading"><div><span class="compare-featured-kicker">Después de clasificar</span><h4 id="compare-result-threshold-title">Modificar umbrales</h4></div><button id="compare-threshold-reset" class="btn btn-secondary" type="button">Restablecer</button></div>
                    <label for="compare-results-confidence"><span><strong>Scrum y piso general</strong><output id="compare-results-confidence-value">0.70</output></span><input type="range" id="compare-results-confidence" min="0" max="1" step="0.01" value="0.70"></label>
                    <label for="compare-results-line-confidence"><span><strong>Umbral para line</strong><output id="compare-results-line-confidence-value">0.95</output></span><input type="range" id="compare-results-line-confidence" min="0" max="1" step="0.01" value="0.95"></label>
                    <p id="compare-threshold-status" class="compare-threshold-status">Usa las probabilidades ya calculadas; no vuelve a procesar el video.</p>
                </aside>
            </div>
            <section class="compare-timelines-panel" aria-labelledby="compare-timelines-title">
                <div class="compare-timelines-heading"><div><h4 id="compare-timelines-title">Timelines alineados</h4><p>Hacé clic en cualquier marca para ir a ese momento del video.</p></div><div class="compare-timeline-legend" aria-label="Colores del timeline"><span class="line">Line</span><span class="scrum">Scrum</span><span class="other">Other real</span><span class="pending">Pendiente</span></div></div>
                <div id="compare-model-timelines" class="compare-model-timelines"></div>
                <div id="compare-manual-timeline" class="compare-manual-timeline" hidden><div class="compare-timeline-label"><strong>3 · Etiquetado real</strong><span><span id="compare-manual-summary"></span> · <a id="compare-manual-link" href="/clip-generator">Abrir generación</a></span></div><div id="compare-manual-track" class="compare-timeline-track manual" aria-label="Timeline real del generador de clips"></div></div>
            </section>
            <details class="compare-secondary-results"><summary>Ver métricas, detecciones y gráficos</summary><div class="comparison-table-container"><table id="comparison-table" class="comparison-table"></table></div><div id="side-by-side-detections" class="side-by-side"></div><div class="charts-grid"><div class="chart-container"><canvas id="detections-chart"></canvas></div><div class="chart-container"><canvas id="confidence-chart"></canvas></div></div></details>`;
    }

    const confidenceLabel = document.querySelector('label[for="compare-confidence"]');
    if (confidenceLabel) confidenceLabel.childNodes[0].textContent = '\n                        Confianza mínima para scrum y piso general\n                        ';
    const lineDescription = document.querySelector('#compare-line-confidence + .param-description');
    if (lineDescription) lineDescription.textContent = 'Se aplica a line antes de agrupar las ventanas temporales.';

    ensureModelInfoInterface();
}

function ensureModelInfoInterface() {
    const modelGrid = document.getElementById('models-comparison-grid');
    if (modelGrid && !document.querySelector('.model-card-help')) {
        modelGrid.insertAdjacentHTML('beforebegin', '<p class="model-card-help">Hacé clic para seleccionar. Usá <strong>Ver arquitectura</strong> o botón derecho para abrir la ficha del modelo.</p>');
    }
    if (document.getElementById('compare-model-info-modal')) return;
    document.body.insertAdjacentHTML('beforeend', `
        <div id="compare-model-info-modal" class="modal compare-model-info-modal" role="dialog" aria-modal="true"
            aria-labelledby="compare-model-info-title" aria-describedby="compare-model-info-description" aria-hidden="true">
            <div class="modal-content compare-model-info-content">
                <div class="modal-header">
                    <div><span id="compare-model-info-family" class="compare-featured-kicker">Arquitectura del modelo</span><h3 id="compare-model-info-title">Detalle del modelo</h3></div>
                    <button id="compare-model-info-close" class="modal-close" type="button" aria-label="Cerrar detalle del modelo">&times;</button>
                </div>
                <div class="modal-body compare-model-info-body">
                    <p id="compare-model-info-description" class="compare-model-info-description"></p>
                    <div id="compare-model-info-metrics" class="compare-model-info-metrics"></div>
                    <section class="compare-architecture-panel" aria-labelledby="compare-architecture-title">
                        <div class="compare-architecture-heading"><h4 id="compare-architecture-title">Flujo simplificado</h4><span>Las flechas representan el recorrido de la información.</span></div>
                        <div id="compare-model-architecture" class="compare-model-architecture"></div>
                        <p id="compare-model-architecture-note" class="compare-model-architecture-note"></p>
                    </section>
                    <details class="compare-model-technical-details"><summary>Ver identificación técnica</summary><dl id="compare-model-technical-list"></dl></details>
                </div>
            </div>
        </div>`);
}

function initModelInfoModal() {
    const modal = document.getElementById('compare-model-info-modal');
    const closeButton = document.getElementById('compare-model-info-close');
    closeButton?.addEventListener('click', closeModelInfo);
    modal?.addEventListener('click', event => {
        if (event.target === modal) closeModelInfo();
    });
    document.addEventListener('keydown', event => {
        if (event.key === 'Escape' && modal?.classList.contains('active')) closeModelInfo();
    });
}

// ─── Persistent comparison history ──────────────────────────────────────────

function openComparisonHistoryDb() {
    if (!('indexedDB' in window)) {
        return Promise.reject(new Error('Este navegador no admite almacenamiento persistente.'));
    }
    if (comparisonHistoryDbPromise) return comparisonHistoryDbPromise;
    comparisonHistoryDbPromise = new Promise((resolve, reject) => {
        const request = indexedDB.open(COMPARISON_HISTORY_DB, 1);
        request.onupgradeneeded = () => {
            const db = request.result;
            if (!db.objectStoreNames.contains(COMPARISON_HISTORY_STORE)) {
                const store = db.createObjectStore(COMPARISON_HISTORY_STORE, { keyPath: 'id' });
                store.createIndex('created_at', 'created_at');
            }
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error || new Error('No se pudo abrir el historial.'));
    });
    return comparisonHistoryDbPromise;
}

async function comparisonHistoryRequest(mode, action) {
    const db = await openComparisonHistoryDb();
    return new Promise((resolve, reject) => {
        const transaction = db.transaction(COMPARISON_HISTORY_STORE, mode);
        const store = transaction.objectStore(COMPARISON_HISTORY_STORE);
        let request;
        try {
            request = action(store);
        } catch (error) {
            reject(error);
            return;
        }
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error || new Error('No se pudo acceder al historial.'));
    });
}

function initComparisonHistory() {
    document.getElementById('refresh-comparison-history')?.addEventListener('click', renderComparisonHistory);
    document.getElementById('compare-history-toggle')?.addEventListener('click', toggleComparisonHistory);
    document.getElementById('compare-history-list')?.addEventListener('click', event => {
        const openButton = event.target.closest('[data-open-comparison]');
        if (openButton) {
            openComparisonFromHistory(openButton.dataset.openComparison);
            return;
        }
        const deleteButton = event.target.closest('[data-delete-comparison]');
        if (deleteButton) deleteComparisonFromHistory(deleteButton.dataset.deleteComparison);
    });
    renderComparisonHistory();
    const requestedId = new URLSearchParams(window.location.search).get('comparison');
    if (requestedId) openComparisonFromHistory(requestedId);
}

function setComparisonHistoryCollapsed(collapsed) {
    const section = document.getElementById('compare-history-section');
    const button = document.getElementById('compare-history-toggle');
    if (!section || !button) return;
    section.classList.toggle('is-collapsed', collapsed);
    button.setAttribute('aria-expanded', String(!collapsed));
    button.textContent = collapsed ? 'Mostrar historial' : 'Ocultar';
}

function toggleComparisonHistory() {
    const section = document.getElementById('compare-history-section');
    setComparisonHistoryCollapsed(!section?.classList.contains('is-collapsed'));
}

function comparisonHistoryModelLabel(modelName) {
    const knownLabels = {
        'best_model/best_model.keras': 'ResNet-50 + BiGRU',
        rgb_yolo_quantile_highres_roi: 'RGB + YOLO high-res',
        rgb_yolo_historical_primera_calibrated: 'Histórico Primera + RGB/YOLO temporal',
    };
    return knownLabels[modelName] || modelLabel(modelName);
}

function formatComparisonDate(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return 'Fecha no disponible';
    return new Intl.DateTimeFormat('es-AR', {
        dateStyle: 'short',
        timeStyle: 'short',
    }).format(date);
}

function comparisonDetectionCount(results) {
    return Object.values(results || {}).reduce(
        (total, result) => total + (result?.detections || []).length,
        0,
    );
}

async function getComparisonHistory() {
    const browserRecords = await comparisonHistoryRequest('readonly', store => store.getAll());
    try {
        const response = await fetch('/api/comparison-history');
        if (!response.ok) throw new Error(`History API unavailable (${response.status})`);
        const data = await response.json();
        if (!data.success) throw new Error(data.error || 'History API unavailable');
        const serverRecords = data.comparisons || [];
        const serverIds = new Set(serverRecords.map(record => record.id));

        for (const browserRecord of browserRecords) {
            if (serverIds.has(browserRecord.id)) continue;
            try {
                const migrationResponse = await fetch('/api/comparison-history', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(browserRecord),
                });
                const migrationData = await migrationResponse.json();
                if (migrationResponse.ok && migrationData.success && migrationData.comparison) {
                    serverRecords.push(migrationData.comparison);
                    serverIds.add(browserRecord.id);
                }
            } catch (migrationError) {
                console.warn('Could not migrate browser comparison:', migrationError);
            }
        }

        return serverRecords
            .map(record => ({ ...record, _historySource: 'server' }))
            .sort((left, right) => String(right.created_at).localeCompare(String(left.created_at)));
    } catch (error) {
        console.info('Using browser comparison history fallback:', error.message);
        return (browserRecords || [])
            .map(record => ({ ...record, _historySource: 'browser' }))
            .sort((left, right) => String(right.created_at).localeCompare(String(left.created_at)));
    }
}

async function renderComparisonHistory() {
    const list = document.getElementById('compare-history-list');
    const status = document.getElementById('compare-history-status');
    if (!list || !status) return;
    list.innerHTML = '<div class="loading">Cargando historial...</div>';
    try {
        const records = await getComparisonHistory();
        comparisonHistoryRecordsById = new Map(records.map(record => [record.id, record]));
        status.textContent = records.length
            ? `${records.length} ${records.length === 1 ? 'comparación guardada' : 'comparaciones guardadas'}.`
            : 'Las nuevas comparaciones se guardarán automáticamente.';
        if (!records.length) {
            list.innerHTML = '<div class="compare-history-empty">Todavía no hay comparaciones guardadas. Ejecutá una y aparecerá acá.</div>';
            return;
        }
        list.innerHTML = records.map(record => {
            const filename = String(record.video_name || record.video_path || 'Video').split(/[\\/]/).pop();
            const models = (record.model_names || Object.keys(record.results || {}))
                .map(comparisonHistoryModelLabel)
                .join(' vs. ');
            const parameters = record.parameters || {};
            const detectionCount = Number(record.total_detections ?? comparisonDetectionCount(record.results));
            const manual = record.clip_generator;
            const manualSummary = manual?.available
                ? `<span class="compare-history-real"><b>Real:</b> ${Number(manual.line || 0)} line · ${Number(manual.scrum || 0)} scrum · disponible en el timeline</span>`
                : '<span class="compare-history-real unavailable">Sin etiquetado real</span>';
            return `
                <article class="compare-history-card">
                    <div class="compare-history-card-main">
                        <span class="compare-history-date">${escapeHtml(formatComparisonDate(record.created_at))}</span>
                        <strong title="${escapeHtml(filename)}">${escapeHtml(filename)}</strong>
                        <span class="compare-history-models">${escapeHtml(models)}</span>
                        <small>${parameters.saved_decisions_locked
                            ? `${detectionCount} candidatos entre variantes · decisiones guardadas · S01–S04`
                            : `${detectionCount} ${detectionCount === 1 ? 'jugada' : 'jugadas'} · umbrales ${Number(parameters.confidence_threshold ?? 0.70).toFixed(2)} / ${Number(parameters.line_confidence_threshold ?? 0.95).toFixed(2)}`}</small>
                        ${manualSummary}
                    </div>
                    <div class="compare-history-actions">
                        <button class="btn btn-primary" type="button" data-open-comparison="${escapeHtml(record.id)}">Ver comparación y timelines</button>
                        <button class="btn btn-quiet compare-history-delete" type="button" data-delete-comparison="${escapeHtml(record.id)}" aria-label="Eliminar comparación de ${escapeHtml(filename)}">Eliminar</button>
                    </div>
                </article>`;
        }).join('');
    } catch (error) {
        console.error('Error loading comparison history:', error);
        status.textContent = 'No se pudo acceder al historial persistente.';
        list.innerHTML = `<div class="error">${escapeHtml(error.message)}</div>`;
    }
}

async function saveComparisonToHistory(results, comparison, parameters) {
    const createdAt = new Date().toISOString();
    const id = `${createdAt.replace(/\D/g, '').slice(0, 14)}-${crypto.randomUUID?.() || Math.random().toString(16).slice(2)}`;
    const selectedVideoInfo = [...availableVideos, ...availableProcessedVideos].find(item => item.path === selectedVideo);
    const record = {
        id,
        created_at: createdAt,
        video_path: selectedVideo,
        video_name: selectedVideoInfo?.name || String(selectedVideo || '').split(/[\\/]/).pop(),
        model_names: [...selectedModels],
        parameters: {
            confidence_threshold: Number(parameters.confidence_threshold),
            line_confidence_threshold: Number(parameters.line_confidence_threshold),
        },
        results: cloneResults(results || {}),
        comparison: cloneResults(comparison || {}),
    };
    await comparisonHistoryRequest('readwrite', store => store.put(record));
    await renderComparisonHistory();
    return record;
}

async function openComparisonFromHistory(id) {
    const status = document.getElementById('compare-history-status');
    try {
        status.textContent = 'Abriendo comparación guardada...';
        const summary = comparisonHistoryRecordsById.get(id);
        let record;
        if (summary?._historySource === 'server' || !summary) {
            const response = await fetch(`/api/comparison-history/${encodeURIComponent(id)}`);
            const data = await response.json();
            if (response.status === 404 && !summary) {
                record = await comparisonHistoryRequest('readonly', store => store.get(id));
            } else {
                if (!response.ok || !data.success) throw new Error(data.error || 'La comparación ya no existe.');
                record = data.comparison;
            }
        } else {
            record = await comparisonHistoryRequest('readonly', store => store.get(id));
        }
        if (!record) throw new Error('La comparación ya no existe.');
        selectedVideo = record.video_path;
        selectedModels = [...(record.model_names || Object.keys(record.results || {}))];
        const parameters = record.parameters || {};
        const confidence = Number(parameters.confidence_threshold ?? 0.70);
        const line = Number(parameters.line_confidence_threshold ?? 0.95);
        document.getElementById('compare-confidence').value = confidence;
        document.getElementById('compare-confidence-value').textContent = confidence.toFixed(2);
        document.getElementById('compare-line-confidence').value = line;
        document.getElementById('compare-line-confidence-value').textContent = line.toFixed(2);
        renderModels();
        renderVideoCollections();
        updateSelectedModelCards();
        validateInputs();
        await displayResults(record.results, record.comparison, {
            saveHistory: false,
            parameters: { confidence_threshold: confidence, line_confidence_threshold: line },
        });
        status.textContent = `Comparación del ${formatComparisonDate(record.created_at)} abierta sin nueva inferencia.`;
    } catch (error) {
        console.error('Error opening comparison history:', error);
        status.textContent = `No se pudo abrir: ${error.message}`;
    }
}

async function deleteComparisonFromHistory(id) {
    if (!window.confirm('¿Eliminar esta comparación del historial?')) return;
    try {
        const summary = comparisonHistoryRecordsById.get(id);
        if (summary?._historySource === 'server') {
            const response = await fetch(`/api/comparison-history/${encodeURIComponent(id)}`, { method: 'DELETE' });
            const data = await response.json();
            if (!response.ok || !data.success) throw new Error(data.error || 'No se pudo eliminar.');
        }
        await comparisonHistoryRequest('readwrite', store => store.delete(id));
        await renderComparisonHistory();
    } catch (error) {
        document.getElementById('compare-history-status').textContent = `No se pudo eliminar: ${error.message}`;
    }
}

// ─── Tabs ─────────────────────────────────────────────────────────────────────

function setupTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            // Only toggle buttons/content within the same parent section
            const section = btn.closest('.section');
            section.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b === btn));
            section.querySelectorAll('.tab-content').forEach(c => {
                c.classList.toggle('active', c.id === `${tab}-tab`);
            });
        });
    });
}

// ─── Sliders ──────────────────────────────────────────────────────────────────

function setupSliders() {
    [
        { id: 'compare-confidence', valueId: 'compare-confidence-value', fmt: v => parseFloat(v).toFixed(2) },
        { id: 'compare-line-confidence', valueId: 'compare-line-confidence-value', fmt: v => parseFloat(v).toFixed(2) },
    ].forEach(({ id, valueId, fmt }) => {
        const slider = document.getElementById(id);
        const display = document.getElementById(valueId);
        slider?.addEventListener('input', e => {
            display.textContent = fmt(e.target.value);
        });
    });
}

// ─── Models ──────────────────────────────────────────────────────────────────

async function loadModels() {
    const grid = document.getElementById('models-comparison-grid');
    grid.innerHTML = '<div class="loading">Cargando modelos...</div>';

    try {
        const resp = await fetch('/api/models');
        const data = await resp.json();

        if (!data.success) {
            grid.innerHTML = `<div class="error">No se pudieron cargar los modelos: ${data.error}</div>`;
            return;
        }

        availableModels = data.models || [];
        updateCoreComparisonPreset();
        renderModels();
    } catch (err) {
        console.error('Error loading models:', err);
        grid.innerHTML = '<div class="error">No se pudieron cargar los modelos.</div>';
    }
}

function modelIsRecommended(model) {
    return model.recommended === true || RECOMMENDED_COMPARISON_MODELS.has(model.name);
}

function modelDisplayName(model) {
    const directTranslations = {
        'No-play Tuned Ensemble (best macro-F1)': 'Ensemble ajustado sin jugada · mejor macro-F1',
        'No-play Tuned Ensemble (best accuracy)': 'Ensemble ajustado sin jugada · mejor exactitud',
        'Ensemble Top-5 (soft voting)': 'Ensemble Top-5 · votación suave',
        'Ensemble Top-3 (soft voting)': 'Ensemble Top-3 · votación suave',
        'Two-Stream Precision Review (line1 + scrum7)': 'Two-Stream para revisión de alta precisión',
        'Two-Stream Full Classifier (strict production)': 'Two-Stream completo · producción estricta',
    };
    if (model.name === 'rgb_yolo_historical_primera_calibrated') return 'Mejor modelo · Histórico Primera + RGB/YOLO temporal calibrado';
    if (model.name === 'rgb_yolo_quantile_highres_roi') return 'RGB + YOLO con ROI de alta resolución';
    if (model.name === 'rgb_yolo_quantile_fusion_all') return 'RGB + YOLO con ROI 224 (referencia)';
    if (model.name === 'player_geometry_fusion_calibrated') return 'Modelo recomendado (RGB + pose + geometría, 83.7%)';
    if (model.name === 'm1_h2_d2_cross_fold_ensemble') return 'M1-H2-D2 · ensemble de 6 folds';
    if (model.name === 'best_model/best_model.keras') return 'Modelo principal (ResNet-50 + BiGRU, 81.3%)';
    if (model.name === 'light_live/resnet50__lstm__frames20/model.keras') return 'Modelo ligero para CPU (ResNet-50 + LSTM, 20 cuadros)';
    if (model.name?.includes('human_review_retrain_860')) return 'Reentrenamiento con 860 revisiones (experimental)';
    if (model.name?.includes('best_model_focused')) return 'Mejor modelo de la búsqueda focalizada (etapa 2)';
    if (model.display_name?.startsWith('InceptionV3-GRU')) return model.display_name.replace('InceptionV3-GRU', 'ResNet-50 + modelo temporal');
    return directTranslations[model.display_name] || model.display_name || model.name;
}

function modelLabel(modelName) {
    const model = availableModels.find(item => item.name === modelName);
    return model ? modelDisplayName(model) : modelName;
}

function modelInfoDescription(model) {
    if (MODEL_INFO_DESCRIPTIONS[model.name]) return MODEL_INFO_DESCRIPTIONS[model.name];
    if (model.name?.includes('best_model_focused')) {
        return 'Mejor configuración de la segunda búsqueda focalizada. Usa ResNet-50 para la apariencia y una BiGRU de 128 unidades para la secuencia temporal.';
    }
    if (model.name?.includes('human_review_retrain_860')) {
        return 'Reentrenamiento experimental del CNN-RNN con 860 clips revisados manualmente. Aumenta sensibilidad a line y scrum, con menor recall para other.';
    }
    return model.description || 'Modelo disponible para clasificar ventanas de video en line, scrum u other.';
}

function sequenceModuleName(model) {
    const name = String(model.name || '').toLowerCase();
    if (name.includes('lstm')) return 'LSTM';
    if (name.includes('gru_big')) return 'GRU grande';
    if (name.includes('gru_small')) return 'GRU pequeña';
    if (name.includes('u128')) return 'BiGRU · 128u';
    return 'BiGRU';
}

function modelArchitectureDescriptor(model) {
    const name = String(model.name || '');
    const type = String(model.model_type || '');
    const stage = (...nodes) => ({ nodes });

    if (type === 'tuned_no_play_ensemble') {
        return {
            family: 'Ensemble por votación suave',
            stages: [stage('Ventana de video'), stage('CNN-RNN A', 'CNN-RNN B'), stage('Promedio de probabilidades'), stage('Ajuste no-play'), stage('Line · Other · Scrum')],
            note: name.includes('macro') ? 'Los pesos priorizan el equilibrio entre las tres clases mediante F1 macro.' : 'Los pesos priorizan la cantidad total de predicciones correctas.',
        };
    }
    if (type === 'ensemble') {
        const memberCount = name.includes('top5') ? 5 : 3;
        return {
            family: `Ensemble Top-${memberCount}`,
            stages: [stage('Ventana de video'), stage('CNN-RNN 1', 'CNN-RNN 2', `… CNN-RNN ${memberCount}`), stage('Promedio softmax'), stage('Line · Other · Scrum')],
            note: `Integra ${memberCount} modelos completos; el diagrama resume sus ramas paralelas.`,
        };
    }
    if (type === 'historical_yolo_residual') {
        return {
            family: 'CNN-RNN con corrección YOLO residual',
            stages: [stage('Video · 7 s'), stage('ResNet-50 + BiGRU', 'YOLO pose · 200 cuadros'), stage('Compuerta de confianza < 0,85'), stage('Corrección residual'), stage('Clase final')],
            note: 'YOLO interviene solamente cuando el modelo histórico no alcanza la confianza definida.',
        };
    }
    if (type === 'player_geometry_fusion') {
        return {
            family: 'Fusión RGB, pose y geometría',
            stages: [stage('Video · 7 s'), stage('ResNet-50 + BiGRU', 'YOLO11n-pose', 'Geometría temporal'), stage('Media geométrica'), stage('Calibración de clases'), stage('Clase final')],
            note: 'Las tres ramas describen apariencia, postura de jugadores y distribución espacial a lo largo del tiempo.',
        };
    }
    if (type === 'm1_h2_d2_cross_fold') {
        return {
            family: 'M1-H2-D2 multimodal',
            stages: [stage('Video · 7 s'), stage('ResNet-50 + BiGRU RGB', 'YOLO11n-pose + BiGRU', 'Geometría · Random Forest'), stage('Fusión por fold'), stage('Promedio de 6 folds'), stage('Line · Other · Scrum')],
            note: 'Los seis folds comparten la extracción RGB/pose y aportan probabilidades con igual peso. La política web de ensemble todavía es exploratoria.',
        };
    }
    if (type === 'rgb_yolo_historical_primera_calibrated') {
        return {
            family: 'Fusión histórica calibrada',
            stages: [stage('Video · 7 s'), stage('Histórico afinado', 'Dinámica RGB', 'YOLO high-res'), stage('Evidencia temporal'), stage('Fusión calibrada por clase'), stage('Eventos line / scrum')],
            note: 'Scrum conserva la mezcla histórica/RGB-YOLO; line incorpora una calibración y evidencia temporal específicas.',
        };
    }
    if (type === 'rgb_yolo_temporal_event_verifier') {
        return {
            family: 'Clasificador high-res con verificador temporal',
            stages: [stage('Video · 7 s'), stage('RGB + YOLO high-res'), stage('TCN + BiGRU + atención'), stage('Verificador line · 208 variables'), stage('Eventos verificados')],
            note: 'Primero clasifica ventanas; después revisa el evento completo para descartar falsos positivos de line.',
        };
    }
    if (type === 'rgb_yolo_quantile_highres') {
        return {
            family: 'Ensemble RGB + YOLO high-res',
            stages: [stage('Video en resolución fuente'), stage('ResNet-50 + BiGRU', 'Dinámica RGB · 224', 'ROI fuente → YOLO 640'), stage('Transformación cuantílica'), stage('Fusión logística'), stage('Line · Other · Scrum')],
            note: 'YOLO usa el recorte equivalente directamente desde la fuente; la rama RGB conserva su normalización propia.',
        };
    }
    if (type === 'rgb_yolo_quantile_fusion') {
        return {
            family: 'Fusión cuantílica RGB + YOLO',
            stages: [stage('Video · 7 s'), stage('Histórico ResNet-BiGRU', '704 variables RGB', 'YOLO E04 · 707', 'YOLO E06 · 275'), stage('Transformación cuantílica'), stage('Fusión 40% / 60%'), stage('Clase final')],
            note: 'Las variables de las ramas se normalizan por cuantiles antes de aplicar la combinación aprendida.',
        };
    }
    if (type === 'meta_correction') {
        return {
            family: 'CNN-RNN con corrector few-shot',
            stages: [stage('Video · 7 s'), stage('CNN-RNN histórico'), stage('Probabilidades base'), stage('Calibrador del mismo partido'), stage('Probabilidades corregidas')],
            note: 'El corrector depende de clips etiquetados del mismo video; debe recalibrarse para otro partido.',
        };
    }
    if (type === 'two_stream_review_workflow') {
        return {
            family: 'Two-Stream para revisión',
            stages: [stage('Video'), stage('Rama espacial CNN', 'Flujo óptico'), stage('Fusión de scores'), stage('Ranking por percentil'), stage('Cola de revisión')],
            note: 'La salida principal es un orden de revisión de alta precisión, no una clasificación aislada convencional.',
        };
    }
    if (type === 'two_stream_full_classifier') {
        return {
            family: 'Two-Stream de tres clases',
            stages: [stage('Video'), stage('Descriptores CNN', 'Estadísticas de flujo óptico'), stage('Fusión espacial-temporal'), stage('Clasificador'), stage('Line · Other · Scrum')],
            note: 'La rama espacial describe apariencia; el flujo óptico resume el movimiento entre cuadros.',
        };
    }
    if (type === 'inceptionv3_gru_clahe') {
        return {
            family: 'CNN-RNN con realce CLAHE',
            stages: [stage('Video'), stage('Escala + recorte'), stage('CLAHE'), stage('ResNet-50'), stage('BiGRU'), stage('Softmax')],
            note: 'CLAHE realza el contraste local antes de extraer los rasgos visuales.',
        };
    }
    if (type === 'inceptionv3_gru') {
        const stages = [stage('Video · 7 s'), stage(name.includes('frames20') ? '20 cuadros' : 'Muestreo temporal'), stage('Escala + recorte'), stage('ResNet-50'), stage(sequenceModuleName(model)), stage('Softmax')];
        if (name.includes('calibrated_alpha')) stages.splice(stages.length - 1, 0, stage('Calibración line ×2,95'));
        return {
            family: `CNN-RNN · ${sequenceModuleName(model)}`,
            stages,
            note: name.includes('class_weights')
                ? 'Los pesos de clase modifican el entrenamiento; la arquitectura de inferencia permanece igual.'
                : 'ResNet-50 procesa la apariencia de cada cuadro y el módulo recurrente integra su orden temporal.',
        };
    }
    return {
        family: 'Clasificador de video',
        stages: [stage('Video'), stage('Preprocesamiento'), stage('Extractor de variables'), stage('Clasificador'), stage('Clase final')],
        note: 'Diagrama general construido a partir del tipo técnico informado por el catálogo.',
    };
}

function renderModelArchitecture(model, architecture) {
    const container = document.getElementById('compare-model-architecture');
    const accessibleFlow = architecture.stages
        .map(item => item.nodes.join(' más '))
        .join(', luego ');
    container.setAttribute('role', 'img');
    container.setAttribute('aria-label', `Arquitectura simplificada de ${modelDisplayName(model)}: ${accessibleFlow}.`);
    container.innerHTML = architecture.stages.map((item, index) => `
        ${index ? '<span class="compare-architecture-arrow" aria-hidden="true">→</span>' : ''}
        <div class="compare-architecture-stage ${item.nodes.length > 1 ? 'parallel' : ''}">
            ${item.nodes.map(node => `<div class="compare-architecture-node">${escapeHtml(node)}</div>`).join('')}
        </div>
    `).join('');
}

function openModelInfo(modelName, trigger = null) {
    const model = availableModels.find(item => item.name === modelName);
    const modal = document.getElementById('compare-model-info-modal');
    if (!model || !modal) return;
    const architecture = modelArchitectureDescriptor(model);
    const reportedAccuracy = Number(model.metrics?.test_accuracy ?? model.metrics?.accuracy);
    const metrics = [
        ['Familia', architecture.family],
        ['Tamaño', Number(model.size_mb) > 0 ? `${Number(model.size_mb).toFixed(1)} MB` : 'No informado'],
        ['Exactitud', Number.isFinite(reportedAccuracy) ? `${(reportedAccuracy * 100).toFixed(1)}%` : 'No informada'],
    ];

    document.getElementById('compare-model-info-family').textContent = architecture.family;
    document.getElementById('compare-model-info-title').textContent = modelDisplayName(model);
    document.getElementById('compare-model-info-description').textContent = modelInfoDescription(model);
    document.getElementById('compare-model-info-metrics').innerHTML = metrics.map(([label, value]) => `
        <div class="compare-model-info-metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>
    `).join('');
    document.getElementById('compare-model-architecture-note').textContent = architecture.note;
    document.getElementById('compare-model-technical-list').innerHTML = `
        <dt>Identificador</dt><dd>${escapeHtml(model.name)}</dd>
        <dt>Tipo</dt><dd>${escapeHtml(model.model_type || 'No informado')}</dd>
        <dt>Recomendado</dt><dd>${modelIsRecommended(model) ? 'Sí' : 'No'}</dd>
    `;
    renderModelArchitecture(model, architecture);

    lastModelInfoTrigger = trigger || document.activeElement;
    modal.classList.add('active');
    modal.setAttribute('aria-hidden', 'false');
    document.body.classList.add('compare-model-modal-open');
    document.getElementById('compare-model-info-close')?.focus();
}

function closeModelInfo() {
    const modal = document.getElementById('compare-model-info-modal');
    if (!modal?.classList.contains('active')) return;
    modal.classList.remove('active');
    modal.setAttribute('aria-hidden', 'true');
    document.body.classList.remove('compare-model-modal-open');
    if (lastModelInfoTrigger?.isConnected) lastModelInfoTrigger.focus();
    lastModelInfoTrigger = null;
}

function orderedModelNames(results) {
    const keys = Object.keys(results || {});
    return [
        ...selectedModels.filter(name => keys.includes(name)),
        ...keys.filter(name => !selectedModels.includes(name)),
    ];
}

function timelineModelDescriptor(modelName, result) {
    const architecture = String(result?.fusion_metadata?.architecture || '').toLowerCase();
    if (modelName === 'rgb_yolo_quantile_highres_roi' || architecture.includes('yolo')) {
        return 'Ensemble con YOLO high-res';
    }
    if (modelName === 'best_model/best_model.keras') return 'Modelo base, sin YOLO';
    return result?.model_name || '';
}

function updateCoreComparisonPreset() {
    const button = document.getElementById('select-core-comparison');
    if (!button) return;
    const missing = CORE_COMPARISON_MODELS.filter(name => !availableModels.some(model => model.name === name));
    button.disabled = missing.length > 0;
    button.title = missing.length
        ? `Falta cargar: ${missing.join(', ')}`
        : 'Seleccionar ResNet-50 + BiGRU y RGB + YOLO high-res';
}

function selectCoreComparison() {
    const available = CORE_COMPARISON_MODELS.filter(name => availableModels.some(model => model.name === name));
    if (available.length !== CORE_COMPARISON_MODELS.length) return;
    selectedModels = [...available];
    renderModels();
    updateSelectedModelCards();
    validateInputs();
}

function renderModels(models = availableModels) {
    const grid = document.getElementById('models-comparison-grid');
    const query = (document.getElementById('compare-model-search')?.value || '').trim().toLowerCase();
    const filter = document.getElementById('compare-model-filter')?.value || 'all';
    const filteredModels = models.filter(model => {
        const haystack = `${model.name} ${model.display_name || ''} ${model.description || ''} ${model.model_type || ''}`.toLowerCase();
        const matchesFilter = filter === 'all'
            || (filter === 'recommended' && modelIsRecommended(model))
            || (filter === 'live' && (haystack.includes('light') || haystack.includes('frames20')))
            || (filter === 'ensemble' && haystack.includes('ensemble'));
        return matchesFilter && (!query || haystack.includes(query));
    });
    const countEl = document.getElementById('compare-model-results-count');
    if (countEl) {
        countEl.textContent = filter === 'recommended' && models.length > filteredModels.length
            ? `${filteredModels.length} recomendado${filteredModels.length === 1 ? '' : 's'} · ${models.length} disponibles`
            : `${filteredModels.length} de ${models.length}`;
    }

    if (filteredModels.length === 0) {
        grid.innerHTML = '<div class="loading">No hay modelos que coincidan con la búsqueda.</div>';
        return;
    }

    grid.innerHTML = filteredModels.map(model => {
        const selectedIndex = selectedModels.indexOf(model.name);
        const isSelected = selectedIndex !== -1;
        return `
        <article class="model-card ${modelIsRecommended(model) ? 'recommended' : ''} ${isSelected ? 'compare-selected' : ''}"
            data-model="${escapeHtml(model.name)}"${isSelected ? ` data-index="${selectedIndex + 1}"` : ''}>
            <button type="button" class="model-card-select" aria-pressed="${isSelected}"
                aria-label="${isSelected ? 'Quitar' : 'Seleccionar'} ${escapeHtml(modelDisplayName(model))}">
                ${modelIsRecommended(model) ? '<span class="recommendation-badge">Recomendado</span>' : ''}
                <h3>${escapeHtml(modelDisplayName(model))}</h3>
                <span class="model-type">${escapeHtml(model.model_type)}</span>
                ${model.description ? `<p class="model-description">${escapeHtml(model.description)}</p>` : ''}
                <div class="model-metrics">
                    <div class="metric">
                        <div class="metric-label">Tamaño</div>
                        <div class="metric-value">${model.size_mb > 0 ? model.size_mb.toFixed(1) + ' MB' : 'N/A'}</div>
                    </div>
                    ${model.metrics && model.metrics.test_accuracy ? `
                        <div class="metric">
                            <div class="metric-label">Exactitud</div>
                            <div class="metric-value">${(model.metrics.test_accuracy * 100).toFixed(1)}%</div>
                        </div>
                    ` : ''}
                    ${model.metrics && model.metrics.accuracy && !model.metrics.test_accuracy ? `
                        <div class="metric">
                            <div class="metric-label">Exactitud</div>
                            <div class="metric-value">${(model.metrics.accuracy * 100).toFixed(1)}%</div>
                        </div>
                    ` : ''}
                </div>
            </button>
            <button type="button" class="model-info-button" data-model-info="${escapeHtml(model.name)}"
                aria-label="Ver descripción y arquitectura de ${escapeHtml(modelDisplayName(model))}">Ver arquitectura →</button>
        </article>
    `;
    }).join('');
}

function toggleModel(modelName) {
    const idx = selectedModels.indexOf(modelName);

    if (idx === -1) {
        if (selectedModels.length >= 4) {
            alert('Podés comparar como máximo 4 modelos a la vez.');
            return;
        }
        selectedModels.push(modelName);
    } else {
        selectedModels.splice(idx, 1);
    }

    updateSelectedModelCards();
    validateInputs();
}

function updateSelectedModelCards() {
    document.querySelectorAll('.model-card').forEach(card => {
        const i = selectedModels.indexOf(card.dataset.model);
        card.classList.toggle('compare-selected', i !== -1);
        const selectButton = card.querySelector('.model-card-select');
        selectButton?.setAttribute('aria-pressed', i !== -1 ? 'true' : 'false');
        selectButton?.setAttribute('aria-label', `${i !== -1 ? 'Quitar' : 'Seleccionar'} ${modelLabel(card.dataset.model)}`);
        if (i !== -1) {
            card.setAttribute('data-index', i + 1);
        } else {
            card.removeAttribute('data-index');
        }
    });

    document.getElementById('models-selected-count').textContent =
        `${selectedModels.length} modelo${selectedModels.length !== 1 ? 's' : ''} seleccionado${selectedModels.length !== 1 ? 's' : ''}`;
}

// ─── Videos ──────────────────────────────────────────────────────────────────

async function loadVideos() {
    const grid = document.getElementById('videos-comparison-grid');
    grid.innerHTML = '<div class="loading">Cargando videos...</div>';

    try {
        // Same endpoint as main page
        const resp = await fetch('/api/videos');
        const data = await resp.json();

        if (!data.success) {
            grid.innerHTML = `<div class="error">No se pudieron cargar los videos: ${data.error}</div>`;
            return;
        }

        availableVideos = data.videos || [];
        renderVideos();
    } catch (err) {
        console.error('Error loading videos:', err);
        grid.innerHTML = '<div class="error">No se pudieron cargar los videos.</div>';
    }
}

function filterVideos(videos) {
    const query = (document.getElementById('compare-video-search')?.value || '').trim().toLowerCase();
    const source = document.getElementById('compare-video-source-filter')?.value || 'all';
    return videos.filter(video => {
        const haystack = `${video.name || ''} ${video.path || ''}`.toLowerCase();
        const isLocal = !haystack.includes('youtube')
            && !haystack.includes('web_downloads')
            && !haystack.includes('final_tests');
        return (!query || haystack.includes(query)) && (source === 'all' || (source === 'local' ? isLocal : haystack.includes(source)));
    });
}

function renderVideoCollections() {
    renderVideos();
    renderProcessedVideos();
}

function formatVideoDuration(duration) {
    const totalSeconds = Math.max(0, Math.round(Number(duration) || 0));
    if (!totalSeconds) return 'No disponible';
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    return hours > 0
        ? `${hours}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
        : `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

function renderVideos(videos = availableVideos) {
    const grid = document.getElementById('videos-comparison-grid');
    const filteredVideos = filterVideos(videos);
    const countEl = document.getElementById('compare-video-results-count');
    if (countEl) countEl.textContent = filteredVideos.length > MAX_VISIBLE_VIDEOS
        ? `Mostrando ${MAX_VISIBLE_VIDEOS} de ${filteredVideos.length}; refiná la búsqueda`
        : `${filteredVideos.length} de ${videos.length}`;

    if (filteredVideos.length === 0) {
        grid.innerHTML = '<div class="loading">No hay videos que coincidan con la búsqueda.</div>';
        return;
    }

    grid.innerHTML = filteredVideos.slice(0, MAX_VISIBLE_VIDEOS).map(v => {
        const nameParts = v.name.replace(/\\/g, '/').split('/');
        const filename = nameParts[nameParts.length - 1];
        const subfolder = nameParts.length > 1 ? nameParts.slice(0, -1).join(' / ') : '';
        const escapedPath = v.path.replace(/\\/g, '\\\\').replace(/'/g, "\\'");

        return `
            <div class="video-card" data-video="${v.path}">
                <button type="button" class="video-card-body" aria-pressed="${selectedVideo === v.path}" onclick="selectVideo('${escapedPath}')">
                    <h4>${filename}</h4>
                    ${subfolder ? `<div class="video-subfolder">${subfolder}</div>` : ''}
                    <div class="video-size">Duración: ${formatVideoDuration(v.duration)} · ${(v.size / (1024 * 1024)).toFixed(1)} MB</div>
                </button>
            </div>
        `;
    }).join('');
}

function selectVideo(videoPath) {
    selectedVideo = videoPath;
    document.querySelectorAll('.video-card').forEach(card => {
        card.classList.toggle('selected', card.dataset.video === videoPath);
        card.querySelector('.video-card-body')?.setAttribute('aria-pressed', card.dataset.video === videoPath ? 'true' : 'false');
    });
    validateInputs();
}

// ─── Processed Videos ─────────────────────────────────────────────────────────

async function loadProcessedVideos() {
    const grid = document.getElementById('processed-comparison-grid');
    if (!grid) return;
    grid.innerHTML = '<div class="loading">Cargando videos procesados...</div>';

    try {
        const resp = await fetch('/api/processed_videos');
        const data = await resp.json();
        if (!data.success) {
            grid.innerHTML = `<div class="error">Error: ${data.error}</div>`;
            return;
        }
        availableProcessedVideos = data.processed_videos || [];
        renderProcessedVideos();
    } catch (err) {
        console.error('Error loading processed videos:', err);
        grid.innerHTML = '<div class="error">No se pudieron cargar los videos procesados.</div>';
    }
}

function renderProcessedVideos(videos = availableProcessedVideos) {
    const grid = document.getElementById('processed-comparison-grid');
    if (!grid) return;

    const filteredVideos = filterVideos(videos);
    if (filteredVideos.length === 0) {
        grid.innerHTML = '<div class="loading">No hay videos procesados que coincidan.</div>';
        return;
    }

    grid.innerHTML = filteredVideos.slice(0, MAX_VISIBLE_VIDEOS).map(v => {
        const escapedPath = v.path.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
        return `
            <div class="video-card" data-video="${v.path}">
                <button type="button" class="video-card-body" aria-pressed="${selectedVideo === v.path}" onclick="selectVideo('${escapedPath}')">
                    <h4>${v.name}</h4>
                    <div class="video-size">Duración: ${formatVideoDuration(v.duration)} · ${(v.size / (1024 * 1024)).toFixed(1)} MB</div>
                </button>
            </div>
        `;
    }).join('');
}

// ─── Validation ──────────────────────────────────────────────────────────────

function validateInputs() {
    const btn = document.getElementById('run-comparison');
    const msg = document.getElementById('compare-validation');

    if (selectedModels.some(name => currentComparisonResults[name]?.metadata?.saved_decisions_locked)) {
        btn.disabled = true;
        msg.textContent = 'Variantes guardadas: elegí modelos del catálogo para ejecutar otra comparación.';
        return false;
    }

    if (selectedModels.length < 2) {
        btn.disabled = true;
        msg.textContent = 'Elegí al menos 2 modelos para comparar';
        return false;
    }

    if (!selectedVideo) {
        btn.disabled = true;
        msg.textContent = 'Elegí un video';
        return false;
    }

    btn.disabled = false;
    msg.textContent = 'Listo para comparar';
    return true;
}

// ─── Run Comparison ───────────────────────────────────────────────────────────

async function runComparison() {
    if (!validateInputs()) return;

    const btn = document.getElementById('run-comparison');
    btn.disabled = true;
    btn.textContent = 'Procesando comparación...';

    const parameters = {
        confidence_threshold: parseFloat(document.getElementById('compare-confidence').value),
        line_confidence_threshold: parseFloat(document.getElementById('compare-line-confidence').value),
    };

    try {
        const resp = await fetch('/api/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_names: selectedModels,
                video_path: selectedVideo,
                parameters,
            }),
        });

        const data = await resp.json();
        if (!data.success) throw new Error(data.error || 'Comparison failed');

        await displayResults(data.results, data.comparison, {
            saveHistory: true,
            parameters,
            serverHistoryId: data.comparison_id,
        });

    } catch (err) {
        alert('La comparación falló: ' + err.message);
        console.error('Comparison error:', err);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Ejecutar comparación';
        validateInputs();
    }
}

// ─── Display Results ──────────────────────────────────────────────────────────

function formatTime(s) {
    const total = Math.max(0, Math.floor(Number(s) || 0));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const sec = total % 60;
    return h > 0
        ? `${h}:${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`
        : `${m}:${sec.toString().padStart(2, '0')}`;
}

async function displayResults(results, comparison, options = {}) {
    const section = document.getElementById('comparison-results');
    section.style.display = 'block';
    setComparisonHistoryCollapsed(true);
    originalComparisonResults = results || {};
    currentComparisonResults = cloneResults(originalComparisonResults);
    manualComparison = null;
    comparisonDuration = getComparisonDuration(currentComparisonResults);

    const parameters = options.parameters || {};
    initialComparisonThresholds = {
        confidence: Number(parameters.confidence_threshold ?? document.getElementById('compare-confidence').value),
        line: Number(parameters.line_confidence_threshold ?? document.getElementById('compare-line-confidence').value),
    };
    syncResultThresholdControls(initialComparisonThresholds.confidence, initialComparisonThresholds.line);

    const savedVariants = Object.values(results || {});
    const fixedDecisions = savedVariants.length > 0
        && savedVariants.every(result => result.metadata?.saved_decisions_locked);
    ['compare-results-confidence', 'compare-results-line-confidence', 'compare-threshold-reset'].forEach(id => {
        const control = document.getElementById(id);
        if (control) control.disabled = fixedDecisions;
    });
    if (fixedDecisions) {
        document.getElementById('compare-threshold-status').textContent =
            (savedVariants[0].metadata.coverage_note || 'Se muestran las decisiones guardadas de cada variante.')
            + ' Cada fila conserva su regla guardada; scrum permanece igual a argmax.';
    }
    validateInputs();

    const video = document.getElementById('compare-video');
    const nextVideoSource = `/api/video-by-path?video_path=${encodeURIComponent(selectedVideo)}`;
    if (video.getAttribute('src') !== nextVideoSource) {
        video.src = nextVideoSource;
        video.load();
    }
    const selectedVideoInfo = [...availableVideos, ...availableProcessedVideos].find(item => item.path === selectedVideo);
    const filename = (selectedVideoInfo?.name || selectedVideo || 'Video comparado').split(/[\\/]/).pop();
    document.getElementById('compare-video-title').textContent = filename;
    document.getElementById('compare-results-subtitle').textContent =
        fixedDecisions
            ? `${Object.keys(results).length} variantes con decisiones guardadas sobre ${filename}`
            : `${Object.keys(results).length} modelos sobre ${filename}`;

    renderComparisonDetails();
    await loadManualComparison();
    renderComparisonTimelines();
    updateComparisonPlaybackState();
    if (options.saveHistory) {
        try {
            if (options.serverHistoryId) {
                await renderComparisonHistory();
            } else {
                await saveComparisonToHistory(results, comparison, {
                    confidence_threshold: initialComparisonThresholds.confidence,
                    line_confidence_threshold: initialComparisonThresholds.line,
                });
            }
            document.getElementById('compare-threshold-status').textContent =
                'Comparación terminada y guardada automáticamente en el historial.';
        } catch (error) {
            console.error('Error saving comparison history:', error);
            document.getElementById('compare-threshold-status').textContent =
                `La comparación terminó, pero no pudo guardarse en el historial: ${error.message}`;
        }
    }
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderTable(results, comparison) {
    const table = document.getElementById('comparison-table');
    const models = orderedModelNames(results);

    const rows = [
        ['Detecciones totales', m => results[m].detections.length],
        ['Scrums',           m => results[m].detections.filter(d => d.play_type === 'scrum').length],
        ['Lines',            m => results[m].detections.filter(d => d.play_type === 'line').length],
        ['Other',            m => results[m].detections.filter(d => d.play_type === 'other').length],
        ['Confianza media',  m => {
            const dets = results[m].detections;
            if (!dets.length) return '—';
            const avg = dets.reduce((s, d) => s + d.confidence, 0) / dets.length;
            return (avg * 100).toFixed(1) + '%';
        }],
        ['Duración media (s)', m => {
            const dets = results[m].detections;
            if (!dets.length) return '—';
            const avg = dets.reduce((s, d) => s + (d.duration || d.end_time - d.start_time), 0) / dets.length;
            return avg.toFixed(1);
        }],
    ];

    table.innerHTML = `
        <thead>
            <tr>
                <th>Métrica</th>
                ${models.map(m => `<th>${escapeHtml(modelLabel(m))}</th>`).join('')}
            </tr>
        </thead>
        <tbody>
            ${rows.map(([label, fn]) => `
                <tr>
                    <td><strong>${label}</strong></td>
                    ${models.map(m => `<td>${fn(m)}</td>`).join('')}
                </tr>
            `).join('')}
        </tbody>
    `;
}

function renderSideBySide(results) {
    const container = document.getElementById('side-by-side-detections');
    const models = orderedModelNames(results);

    container.style.gridTemplateColumns = `repeat(${models.length}, 1fr)`;

    container.innerHTML = models.map(model => {
        const dets = results[model].detections;
        const items = dets.length === 0
            ? '<p style="color:var(--text-secondary);padding:0.5rem;">Sin detecciones</p>'
            : dets.map(d => `
                <div class="compare-detection-item">
                    <span class="badge ${d.play_type}">${d.play_type.toUpperCase()}</span>
                    <span>${formatTime(d.start_time)} – ${formatTime(d.end_time)}</span>
                    <span>${(d.confidence * 100).toFixed(0)}%</span>
                </div>
            `).join('');

        return `
            <div class="model-column">
                <div class="model-column-header">${escapeHtml(modelLabel(model))}</div>
                <div class="model-column-body">${items}</div>
            </div>
        `;
    }).join('');
}

function renderCharts(results) {
    if (typeof Chart === 'undefined') return;
    const models = orderedModelNames(results);
    const colors = ['#2563eb', '#10b981', '#ef4444', '#f59e0b'];

    // Chart 1 – detections by type per model
    const ctx1 = document.getElementById('detections-chart');
    if (ctx1) {
        if (ctx1._chartInstance) ctx1._chartInstance.destroy();
        ctx1._chartInstance = new Chart(ctx1, {
            type: 'bar',
            data: {
                labels: ['Scrum', 'Line', 'Other'],
                datasets: models.map((m, i) => ({
                    label: modelLabel(m),
                    data: [
                        results[m].detections.filter(d => d.play_type === 'scrum').length,
                        results[m].detections.filter(d => d.play_type === 'line').length,
                        results[m].detections.filter(d => d.play_type === 'other').length,
                    ],
                    backgroundColor: colors[i] + 'cc',
                    borderColor: colors[i],
                    borderWidth: 1,
                })),
            },
            options: {
                responsive: true,
                plugins: { legend: { position: 'top' }, title: { display: true, text: 'Detecciones por tipo' } },
            },
        });
    }

    // Chart 2 – avg confidence per model
    const ctx2 = document.getElementById('confidence-chart');
    if (ctx2) {
        if (ctx2._chartInstance) ctx2._chartInstance.destroy();
        ctx2._chartInstance = new Chart(ctx2, {
            type: 'bar',
            data: {
                labels: models.map(modelLabel),
                datasets: [{
                    label: 'Confianza media',
                    data: models.map(m => {
                        const dets = results[m].detections;
                        if (!dets.length) return 0;
                        return (dets.reduce((s, d) => s + d.confidence, 0) / dets.length * 100).toFixed(1);
                    }),
                    backgroundColor: colors,
                    borderWidth: 1,
                }],
            },
            options: {
                responsive: true,
                scales: { y: { min: 0, max: 100, title: { display: true, text: 'Confianza (%)' } } },
                plugins: { legend: { display: false }, title: { display: true, text: 'Confianza media por modelo' } },
            },
        });
    }
}

// ─── Video-first comparison review ───────────────────────────────────────────

function escapeHtml(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function cloneResults(results) {
    if (typeof structuredClone === 'function') return structuredClone(results);
    return JSON.parse(JSON.stringify(results));
}

function getComparisonDuration(results) {
    const saved = Object.values(results || {});
    if (saved.length && saved.every(result => result.metadata?.saved_decisions_locked)) {
        return Math.max(0, ...saved.map(result => Number(result.metadata.evaluation_coverage_seconds || result.metadata.duration) || 0));
    }
    const durations = Object.values(results || {}).map(result => Number(result?.metadata?.duration) || 0);
    const selectedVideoInfo = [...availableVideos, ...availableProcessedVideos].find(item => item.path === selectedVideo);
    durations.push(Number(selectedVideoInfo?.duration) || 0);
    return Math.max(0, ...durations);
}

function renderComparisonDetails() {
    const comparison = {};
    Object.entries(currentComparisonResults).forEach(([modelName, result]) => {
        const detections = result.detections || [];
        comparison[modelName] = {
            total: detections.length,
            avg_confidence: detections.length
                ? detections.reduce((sum, item) => sum + Number(item.confidence || 0), 0) / detections.length
                : 0,
        };
    });
    renderTable(currentComparisonResults, comparison);
    renderSideBySide(currentComparisonResults);
    renderCharts(currentComparisonResults);
}

function toggleWideComparison() {
    const section = document.getElementById('comparison-results');
    const button = document.getElementById('compare-wide-toggle');
    const isWide = section.classList.toggle('is-wide');
    document.body.classList.toggle('compare-results-focus', isWide);
    button.setAttribute('aria-pressed', String(isWide));
    button.textContent = isWide ? 'Vista normal' : 'Vista amplia';
    window.setTimeout(renderComparisonTimelines, 180);
}

function manualComparisonFromSession(session) {
    const candidates = Array.isArray(session?.candidates) ? session.candidates : [];
    if (!candidates.length) return null;
    return {
        session_id: session.session_id,
        dataset_split: session.dataset_split || 'train',
        updated_at: session.updated_at,
        labels: candidates.map((candidate, index) => ({
            block_index: Number(candidate.candidate_index ?? index),
            start_time: Number(candidate.start_time ?? index * 7),
            end_time: Number(candidate.end_time ?? (index + 1) * 7),
            final_label: candidate.final_label || null,
            predicted_label: candidate.predicted_label || null,
        })),
        summary: session.summary || {},
        clip_generator_url: `/clip-generator?session_id=${encodeURIComponent(session.session_id || '')}`,
    };
}

async function loadManualComparisonFromVideoSession() {
    let videoInfo = [...availableVideos, ...availableProcessedVideos]
        .find(item => item.path === selectedVideo);
    if (!videoInfo?.clip_generator?.session_id) {
        const videosResponse = await fetch('/api/videos');
        if (videosResponse.ok) {
            const videosData = await videosResponse.json();
            videoInfo = (videosData.videos || []).find(item => item.path === selectedVideo);
        }
    }

    const sessionId = videoInfo?.clip_generator?.session_id;
    if (!sessionId) return null;
    const sessionResponse = await fetch(`/api/review/session/${encodeURIComponent(sessionId)}`);
    if (!sessionResponse.ok) return null;
    const sessionData = await sessionResponse.json();
    if (!sessionData.success) return null;
    return manualComparisonFromSession(sessionData.session);
}

async function loadManualComparison() {
    const container = document.getElementById('compare-manual-timeline');
    container.hidden = true;
    manualComparison = null;
    if (!selectedVideo) return;

    const embedded = Object.values(currentComparisonResults)
        .map(result => result.manual_comparison)
        .find(reference => reference?.labels?.length
            && (reference.composite_reference || reference.reference_session_pinned));
    if (embedded) {
        manualComparison = embedded;
        return;
    }

    try {
        const directResponse = await fetch(`/api/clip-generator/labels?video_path=${encodeURIComponent(selectedVideo)}`);
        if (directResponse.ok) {
            const directData = await directResponse.json();
            if (directData.success && directData.available && directData.comparison?.labels?.length) {
                manualComparison = directData.comparison;
                return;
            }
        }

        // Older running backends do not expose the direct labels endpoint yet,
        // but the video list already links each source to its clip-generator session.
        manualComparison = await loadManualComparisonFromVideoSession();
        if (manualComparison?.labels?.length) return;

        // Compatibility fallback for servers that still expose labels only
        // through a saved individual analysis.
        const analysisResponse = await fetch(`/api/clip-generator/analysis?video_path=${encodeURIComponent(selectedVideo)}`);
        const analysisData = await analysisResponse.json();
        const resultId = analysisData?.analysis?.result_id;
        if (!analysisData.success || !analysisData.available || !resultId) return;

        const labelsResponse = await fetch(`/api/result/${encodeURIComponent(resultId)}/clip-labels`);
        const labelsData = await labelsResponse.json();
        if (!labelsData.success || !labelsData.available || !labelsData.comparison?.labels?.length) return;
        manualComparison = labelsData.comparison;
    } catch (error) {
        console.warn('No se pudo cargar el timeline manual:', error);
    }
}

function timelineSegment(item, duration, extraClass = '') {
    const start = Math.max(0, Number(item.start_time) || 0);
    const end = Math.max(start, Number(item.end_time) || start);
    const left = duration > 0 ? (start / duration) * 100 : 0;
    const width = duration > 0 ? ((end - start) / duration) * 100 : 0;
    const label = item.play_type || item.final_label || 'pending';
    const confidence = Number(item.confidence);
    const confidenceText = Number.isFinite(confidence) ? ` · ${(confidence * 100).toFixed(0)}%` : '';
    const title = `${label.toUpperCase()} · ${formatTime(start)}–${formatTime(end)}${confidenceText}`;
    return `<button type="button" class="compare-timeline-segment ${escapeHtml(label)} ${extraClass}"
        style="left:${left.toFixed(5)}%;width:max(${width.toFixed(5)}%,3px)"
        data-start="${start}" data-end="${end}" title="${escapeHtml(title)}"
        aria-label="${escapeHtml(title)}"></button>`;
}

function renderComparisonTimelines() {
    const container = document.getElementById('compare-model-timelines');
    if (!container) return;
    comparisonDuration = Math.max(comparisonDuration, getComparisonDuration(currentComparisonResults));
    const duration = comparisonDuration || 1;

    container.innerHTML = orderedModelNames(currentComparisonResults).map((modelName, modelIndex) => {
        const result = currentComparisonResults[modelName];
        const detections = result.detections || [];
        return `
            <div class="compare-timeline-row" data-model="${escapeHtml(modelName)}">
                <div class="compare-timeline-label">
                    <strong title="${escapeHtml(modelLabel(modelName))}">${modelIndex + 1} · ${escapeHtml(modelLabel(modelName))}</strong>
                    <span>${escapeHtml(timelineModelDescriptor(modelName, result))} · ${detections.length} ${detections.length === 1 ? 'jugada' : 'jugadas'}</span>
                </div>
                <div class="compare-timeline-track" aria-label="Timeline de ${escapeHtml(modelLabel(modelName))}">
                    ${detections.map(item => timelineSegment(item, duration)).join('')}
                    <span class="compare-timeline-playhead" style="left:0%"></span>
                </div>
            </div>`;
    }).join('');

    const manualContainer = document.getElementById('compare-manual-timeline');
    if (manualComparison?.labels?.length) {
        const showOther = document.getElementById('compare-show-manual-other')?.checked ?? false;
        const labels = manualComparison.labels.map(item => ({
            ...item,
            final_label: item.final_label || 'pending',
        }));
        document.getElementById('compare-manual-track').innerHTML =
            labels
                .filter(item => showOther || item.final_label !== 'other')
                .map(item => timelineSegment(item, duration, 'manual-block')).join('')
            + '<span class="compare-timeline-playhead" style="left:0%"></span>';
        const summary = manualComparison.summary || {};
        const reviewed = Number(summary.reviewed) || labels.filter(item => item.final_label !== 'pending').length;
        const finalCounts = summary.final_counts || {};
        const lineCount = Number(finalCounts.line) || labels.filter(item => item.final_label === 'line').length;
        const scrumCount = Number(finalCounts.scrum) || labels.filter(item => item.final_label === 'scrum').length;
        document.getElementById('compare-manual-summary').textContent =
            summary.unit === 'events'
                ? `${lineCount} line · ${scrumCount} scrum · referencia manual de S01–S04`
                : `${lineCount} line · ${scrumCount} scrum · ${reviewed}/${labels.length} bloques revisados`;
        const link = document.getElementById('compare-manual-link');
        manualContainer.querySelector('strong').textContent = `${Object.keys(currentComparisonResults).length + 1} · Etiquetado real`;
        link.hidden = Boolean(manualComparison.composite_reference);
        link.href = manualComparison.clip_generator_url || `/clip-generator?session_id=${encodeURIComponent(manualComparison.session_id || '')}`;
        manualContainer.hidden = false;
    } else {
        manualContainer.hidden = true;
    }

    document.querySelectorAll('.compare-timeline-segment').forEach(segment => {
        segment.addEventListener('click', () => {
            const video = document.getElementById('compare-video');
            video.currentTime = Number(segment.dataset.start) || 0;
            video.play().catch(() => {});
        });
    });
    window.RugbyOmissionReview?.render({
        results: currentComparisonResults,
        host: document.querySelector('.compare-timelines-panel'),
        video: document.getElementById('compare-video')
    });
    updateComparisonPlaybackState();
}

function updateComparisonPlaybackState() {
    const video = document.getElementById('compare-video');
    if (!video) return;
    const currentTime = Number(video.currentTime) || 0;
    const duration = Number(video.duration) || comparisonDuration || 0;
    const progress = duration > 0 ? Math.max(0, Math.min(100, (currentTime / duration) * 100)) : 0;
    document.getElementById('compare-current-time').textContent = `${formatTime(currentTime)} / ${formatTime(duration)}`;
    document.querySelectorAll('.compare-timeline-playhead').forEach(item => {
        item.style.left = `${progress}%`;
    });
    document.querySelectorAll('.compare-timeline-segment').forEach(item => {
        const active = currentTime >= Number(item.dataset.start) && currentTime <= Number(item.dataset.end);
        item.classList.toggle('is-active', active);
    });

    const activeModels = orderedModelNames(currentComparisonResults).flatMap(modelName =>
        (currentComparisonResults[modelName].detections || [])
            .filter(item => currentTime >= Number(item.start_time) && currentTime <= Number(item.end_time))
            .map(item => `${modelLabel(modelName)}: ${String(item.play_type).toUpperCase()}`)
    );
    document.getElementById('compare-active-events').textContent = activeModels.length
        ? activeModels.join(' · ')
        : 'Sin jugadas activas';
}

// ─── Post-classification thresholds (no new inference) ──────────────────────

function syncResultThresholdControls(confidence, line) {
    const confidenceInput = document.getElementById('compare-results-confidence');
    const lineInput = document.getElementById('compare-results-line-confidence');
    confidenceInput.value = Number(confidence).toFixed(2);
    lineInput.value = Number(line).toFixed(2);
    document.getElementById('compare-results-confidence-value').textContent = Number(confidence).toFixed(2);
    document.getElementById('compare-results-line-confidence-value').textContent = Number(line).toFixed(2);
}

function scheduleThresholdUpdate() {
    const confidence = Number(document.getElementById('compare-results-confidence').value);
    const line = Number(document.getElementById('compare-results-line-confidence').value);
    document.getElementById('compare-results-confidence-value').textContent = confidence.toFixed(2);
    document.getElementById('compare-results-line-confidence-value').textContent = line.toFixed(2);
    const status = document.getElementById('compare-threshold-status');
    status.textContent = 'Actualizando ambos timelines…';
    window.clearTimeout(thresholdUpdateTimer);
    thresholdUpdateTimer = window.setTimeout(() => applyComparisonThresholds(confidence, line), 90);
}

function resetComparisonThresholds() {
    syncResultThresholdControls(initialComparisonThresholds.confidence, initialComparisonThresholds.line);
    applyComparisonThresholds(initialComparisonThresholds.confidence, initialComparisonThresholds.line);
}

function applyComparisonThresholds(confidence, line) {
    const currentTime = Number(document.getElementById('compare-video')?.currentTime) || 0;
    currentComparisonResults = Object.fromEntries(
        Object.entries(originalComparisonResults).map(([modelName, result]) => [
            modelName,
            { ...result, detections: rebuildEventDetections(result, confidence, line) },
        ])
    );
    renderComparisonDetails();
    renderComparisonTimelines();
    const video = document.getElementById('compare-video');
    if (video && Number.isFinite(currentTime)) video.currentTime = currentTime;
    updateComparisonPlaybackState();
    const total = Object.values(currentComparisonResults)
        .reduce((sum, result) => sum + (result.detections || []).length, 0);
    document.getElementById('compare-threshold-status').textContent =
        `Actualizado sin nueva inferencia: ${total} ${total === 1 ? 'jugada' : 'jugadas'} entre los modelos.`;
}

function rebuildEventDetections(result, confidenceThreshold, lineThreshold) {
    if (result.metadata?.saved_decisions_locked) {
        return cloneResults(result.detections || []);
    }
    const windows = result.window_predictions || [];
    if (!windows.length) {
        return (result.detections || []).filter(item => Number(item.confidence || 0) >= (
            item.play_type === 'line' ? lineThreshold : confidenceThreshold
        ));
    }

    const policy = result.event_post_processing || result.metadata?.event_post_processing || {};
    const targetLabels = (policy.target_labels || ['scrum', 'line'])
        .filter(label => windows.some(item => Object.hasOwn(item.probabilities || {}, label)));
    if (!targetLabels.length) return [];

    const contextRadius = Number(policy.context_radius_seconds) || 0;
    const mergeGap = Number(policy.merge_gap_seconds ?? 1.5);
    const anchorConfidence = Number(policy.anchor_confidence ?? 0.85);
    const neighborBoost = Number(policy.neighbor_boost) || 0;
    const requireWinner = policy.require_target_winner !== false;
    const mergeByStart = policy.merge_by_start_time !== false;
    const policyMinimum = Number(policy.min_target_confidence) || 0;
    const minSupport = Math.max(1, Number(policy.min_support_windows) || 1);
    const maxDuration = policy.max_duration_seconds == null ? null : Number(policy.max_duration_seconds);
    const thresholds = Object.fromEntries(targetLabels.map(label => [
        label,
        Math.max(confidenceThreshold, policyMinimum, label === 'line' ? lineThreshold : 0),
    ]));
    const anchors = Object.fromEntries(targetLabels.map(label => [
        label,
        windows.filter(item => Number(item.probabilities?.[label] || 0) >= anchorConfidence),
    ]));

    const evidence = [];
    windows.forEach(item => {
        const rawLabel = item.play_type;
        if (requireWinner && !targetLabels.includes(rawLabel)) return;
        const probabilities = { ...(item.probabilities || {}) };
        const center = (Number(item.start_time) + Number(item.end_time)) / 2;

        targetLabels.forEach(label => {
            const base = Number(probabilities[label] || 0);
            let bestInfluence = 0;
            anchors[label].forEach(anchor => {
                if (anchor === item) return;
                const gap = Math.max(0,
                    Math.max(Number(item.start_time), Number(anchor.start_time))
                    - Math.min(Number(item.end_time), Number(anchor.end_time)));
                const anchorCenter = (Number(anchor.start_time) + Number(anchor.end_time)) / 2;
                const centerDistance = Math.abs(center - anchorCenter);
                const radius = contextRadius + Number(item.duration || (item.end_time - item.start_time));
                if (gap <= mergeGap && centerDistance <= radius) {
                    const temporalWeight = Math.max(0, 1 - (centerDistance / Math.max(radius, 1e-6)));
                    bestInfluence = Math.max(
                        bestInfluence,
                        Number(anchor.probabilities?.[label] || 0) * temporalWeight,
                    );
                }
            });
            if (bestInfluence > 0) {
                probabilities[label] = Math.min(1, base + neighborBoost * bestInfluence * (1 - base));
            }
        });

        const bestTarget = targetLabels.reduce((best, label) =>
            Number(probabilities[label] || 0) > Number(probabilities[best] || 0) ? label : best
        );
        const bestProbability = Number(probabilities[bestTarget] || 0);
        if (bestProbability >= thresholds[bestTarget] && (!requireWinner || bestTarget === rawLabel)) {
            evidence.push({
                ...item,
                play_type: bestTarget,
                confidence: bestProbability,
                original_confidence: Number(item.probabilities?.[bestTarget] || 0),
                probabilities,
            });
        }
    });
    if (!evidence.length) return [];

    evidence.sort((a, b) => String(a.play_type).localeCompare(String(b.play_type))
        || Number(a.start_time) - Number(b.start_time)
        || Number(a.end_time) - Number(b.end_time));
    const merged = [];
    let current = null;

    const hasSupport = event => event.windows.length >= minSupport
        || Math.max(...event.windows.map(item => Number(item.confidence || 0))) >= anchorConfidence;
    const flush = () => {
        if (current && hasSupport(current)) merged.push(finalizeEvent(current, result));
    };

    evidence.forEach(item => {
        const mergeReference = current
            ? (mergeByStart ? Number(current.last_start_time) : Number(current.end_time))
            : null;
        if (!current || item.play_type !== current.play_type || Number(item.start_time) - mergeReference > mergeGap) {
            flush();
            current = {
                play_type: item.play_type,
                start_time: Number(item.start_time),
                end_time: Number(item.end_time),
                last_start_time: Number(item.start_time),
                windows: [item],
            };
            return;
        }
        if (maxDuration != null && Number(item.end_time) - current.start_time > maxDuration) {
            flush();
            current = {
                play_type: item.play_type,
                start_time: Math.max(Number(item.start_time), Number(current.end_time)),
                end_time: Number(item.end_time),
                last_start_time: Number(item.start_time),
                windows: [item],
            };
            return;
        }
        current.start_time = Math.min(current.start_time, Number(item.start_time));
        current.end_time = Math.max(current.end_time, Number(item.end_time));
        current.last_start_time = Number(item.start_time);
        current.windows.push(item);
    });
    flush();
    return merged.sort((a, b) => a.start_time - b.start_time);
}

function finalizeEvent(event, result) {
    const duration = Number(result.metadata?.duration) || comparisonDuration || Number(event.end_time);
    const start = Math.max(0, Number(event.start_time));
    const end = Math.min(duration, Number(event.end_time));
    const peakWindow = event.windows.reduce((best, item) =>
        Number(item.confidence || 0) > Number(best.confidence || 0) ? item : best
    );
    const probabilitySums = {};
    event.windows.forEach(item => Object.entries(item.probabilities || {}).forEach(([label, value]) => {
        probabilitySums[label] = (probabilitySums[label] || 0) + Number(value || 0);
    }));
    const count = event.windows.length || 1;
    const probabilities = Object.fromEntries(
        Object.entries(probabilitySums).map(([label, value]) => [label, value / count])
    );
    return {
        play_type: event.play_type,
        start_time: start,
        end_time: end,
        duration: Math.max(0, end - start),
        confidence: Math.max(...event.windows.map(item => Number(item.confidence || 0))),
        mean_confidence: event.windows.reduce((sum, item) => sum + Number(item.confidence || 0), 0) / count,
        frame_start: peakWindow.frame_start,
        frame_end: peakWindow.frame_end,
        probabilities,
        source_window_count: count,
        peak_window: {
            start_time: Number(peakWindow.start_time),
            end_time: Number(peakWindow.end_time),
            confidence: Number(peakWindow.confidence || 0),
            original_confidence: Number(peakWindow.original_confidence ?? peakWindow.confidence ?? 0),
        },
    };
}
