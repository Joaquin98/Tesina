// Main JavaScript for Rugby Play Detection Interface

let selectedModel = null;
let selectedVideo = null;
let currentResultId = null;
let currentAnalysisJobId = null;
let analysisStartedAt = null;
let currentDetectionFilter = 'all';
let currentLiveJobId = null;
let liveStatusPollTimer = null;
let livePlaybackSyncTimer = null;
let livePredictionHistory = [];
let trainingDatasetSummary = null;
let availableModelsByName = {};
let availableModels = [];
let availableVideos = [];
let availableProcessedVideos = [];
let historyResults = [];
let historyPage = 1;
const HISTORY_PAGE_SIZE = 15;
const LIGHT_LIVE_MODEL_NAME = 'light_live/resnet50__lstm__frames20/model.keras';
const BEST_ANALYSIS_MODEL_NAME = 'best_model/best_model.keras';
const GEOMETRY_FUSION_MODEL_NAME = 'player_geometry_fusion_calibrated';
const M1_H2_D2_MODEL_NAME = 'm1_h2_d2_cross_fold_ensemble';
const M3_H2_D2_MODEL_NAME = 'm3_h2_d2_cross_fold_ensemble';
const M3_H1_D2_MODEL_NAME = 'm3_h1_d2_cross_fold_ensemble';
const HISTORICAL_YOLO_RESIDUAL_MODEL_NAME = 'historical_yolo_residual_ensemble';
const RGB_YOLO_QUANTILE_FUSION_MODEL_NAME = 'rgb_yolo_quantile_fusion_all';
const RGB_YOLO_EVENT_VERIFIER_MODEL_NAME = 'rgb_yolo_highres_temporal_event_verifier';
const RGB_YOLO_PRIMERA_CALIBRATED_MODEL_NAME = 'rgb_yolo_historical_primera_calibrated';
const META_CORRECTION_MODEL_NAME = 'meta_correction/recommended_same_video_50pct';
const RECOMMENDED_ANALYSIS_MODELS = new Set([
    M1_H2_D2_MODEL_NAME,
    M3_H2_D2_MODEL_NAME,
    M3_H1_D2_MODEL_NAME,
    BEST_ANALYSIS_MODEL_NAME,
    'rgb_yolo_quantile_highres_roi',
    RGB_YOLO_PRIMERA_CALIBRATED_MODEL_NAME,
    GEOMETRY_FUSION_MODEL_NAME,
    LIGHT_LIVE_MODEL_NAME,
]);
const MAX_VISIBLE_VIDEOS = 36;
const CONFIDENCE_STORAGE_KEY = 'rugby-play-detection.confidence-threshold';
const LINE_CONFIDENCE_STORAGE_KEY = 'rugby-play-detection.line-confidence-threshold';
const replayClassVisibility = {
    scrum: true,
    line: true,
    other: true,
};
const replayState = {
    duration: 0,
    animationFrameId: null,
    controlsBound: false,
    initialSeekApplied: false,
    fallbackAttempted: false,
    sourcePath: null,
    clipComparisonResultId: null,
    clipComparison: null,
    clipComparisonLoading: false,
    clipLabelsVisible: false,
};
let replayModelRoiController = null;
let preprocessingPreviewRoiController = null;
let thresholdPreviewTimer = null;
let thresholdPreviewSequence = 0;
let initialResultThresholds = { confidence: 0.70, line: 0.95 };

function roiInputModeForModel(modelOrName) {
    const model = typeof modelOrName === 'object' && modelOrName !== null
        ? modelOrName
        : (availableModelsByName[String(modelOrName || '')] || { name: modelOrName });
    const descriptor = `${model.name || ''} ${model.display_name || ''} ${model.model_type || ''}`.toLowerCase();
    return model.pose_input_mode === 'source_equivalent_roi'
        || descriptor.includes('highres')
        || descriptor.includes('alta resolución')
        ? 'source_equivalent_roi'
        : 'rgb_224';
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    setupEventListeners();
    setupPreprocessingPreviewOverlay();
    setupParameterSliders();
    setupPostClassificationThresholdControls();
    restoreSavedConfidenceThreshold();
    restoreSavedLineConfidenceThreshold();
    if (document.getElementById('models-grid')) loadModels();
    if (document.getElementById('videos-grid')) loadVideos();
    if (document.getElementById('processed-videos-grid')) loadProcessedVideos();
    if (document.getElementById('live-detection-section')) {
        renderLiveContext();
        renderLiveHistory();
    }
    if (document.getElementById('training-dataset-section')) loadTrainingDatasetSummary();
    openModalFromQueryParam();
}

function setupPreprocessingPreviewOverlay() {
    preprocessingPreviewRoiController = window.ModelRoiOverlay?.attach({
        video: document.getElementById('preview-video'),
        stage: document.getElementById('preview-video-container'),
        overlay: document.getElementById('preview-model-roi'),
        toggle: document.getElementById('show-preview-model-roi'),
        inputMode: roiInputModeForModel(selectedModel),
    }) || null;
}

function openModalFromQueryParam() {
    const params = new URLSearchParams(window.location.search);
    const open = (params.get('open') || '').toLowerCase();

    if (open === 'live') {
        document.getElementById('live-detection-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else if (open === 'training-dataset') {
        document.getElementById('training-dataset-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else if (open === 'history') {
        openHistoryModal();
    } else if (open === 'about') {
        document.getElementById('about-modal')?.classList.add('active');
    } else {
        return;
    }

    // Clean query parameter once modal is opened.
    if (window.history?.replaceState) {
        window.history.replaceState({}, document.title, window.location.pathname);
    }
}

// Event Listeners
function setupEventListeners() {
    // Refresh buttons
    document.getElementById('refresh-models')?.addEventListener('click', loadModels);
    document.getElementById('model-search')?.addEventListener('input', () => renderModels());
    document.getElementById('model-filter')?.addEventListener('change', () => renderModels());
    document.getElementById('video-search')?.addEventListener('input', renderVideoCollections);
    document.getElementById('video-source-filter')?.addEventListener('change', renderVideoCollections);
    
    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });
    
    // File upload
    const fileInput = document.getElementById('file-input');
    const uploadZone = document.getElementById('upload-zone');
    
    fileInput?.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadZone?.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    
    uploadZone?.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });
    
    uploadZone?.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            uploadFile(files[0]);
        }
    });
    
    // Run analysis
    document.getElementById('run-analysis')?.addEventListener('click', runAnalysis);
    
    // Reset parameters
    document.getElementById('reset-params')?.addEventListener('click', resetParameters);
    
    // About modal
    document.getElementById('about-link')?.addEventListener('click', (e) => {
        const modal = document.getElementById('about-modal');
        if (!modal) return;
        e.preventDefault();
        modal.classList.add('active');
    });

    // History modal
    document.getElementById('history-link')?.addEventListener('click', (e) => {
        if (!document.getElementById('history-modal')) return;
        e.preventDefault();
        openHistoryModal();
    });

    document.getElementById('training-dataset-link')?.addEventListener('click', (e) => {
        const section = document.getElementById('training-dataset-section');
        if (!section) return;
        e.preventDefault();
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });

    // Live section shortcut
    document.getElementById('live-link')?.addEventListener('click', (e) => {
        const section = document.getElementById('live-detection-section');
        if (!section) return;
        e.preventDefault();
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });

    // Modal close handlers - handle all modals
    document.querySelectorAll('.modal-close').forEach(closeBtn => {
        closeBtn.addEventListener('click', (e) => {
            const modal = e.target.closest('.modal');
            if (modal) {
                modal.classList.remove('active');
                modal.style.display = 'none';
                
                // Stop video if it's the clip modal
                if (modal.id === 'clip-modal') {
                    const video = modal.querySelector('#clip-video');
                    if (video) {
                        // Clean up event listeners
                        if (video._clipCleanup) {
                            video._clipCleanup();
                            video._clipCleanup = null;
                        }
                        video.pause();
                        video.src = '';
                    }
                }
            }
        });
    });
    
    // Close modal when clicking outside
    window.addEventListener('click', (e) => {
        if (e.target.classList.contains('modal')) {
            e.target.classList.remove('active');
            e.target.style.display = 'none';
            
            // Stop video if it's the clip modal
            if (e.target.id === 'clip-modal') {
                const video = e.target.querySelector('#clip-video');
                if (video) {
                    // Clean up event listeners
                    if (video._clipCleanup) {
                        video._clipCleanup();
                        video._clipCleanup = null;
                    }
                    video.pause();
                    video.src = '';
                }
            }
        }
    });
    
    // Export buttons
    document.getElementById('start-review')?.addEventListener('click', startActiveReview);
    document.getElementById('start-review-main')?.addEventListener('click', startActiveReview);
    document.getElementById('start-review-non-overlap')?.addEventListener(
        'click',
        () => startActiveReview('non_overlapping'),
    );
    document.getElementById('start-review-full-match')?.addEventListener(
        'click',
        () => startActiveReview('full_match'),
    );
    document.getElementById('export-json')?.addEventListener('click', () => exportResults('json'));
    document.getElementById('export-csv')?.addEventListener('click', () => exportResults('csv'));
    document.getElementById('view-timeline')?.addEventListener('click', viewTimeline);
    document.getElementById('results-replay-focus-toggle')?.addEventListener('click', toggleReplayFocusMode);
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && document.getElementById('results-replay-shell')?.classList.contains('is-focus-mode')) {
            setReplayFocusMode(false);
        }
    });

    // Live detection controls
    document.getElementById('use-light-live-model')?.addEventListener('click', useLightLiveModel);
    document.getElementById('start-live-detection')?.addEventListener('click', startLiveDetection);
    document.getElementById('stop-live-detection')?.addEventListener('click', stopLiveDetection);
    document.getElementById('refresh-live-status')?.addEventListener('click', refreshLiveStatus);
    document.getElementById('live-video-preview')?.addEventListener('timeupdate', renderLivePredictionForPlayback);
    document.getElementById('live-video-preview')?.addEventListener('seeked', renderLivePredictionForPlayback);
    document.getElementById('live-video-preview')?.addEventListener('play', syncLivePlaybackTime);
    document.getElementById('live-video-preview')?.addEventListener('pause', syncLivePlaybackTime);
    
    // Preview preprocessing button
    document.getElementById('preview-preprocessing')?.addEventListener('click', previewPreprocessing);

    // Training dataset panel
    document.getElementById('refresh-training-dataset')?.addEventListener('click', loadTrainingDatasetSummary);

    const temporalModeSelect = document.getElementById('temporal-fps-mode');
    const customFpsInput = document.getElementById('custom-effective-fps');
    temporalModeSelect?.addEventListener('change', () => {
        if (!customFpsInput) return;
        customFpsInput.style.display = temporalModeSelect.value === 'custom' ? 'block' : 'none';
    });
}

function setupParameterSliders() {
    const sliders = [
        { id: 'confidence-threshold', valueId: 'confidence-value', format: v => v.toFixed(2) },
        { id: 'line-confidence-threshold', valueId: 'line-confidence-value', format: v => v.toFixed(2) },
        { id: 'batch-size', valueId: 'batch-value', format: v => Math.round(v) },
    ];
    
    sliders.forEach(({ id, valueId, format }) => {
        const slider = document.getElementById(id);
        const valueDisplay = document.getElementById(valueId);
        
        slider?.addEventListener('input', (e) => {
            const numericValue = parseFloat(e.target.value);
            valueDisplay.textContent = format(numericValue);
            if (id === 'confidence-threshold') {
                saveConfidenceThreshold(numericValue);
            } else if (id === 'line-confidence-threshold') {
                saveLineConfidenceThreshold(numericValue);
            }
            validateInputs();
        });
    });
}

function setupPostClassificationThresholdControls() {
    const confidence = document.getElementById('results-confidence-threshold');
    const line = document.getElementById('results-line-confidence-threshold');
    const updateValue = (input, outputId) => {
        document.getElementById(outputId).textContent = Number(input.value).toFixed(2);
    };
    const schedulePreview = () => {
        window.clearTimeout(thresholdPreviewTimer);
        thresholdPreviewTimer = window.setTimeout(applyPostClassificationThresholds, 220);
    };

    confidence?.addEventListener('input', () => {
        updateValue(confidence, 'results-confidence-value');
        schedulePreview();
    });
    line?.addEventListener('input', () => {
        updateValue(line, 'results-line-confidence-value');
        schedulePreview();
    });
    document.getElementById('results-threshold-reset')?.addEventListener('click', () => {
        if (!confidence || !line) return;
        confidence.value = initialResultThresholds.confidence.toFixed(2);
        line.value = initialResultThresholds.line.toFixed(2);
        updateValue(confidence, 'results-confidence-value');
        updateValue(line, 'results-line-confidence-value');
        window.clearTimeout(thresholdPreviewTimer);
        applyPostClassificationThresholds();
    });
}

function initializePostClassificationThresholds(results) {
    const panel = document.getElementById('results-threshold-controls');
    const confidence = document.getElementById('results-confidence-threshold');
    const line = document.getElementById('results-line-confidence-threshold');
    if (!panel || !confidence || !line) return;

    const metadata = results?.metadata || {};
    if (metadata.saved_decisions_locked) {
        panel.hidden = true;
        return;
    }
    const eventPolicy = results?.event_post_processing || metadata.event_post_processing || {};
    const fixedLineVerifier = eventPolicy.line_mode === 'event_logistic_verifier';
    const fallbackConfidence = Number(document.getElementById('confidence-threshold')?.value || 0.70);
    const savedConfidence = Number(metadata.confidence_threshold);
    const fallbackLine = Number.isFinite(savedConfidence)
        ? savedConfidence
        : Number(document.getElementById('line-confidence-threshold')?.value || 0.95);
    const savedLine = metadata.line_confidence_threshold === null || metadata.line_confidence_threshold === undefined
        ? fallbackLine
        : Number(metadata.line_confidence_threshold);
    initialResultThresholds = {
        confidence: Number.isFinite(savedConfidence) ? savedConfidence : fallbackConfidence,
        line: Number.isFinite(savedLine) ? savedLine : fallbackLine,
    };
    confidence.value = initialResultThresholds.confidence.toFixed(2);
    line.value = initialResultThresholds.line.toFixed(2);
    line.disabled = fixedLineVerifier;
    document.getElementById('results-confidence-value').textContent = initialResultThresholds.confidence.toFixed(2);
    document.getElementById('results-line-confidence-value').textContent = initialResultThresholds.line.toFixed(2);
    document.getElementById('results-threshold-status').textContent = fixedLineVerifier
        ? `Valores usados al iniciar: confianza ${initialResultThresholds.confidence.toFixed(2)} · verificador de line fijo ${Number(eventPolicy.line_verifier_threshold || 0.11).toFixed(2)}.`
        : `Valores usados al iniciar: confianza ${initialResultThresholds.confidence.toFixed(2)} · line ${initialResultThresholds.line.toFixed(2)}.`;
    document.getElementById('results-threshold-status').className = 'results-threshold-status';
    panel.hidden = false;
}

async function applyPostClassificationThresholds() {
    if (!currentResultId || !_baseAnalysisResults) return;
    const confidence = Number(document.getElementById('results-confidence-threshold')?.value);
    const line = Number(document.getElementById('results-line-confidence-threshold')?.value);
    if (!Number.isFinite(confidence) || !Number.isFinite(line)) return;

    const status = document.getElementById('results-threshold-status');
    const sequence = ++thresholdPreviewSequence;
    status.textContent = 'Recalculando las detecciones guardadas...';
    status.className = 'results-threshold-status';
    try {
        const response = await fetch(`/api/result/${encodeURIComponent(currentResultId)}/threshold-preview`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                confidence_threshold: confidence,
                line_confidence_threshold: line,
            }),
        });
        const data = await response.json();
        if (!response.ok || !data.success) throw new Error(data.error || 'No se pudieron recalcular los umbrales');
        if (sequence !== thresholdPreviewSequence) return;

        const previewResults = {
            ..._baseAnalysisResults,
            detections: data.detections || [],
            statistics: data.statistics || {},
            summary: data.summary || {},
            active_thresholds: data.thresholds,
        };
        displayResults(previewResults, { thresholdPreview: true });
        const effective = data.thresholds || {};
        const sourceNote = data.used_cached_windows
            ? 'sin volver a clasificar el video'
            : 'limitado a las detecciones disponibles en este resultado antiguo';
        const lineStatus = effective.line_threshold_controlled_by_model
            ? `verificador de line fijo ${Number(effective.effective_line_threshold).toFixed(2)}`
            : `umbral efectivo line ${Number(effective.effective_line_threshold || line).toFixed(2)}`;
        status.textContent = `${previewResults.detections.length} jugadas visibles · umbral efectivo scrum ${Number(effective.effective_scrum_threshold || confidence).toFixed(2)} · ${lineStatus} · ${sourceNote}.`;
        status.className = 'results-threshold-status';
    } catch (error) {
        if (sequence !== thresholdPreviewSequence) return;
        status.textContent = error.message;
        status.className = 'results-threshold-status error';
    }
}

function saveConfidenceThreshold(value) {
    if (!Number.isFinite(value)) return;
    const clampedValue = Math.max(0, Math.min(1, value));
    try {
        localStorage.setItem(CONFIDENCE_STORAGE_KEY, clampedValue.toFixed(2));
    } catch (_) {
        // Storage can be unavailable in restricted browser modes.
    }
}

function saveLineConfidenceThreshold(value) {
    if (!Number.isFinite(value)) return;
    const clampedValue = Math.max(0, Math.min(1, value));
    try {
        localStorage.setItem(LINE_CONFIDENCE_STORAGE_KEY, clampedValue.toFixed(2));
    } catch (_) {
        // Storage can be unavailable in restricted browser modes.
    }
}

function restoreSavedConfidenceThreshold() {
    const slider = document.getElementById('confidence-threshold');
    const valueDisplay = document.getElementById('confidence-value');
    if (!slider || !valueDisplay) return;

    let savedValue;
    try {
        savedValue = parseFloat(localStorage.getItem(CONFIDENCE_STORAGE_KEY));
    } catch (_) {
        return;
    }
    if (!Number.isFinite(savedValue)) return;

    const clampedValue = Math.max(0, Math.min(1, savedValue));
    slider.value = clampedValue.toFixed(2);
    valueDisplay.textContent = clampedValue.toFixed(2);

}

function restoreSavedLineConfidenceThreshold() {
    const slider = document.getElementById('line-confidence-threshold');
    const valueDisplay = document.getElementById('line-confidence-value');
    if (!slider || !valueDisplay) return;

    let savedValue;
    try {
        savedValue = parseFloat(localStorage.getItem(LINE_CONFIDENCE_STORAGE_KEY));
    } catch (_) {
        return;
    }
    if (!Number.isFinite(savedValue)) return;

    const clampedValue = Math.max(0, Math.min(1, savedValue));
    slider.value = clampedValue.toFixed(2);
    valueDisplay.textContent = clampedValue.toFixed(2);
}

function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `${tabName}-tab`);
    });
}

// Load Models
async function loadModels() {
    const grid = document.getElementById('models-grid');
    if (!grid) return;
    grid.innerHTML = '<div class="loading">Cargando modelos...</div>';
    
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        if (data.success) {
            availableModels = data.models || [];
            renderModels();
            if (!selectedModel) {
                const preferredName = document.getElementById('live-detection-section')
                    ? LIGHT_LIVE_MODEL_NAME
                    : (availableModelsByName[RGB_YOLO_PRIMERA_CALIBRATED_MODEL_NAME]
                        ? RGB_YOLO_PRIMERA_CALIBRATED_MODEL_NAME
                        : (availableModelsByName[RGB_YOLO_QUANTILE_FUSION_MODEL_NAME]
                            ? RGB_YOLO_QUANTILE_FUSION_MODEL_NAME
                        : (availableModelsByName[GEOMETRY_FUSION_MODEL_NAME]
                            ? GEOMETRY_FUSION_MODEL_NAME
                            : BEST_ANALYSIS_MODEL_NAME)));
                if (availableModelsByName[preferredName]) {
                    selectModel(preferredName, { focusNext: false });
                }
            }
        } else {
            grid.innerHTML = `<div class="error">No se pudieron cargar los modelos: ${data.error}</div>`;
        }
    } catch (error) {
        console.error('Error loading models:', error);
        grid.innerHTML = '<div class="error">No se pudieron cargar los modelos.</div>';
    }
}

function isRecommendedModel(model) {
    return model.recommended === true || RECOMMENDED_ANALYSIS_MODELS.has(model.name);
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
    if (model.name === RGB_YOLO_PRIMERA_CALIBRATED_MODEL_NAME) return 'Mejor modelo · Histórico Primera + RGB/YOLO temporal calibrado';
    if (model.name === RGB_YOLO_EVENT_VERIFIER_MODEL_NAME) return 'Mejor candidato · RGB + YOLO temporal + verificador de line';
    if (model.name === HISTORICAL_YOLO_RESIDUAL_MODEL_NAME) return 'Historico + correccion YOLO residual (87.8%, experimental)';
    if (model.name === RGB_YOLO_QUANTILE_FUSION_MODEL_NAME) return 'Nuevo modelo recomendado (RGB + YOLO, fusion cuantica)';
    if (model.name === GEOMETRY_FUSION_MODEL_NAME) return 'Modelo recomendado (RGB + pose + geometría, 83.7%)';
    if (model.name === M1_H2_D2_MODEL_NAME) return 'M1-H2-D2 · ensemble de 6 folds';
    if (model.name === M3_H2_D2_MODEL_NAME) return 'M3-H2-D2 · ensemble de 6 folds';
    if (model.name === M3_H1_D2_MODEL_NAME) return 'M3-H1-D2 · ensemble de 6 folds';
    if (model.name === BEST_ANALYSIS_MODEL_NAME) return 'Modelo principal (ResNet-50 + BiGRU, 81.3%)';
    if (model.name === LIGHT_LIVE_MODEL_NAME) return 'Modelo ligero para CPU (ResNet-50 + LSTM, 20 cuadros)';
    if (model.name === META_CORRECTION_MODEL_NAME) return 'CNN-RNN + corrector few-shot (mismo video)';
    if (model.name?.includes('human_review_retrain_860')) return 'Reentrenamiento con 860 revisiones (experimental)';
    if (model.name?.includes('best_model_focused')) return 'Mejor modelo de la búsqueda focalizada (etapa 2)';
    if (model.display_name?.startsWith('InceptionV3-GRU')) {
        return model.display_name.replace('InceptionV3-GRU', 'ResNet-50 + modelo temporal');
    }
    return directTranslations[model.display_name] || model.display_name || model.name;
}

function modelMatchesFilter(model, filter) {
    const name = `${model.name} ${model.display_name || ''} ${model.description || ''} ${model.model_type || ''}`.toLowerCase();
    if (filter === 'recommended') return isRecommendedModel(model);
    if (filter === 'live') return name.includes('light') || name.includes('frames20');
    if (filter === 'ensemble') return name.includes('ensemble');
    if (filter === 'meta_correction') return model.model_type === 'meta_correction';
    return true;
}

function renderModels(models = availableModels) {
    const grid = document.getElementById('models-grid');
    if (!grid) return;
    availableModelsByName = {};
    models.forEach(model => {
        availableModelsByName[model.name] = model;
    });

    const query = (document.getElementById('model-search')?.value || '').trim().toLowerCase();
    const filter = document.getElementById('model-filter')?.value || 'all';
    const filteredModels = models.filter(model => {
        const haystack = `${model.name} ${model.display_name || ''} ${model.description || ''} ${model.model_type || ''}`.toLowerCase();
        return (!query || haystack.includes(query)) && modelMatchesFilter(model, filter);
    });

    const countEl = document.getElementById('model-results-count');
    if (countEl) {
        countEl.textContent = filter === 'recommended' && models.length > filteredModels.length
            ? `${filteredModels.length} recomendado${filteredModels.length === 1 ? '' : 's'} · ${models.length} disponibles`
            : `${filteredModels.length} de ${models.length}`;
    }
    
    if (filteredModels.length === 0) {
        grid.innerHTML = '<div class="loading">No hay modelos que coincidan con la búsqueda.</div>';
        return;
    }
    
    grid.innerHTML = filteredModels.map(model => `
        <button type="button" class="model-card ${isRecommendedModel(model) ? 'recommended' : ''}" data-model="${model.name}"
            aria-pressed="${selectedModel === model.name}" onclick="selectModel('${model.name}')">
            <div class="model-card-badges">
                ${isRecommendedModel(model) ? '<span class="recommendation-badge">Recomendado</span>' : ''}
                ${model.experimental ? '<span class="model-type">Experimental</span>' : ''}
                <span class="model-type">${model.model_type}</span>
            </div>
            <h3>${modelDisplayName(model)}</h3>
            ${model.description ? `<p class="model-description">${model.description}</p>` : ''}
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
                ${model.metrics && model.metrics.accuracy ? `
                    <div class="metric">
                        <div class="metric-label">Exactitud</div>
                        <div class="metric-value">${(model.metrics.accuracy * 100).toFixed(1)}%</div>
                    </div>
                ` : ''}
                ${model.metrics && model.metrics.f1_macro ? `
                    <div class="metric">
                        <div class="metric-label">Macro F1</div>
                        <div class="metric-value">${(model.metrics.f1_macro * 100).toFixed(1)}%</div>
                    </div>
                ` : ''}
                ${model.metrics && model.metrics.f1w_temporal ? `
                    <div class="metric">
                        <div class="metric-label">F1 temporal</div>
                        <div class="metric-value">${(model.metrics.f1w_temporal * 100).toFixed(1)}%</div>
                    </div>
                ` : ''}
                ${model.metrics && model.metrics.strict_review_precision ? `
                    <div class="metric">
                        <div class="metric-label">Precisión de revisión</div>
                        <div class="metric-value">${(model.metrics.strict_review_precision * 100).toFixed(1)}%</div>
                    </div>
                ` : ''}
            </div>
        </button>
    `).join('');

    document.querySelectorAll('.model-card').forEach(card => {
        card.classList.toggle('selected', card.dataset.model === selectedModel);
    });

    updateLightLiveShortcut();
}

function selectModel(modelName, { focusNext = true } = {}) {
    selectedModel = modelName;
    preprocessingPreviewRoiController?.setInputMode(roiInputModeForModel(modelName));
    
    document.querySelectorAll('.model-card').forEach(card => {
        card.classList.toggle('selected', card.dataset.model === modelName);
        card.setAttribute('aria-pressed', card.dataset.model === modelName ? 'true' : 'false');
    });

    if (focusNext) focusVideoSelectionSection();
    applyAnalysisDefaultsForModel(modelName);
    applyLiveDefaultsForModel(modelName);
    updateLineThresholdControl(modelName);
    
    validateInputs();
    renderLiveContext();
}

function applyAnalysisDefaultsForModel(modelName) {
    const defaults = availableModelsByName[modelName]?.analysis_defaults;
    if (!defaults) return;

    const values = [
        ['confidence-threshold', 'confidence-value', defaults.confidence_threshold],
        ['line-confidence-threshold', 'line-confidence-value', defaults.line_confidence_threshold],
    ];
    values.forEach(([inputId, outputId, rawValue]) => {
        const value = Number(rawValue);
        const input = document.getElementById(inputId);
        const output = document.getElementById(outputId);
        if (!Number.isFinite(value) || !input || !output) return;
        const clamped = Math.max(0, Math.min(1, value));
        input.value = clamped.toFixed(2);
        output.textContent = clamped.toFixed(2);
    });
}

function updateLineThresholdControl(modelName) {
    const slider = document.getElementById('line-confidence-threshold');
    const description = document.getElementById('line-threshold-description');
    if (!slider || !description) return;
    const fixedVerifier = modelName === RGB_YOLO_EVENT_VERIFIER_MODEL_NAME
        || modelName === RGB_YOLO_PRIMERA_CALIBRATED_MODEL_NAME;
    const model = availableModelsByName[modelName];
    const fixedThreshold = Number(model?.line_verifier_threshold ?? 0.11);
    slider.disabled = fixedVerifier;
    description.textContent = fixedVerifier
        ? `Este modelo usa un verificador de eventos con umbral fijo ${fixedThreshold.toFixed(3)}. El control general continúa aplicándose a scrum.`
        : 'Se aplica solo a line antes de fusionar ventanas. Scrum conserva la confianza mínima general.';
}

function applyLiveDefaultsForModel(modelName) {
    const model = availableModelsByName[modelName];
    const defaults = model?.live_defaults;
    if (!defaults) {
        return;
    }

    const windowInput = document.getElementById('live-window-seconds');
    const targetFpsInput = document.getElementById('live-target-fps');
    const sampledInput = document.getElementById('live-sampled-frame-count');

    if (windowInput && defaults.window_seconds !== undefined) {
        windowInput.value = defaults.window_seconds;
    }
    if (targetFpsInput && defaults.target_fps !== undefined) {
        targetFpsInput.value = defaults.target_fps;
    }
    if (sampledInput && defaults.sampled_frame_count !== undefined) {
        sampledInput.value = defaults.sampled_frame_count;
    }
}

function useLightLiveModel() {
    if (!availableModelsByName[LIGHT_LIVE_MODEL_NAME]) {
        alert('El modelo ligero de 20 cuadros no está disponible.');
        return;
    }

    selectModel(LIGHT_LIVE_MODEL_NAME);
    document.getElementById('live-detection-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function updateLightLiveShortcut() {
    const shortcut = document.getElementById('use-light-live-model');
    if (!shortcut) return;

    const available = Boolean(availableModelsByName[LIGHT_LIVE_MODEL_NAME]);
    shortcut.disabled = !available;
    shortcut.title = available
        ? 'Selecciona el modelo ligero y aplica los parámetros recomendados para CPU.'
        : 'El modelo ligero de 20 cuadros no está disponible.';
}

function focusVideoSelectionSection() {
    // Ensure the default video tab is visible when user picks a model.
    switchTab('library');

    // Scroll to the video section to continue the guided flow (Model -> Video -> Analyze).
    const videosGrid = document.getElementById('videos-grid');
    const videoSection = videosGrid?.closest('.section');
    videoSection?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Load Videos
async function loadVideos() {
    const grid = document.getElementById('videos-grid');
    if (!grid) return;
    grid.innerHTML = '<div class="loading">Cargando videos...</div>';
    
    try {
        const response = await fetch('/api/videos');
        const data = await response.json();
        
        if (data.success) {
            availableVideos = data.videos || [];
            renderVideos();
        } else {
            grid.innerHTML = `<div class="error">No se pudieron cargar los videos: ${data.error}</div>`;
        }
    } catch (error) {
        console.error('Error loading videos:', error);
        grid.innerHTML = '<div class="error">No se pudieron cargar los videos.</div>';
    }
}

function getVideoFilters() {
    return {
        query: (document.getElementById('video-search')?.value || '').trim().toLowerCase(),
        source: document.getElementById('video-source-filter')?.value || 'all',
    };
}

function filterVideos(videos) {
    const { query, source } = getVideoFilters();
    return videos.filter(video => {
        const haystack = `${video.name || ''} ${video.path || ''}`.toLowerCase();
        const matchesQuery = !query || haystack.includes(query);
        const isLocal = !haystack.includes('youtube')
            && !haystack.includes('web_downloads')
            && !haystack.includes('final_tests');
        const matchesSource = source === 'all' || (source === 'local' ? isLocal : haystack.includes(source));
        return matchesQuery && matchesSource;
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
    const grid = document.getElementById('videos-grid');
    if (!grid) return;
    const filteredVideos = filterVideos(videos);
    const visibleVideos = filteredVideos.slice(0, MAX_VISIBLE_VIDEOS);
    const countEl = document.getElementById('video-results-count');
    if (countEl) {
        countEl.textContent = filteredVideos.length > MAX_VISIBLE_VIDEOS
            ? `Mostrando ${MAX_VISIBLE_VIDEOS} de ${filteredVideos.length}; refiná la búsqueda`
            : `${filteredVideos.length} de ${videos.length}`;
    }
    
    if (filteredVideos.length === 0) {
        grid.innerHTML = '<div class="loading">No hay videos que coincidan con la búsqueda.</div>';
        return;
    }
    
    grid.innerHTML = visibleVideos.map(video => {
        // Separate filename from subfolder path (e.g. "youtube/Pablo Granitto/video.mp4")
        const nameParts = video.name.replace(/\\/g, '/').split('/');
        const filename = nameParts[nameParts.length - 1];
        const subfolder = nameParts.length > 1 ? nameParts.slice(0, -1).join(' / ') : '';

        const escapedPath = video.path.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
        return `
            <div class="video-card" data-video="${video.path}">
                <button type="button" class="video-card-body" aria-pressed="${selectedVideo === video.path}" onclick="selectVideo('${escapedPath}')">
                    <h4>${filename}</h4>
                    ${subfolder ? `<div class="video-subfolder">${subfolder}</div>` : ''}
                    <div class="video-size">Duración: ${formatVideoDuration(video.duration)} · ${(video.size / (1024 * 1024)).toFixed(1)} MB</div>
                </button>
                <div class="video-card-actions">
                    <button class="btn btn-secondary" onclick="event.stopPropagation(); processVideo('${escapedPath}')">Procesar</button>
                    <button class="btn" onclick="event.stopPropagation(); selectVideo('${escapedPath}'); previewPreprocessing();">Previsualizar</button>
                </div>
            </div>
        `;
    }).join('');
}

// Load processed videos list
async function loadProcessedVideos() {
    const grid = document.getElementById('processed-videos-grid');
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
    const grid = document.getElementById('processed-videos-grid');
    if (!grid) return;
    const canAnalyzeHere = Boolean(document.getElementById('run-analysis'));
    const filteredVideos = filterVideos(videos);

    if (filteredVideos.length === 0) {
        grid.innerHTML = '<div class="loading">No hay videos procesados que coincidan.</div>';
        return;
    }

    grid.innerHTML = filteredVideos.slice(0, MAX_VISIBLE_VIDEOS).map(v => {
        const escapedPath = v.path.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
        const eligible = v.analysis_eligible !== false;
        const resolution = v.width && v.height ? `${v.width}\u00d7${v.height}` : '';
        const warning = eligible ? '' : `
            <div class="video-analysis-warning" role="note">
                No apto para inferencia: ya est\u00e1 recortado. Us\u00e1 el original de Biblioteca.
            </div>`;
        return `
        <div class="video-card${eligible ? '' : ' video-card-disabled'}" data-video="${v.path}">
            <button type="button" class="video-card-body" aria-pressed="${eligible && selectedVideo === v.path}" ${eligible ? `onclick="selectVideo('${escapedPath}')"` : 'disabled'}>
                <h4>${v.name}</h4>
                <div class="video-size">Duración: ${formatVideoDuration(v.duration)} · ${(v.size / (1024*1024)).toFixed(1)} MB${resolution ? ` \u00b7 ${resolution}` : ''}</div>
                ${warning}
            </button>
            <div class="video-card-actions">
                <button class="btn btn-primary" ${eligible ? `onclick="event.stopPropagation(); selectVideo('${escapedPath}');"` : 'disabled'}>Seleccionar</button>
                ${canAnalyzeHere ? `<button class="btn btn-secondary" ${eligible ? `onclick="event.stopPropagation(); analyzeProcessed('${escapedPath}');"` : 'disabled'}>Analizar</button>` : ''}
            </div>
        </div>
    `;
    }).join('');
}

function analyzeProcessed(processedPath) {
    if (!document.getElementById('run-analysis')) {
        selectVideo(processedPath);
        return;
    }

    // Select the processed video
    selectedVideo = processedPath;
    document.querySelectorAll('.video-card').forEach(card => {
        card.classList.toggle('selected', card.dataset.video === processedPath);
    });
    validateInputs();

    // If no model selected yet, scroll to the model section and warn
    if (!selectedModel) {
        document.getElementById('models-grid')?.scrollIntoView({ behavior: 'smooth' });
        alert('Please select a model first, then click Analyze again.');
        return;
    }

    // Scroll to analysis panel and run
    document.getElementById('run-analysis')?.scrollIntoView({ behavior: 'smooth' });
    runAnalysis();
}

function selectVideo(videoPath) {
    selectedVideo = videoPath;
    
    document.querySelectorAll('.video-card').forEach(card => {
        card.classList.toggle('selected', card.dataset.video === videoPath);
        card.querySelector('.video-card-body')?.setAttribute('aria-pressed', card.dataset.video === videoPath ? 'true' : 'false');
    });

    focusParametersSection();
    
    validateInputs();
    renderLiveContext();
}

function focusParametersSection() {
    const paramsSection = document.getElementById('parameters-section');
    paramsSection?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// File Upload
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        uploadFile(file);
    }
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('video', file);
    
    const progressDiv = document.getElementById('upload-progress');
    const progressFill = document.getElementById('progress-fill');
    const statusText = document.getElementById('upload-status');
    
    progressDiv.style.display = 'block';
    statusText.textContent = 'Uploading...';
    progressFill.style.width = '0%';
    
    try {
        const xhr = new XMLHttpRequest();
        
        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percent = (e.loaded / e.total) * 100;
                progressFill.style.width = percent + '%';
                statusText.textContent = `Uploading... ${percent.toFixed(0)}%`;
            }
        });
        
        xhr.addEventListener('load', () => {
            if (xhr.status === 200) {
                const data = JSON.parse(xhr.responseText);
                if (data.success) {
                    statusText.textContent = 'Upload complete!';
                    selectedVideo = data.filepath;
                    setTimeout(() => {
                        progressDiv.style.display = 'none';
                        loadVideos(); // Refresh video list
                    }, 1000);
                } else {
                    statusText.textContent = 'Upload failed: ' + data.error;
                }
            } else {
                statusText.textContent = 'Upload failed';
            }
        });
        
        xhr.addEventListener('error', () => {
            statusText.textContent = 'Upload error';
        });
        
        xhr.open('POST', '/api/upload');
        xhr.send(formData);
        
    } catch (error) {
        console.error('Upload error:', error);
        statusText.textContent = 'Upload failed: ' + error.message;
    }
}

// Validation
function validateInputs() {
    const runButton = document.getElementById('run-analysis');
    const previewButton = document.getElementById('preview-preprocessing');
    const startLiveButton = document.getElementById('start-live-detection');
    const message = document.getElementById('validation-message');
    
    // Enable preview button if video is selected (don't need model)
    if (selectedVideo) {
        if (previewButton) previewButton.disabled = false;
    } else {
        if (previewButton) previewButton.disabled = true;
    }
    
    if (!selectedModel) {
        if (runButton) runButton.disabled = true;
        if (startLiveButton) startLiveButton.disabled = true;
        if (message) message.textContent = 'Elegí un modelo';
        return false;
    }
    
    if (!selectedVideo) {
        if (runButton) runButton.disabled = true;
        if (startLiveButton) startLiveButton.disabled = true;
        if (message) message.textContent = 'Elegí o subí un video';
        return false;
    }
    
    if (runButton) runButton.disabled = false;
    if (startLiveButton) startLiveButton.disabled = false;
    if (message) message.textContent = 'Listo para analizar';
    return true;
}

function setLiveStatus({ status, jobId = '—', progress = '0%', latest = '—' }) {
    document.getElementById('live-status-text').textContent = status;
    document.getElementById('live-job-id').textContent = jobId;
    document.getElementById('live-progress-text').textContent = progress;
    document.getElementById('live-latest-prediction').textContent = latest;
}

function renderLiveContext() {
    if (!document.getElementById('live-detection-section')) return;
    const modelText = selectedModel || '—';
    const videoName = selectedVideo ? selectedVideo.split(/[\\/]/).pop() : '—';
    const friendlyModelText = selectedModel
        ? modelDisplayName(availableModelsByName[selectedModel] || { name: selectedModel })
        : modelText;
    document.getElementById('live-context-model').textContent = friendlyModelText;
    document.getElementById('live-context-video').textContent = videoName;
    document.getElementById('live-video-context').textContent = selectedVideo
        ? 'El video seleccionado está listo para reproducirse'
        : 'Elegí un modelo y un video';

    const videoEl = document.getElementById('live-video-preview');
    if (videoEl && selectedVideo) {
        const nextSrc = `/api/video-by-path?video_path=${encodeURIComponent(selectedVideo)}`;
        if (videoEl.dataset.src !== nextSrc) {
            videoEl.dataset.src = nextSrc;
            videoEl.src = nextSrc;
            videoEl.load();
            videoEl.pause();
            try {
                videoEl.currentTime = 0;
            } catch (_) {}
        }
    }
}

function waitForVideoMetadata(videoEl) {
    if (!videoEl || videoEl.readyState >= 1) {
        return Promise.resolve();
    }

    return new Promise(resolve => {
        const cleanup = () => {
            videoEl.removeEventListener('loadedmetadata', done);
            videoEl.removeEventListener('error', done);
            clearTimeout(timeoutId);
        };
        const done = () => {
            cleanup();
            resolve();
        };
        const timeoutId = setTimeout(done, 1500);
        videoEl.addEventListener('loadedmetadata', done, { once: true });
        videoEl.addEventListener('error', done, { once: true });
    });
}

async function restartLivePreviewPlayback() {
    const videoEl = document.getElementById('live-video-preview');
    if (!videoEl || !selectedVideo) {
        return;
    }

    const nextSrc = `/api/video-by-path?video_path=${encodeURIComponent(selectedVideo)}`;
    if (videoEl.dataset.src !== nextSrc) {
        videoEl.dataset.src = nextSrc;
        videoEl.src = nextSrc;
        videoEl.load();
    }

    videoEl.pause();
    videoEl.playbackRate = 1;
    await waitForVideoMetadata(videoEl);

    try {
        videoEl.currentTime = 0;
    } catch (_) {}

    try {
        await videoEl.play();
    } catch (error) {
        console.warn('Live preview playback did not start automatically:', error);
        document.getElementById('live-video-context').textContent = 'Presioná reproducir para iniciar la previsualización';
    }
}

function appendLivePrediction(historyItem) {
    const last = livePredictionHistory[livePredictionHistory.length - 1];
    if (last && Math.abs(last.time - historyItem.time) < 0.001) {
        return;
    }
    livePredictionHistory.push(historyItem);
    if (livePredictionHistory.length > 40) {
        livePredictionHistory = livePredictionHistory.slice(-40);
    }
    renderLiveHistory();
    renderLivePredictionForPlayback();
}

function getLivePredictionForTime(videoTime) {
    if (!livePredictionHistory.length || !Number.isFinite(videoTime)) {
        return null;
    }

    const matches = livePredictionHistory.filter(item => {
        const start = Number.isFinite(item.window_start) ? item.window_start : item.time;
        const end = Number.isFinite(item.window_end) ? item.window_end : item.time;
        return videoTime >= start && videoTime <= end;
    });

    if (matches.length) {
        return matches.sort((a, b) => (b.window_start ?? b.time) - (a.window_start ?? a.time))[0];
    }

    const previous = livePredictionHistory
        .filter(item => {
            const end = Number.isFinite(item.window_end) ? item.window_end : item.time;
            return end <= videoTime;
        })
        .sort((a, b) => {
            const aEnd = Number.isFinite(a.window_end) ? a.window_end : a.time;
            const bEnd = Number.isFinite(b.window_end) ? b.window_end : b.time;
            return bEnd - aEnd;
        });

    if (previous.length) {
        return previous[0];
    }

    return [...livePredictionHistory].sort((a, b) => {
        const aStart = Number.isFinite(a.window_start) ? a.window_start : a.time;
        const bStart = Number.isFinite(b.window_start) ? b.window_start : b.time;
        return Math.abs(aStart - videoTime) - Math.abs(bStart - videoTime);
    })[0] || null;
}

function renderLivePredictionForPlayback() {
    const videoEl = document.getElementById('live-video-preview');
    const predictionEl = document.getElementById('live-latest-prediction');
    const probsEl = document.getElementById('live-latest-probs');
    if (!videoEl || !predictionEl || !probsEl) {
        return;
    }

    const active = getLivePredictionForTime(videoEl.currentTime || 0);
    if (!active) {
        predictionEl.textContent = livePredictionHistory.length
            ? `No prediction for ${formatTime(videoEl.currentTime || 0)} yet`
            : '—';
        probsEl.textContent = '—';
        renderLiveHistory();
        return;
    }

    predictionEl.textContent = `${active.play_type} ${(active.confidence * 100).toFixed(1)}% · ${formatTime(active.window_start)}-${formatTime(active.window_end)}`;
    if (active.probabilities) {
        probsEl.textContent = Object.entries(active.probabilities)
            .sort((a, b) => b[1] - a[1])
            .map(([label, p]) => `${label}: ${(p * 100).toFixed(1)}%`)
            .join(' · ');
    } else {
        probsEl.textContent = '—';
    }
    renderLiveHistory();
}

function renderLiveHistory() {
    const container = document.getElementById('live-prediction-history');
    const note = document.getElementById('live-history-note');
    if (!container || !note) return;

    if (!livePredictionHistory.length) {
        note.textContent = 'Todavía no hay predicciones';
        container.innerHTML = '<div class="live-history-empty">Iniciá una ejecución para ver las predicciones.</div>';
        return;
    }

    note.textContent = `${livePredictionHistory.length} updates`;
    const videoEl = document.getElementById('live-video-preview');
    const active = getLivePredictionForTime(videoEl?.currentTime || 0);
    const rows = [...livePredictionHistory].reverse();
    container.innerHTML = rows.map(item => `
        <div class="live-history-item ${item.play_type} ${active && Math.abs(active.time - item.time) < 0.001 ? 'active' : ''}">
            <div class="left">
                <span class="badge ${item.play_type}">${item.play_type.toUpperCase()}</span>
                <span class="time">@ ${formatTime(item.time)}</span>
            </div>
            <div class="right">
                <strong>${(item.confidence * 100).toFixed(1)}%</strong>
                <small>${formatTime(item.window_start)}-${formatTime(item.window_end)}</small>
            </div>
        </div>
    `).join('');
}

function basename(pathValue) {
    if (!pathValue) return 'Origen desconocido';
    return String(pathValue).replace(/\\/g, '/').split('/').pop();
}

function friendlyModelName(modelName) {
    if (!modelName) return 'Modelo seleccionado';
    if (String(modelName).includes('best_model/best_model')) return 'ResNet-50 + BiGRU';
    if (String(modelName).includes('frames20')) return 'Modelo ligero para CPU';
    return basename(modelName);
}

function formatSessionDate(value) {
    if (!value) return 'Fecha desconocida';
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString('es-AR');
}

async function loadTrainingDatasetSummary() {
    const totalEl = document.getElementById('training-dataset-total');
    const sessionsEl = document.getElementById('training-dataset-sessions');
    const correctEl = document.getElementById('training-dataset-correct');
    const correctedEl = document.getElementById('training-dataset-corrected');
    const rootEl = document.getElementById('training-dataset-root');
    const manifestEl = document.getElementById('training-dataset-manifest');
    const reviewLogEl = document.getElementById('training-dataset-review-log');
    const labelsEl = document.getElementById('training-dataset-label-counts');
    const outcomesEl = document.getElementById('training-dataset-outcome-counts');
    const originsEl = document.getElementById('training-dataset-origin-counts');
    const recentEl = document.getElementById('training-dataset-recent-list');
    const clipOriginsEl = document.getElementById('training-dataset-clip-origin-list');

    if (!totalEl || !sessionsEl || !correctEl || !correctedEl || !rootEl || !manifestEl || !reviewLogEl || !labelsEl || !outcomesEl || !originsEl || !recentEl || !clipOriginsEl) {
        return;
    }

    totalEl.textContent = '...';
    sessionsEl.textContent = '...';
    correctEl.textContent = '...';
    correctedEl.textContent = '...';

    try {
        const response = await fetch('/api/training_dataset/summary');
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || 'Failed to load training dataset summary');
        }

        trainingDatasetSummary = data.summary;
        const summary = trainingDatasetSummary;
        totalEl.textContent = summary.total_clips ?? 0;
        sessionsEl.textContent = summary.session_count ?? 0;
        correctEl.textContent = summary.correct_clips ?? 0;
        correctedEl.textContent = summary.corrected_clips ?? 0;
        rootEl.textContent = summary.dataset_root || '—';
        manifestEl.textContent = summary.manifest_path || '—';
        reviewLogEl.textContent = summary.review_log_path || '—';

        const labelCounts = summary.label_counts || {};
        const maxLabelCount = Math.max(1, ...['scrum', 'line', 'other'].map(label => labelCounts[label] || 0));
        labelsEl.innerHTML = ['scrum', 'line', 'other']
            .map(label => `
                <div class="dataset-bar-row">
                    <span>${label.toUpperCase()}</span>
                    <div class="dataset-bar-track"><span class="dataset-bar ${label}" style="width:${((labelCounts[label] || 0) / maxLabelCount) * 100}%"></span></div>
                    <strong>${labelCounts[label] || 0}</strong>
                </div>`)
            .join('');

        const outcomeCounts = summary.review_outcome_counts || {};
        outcomesEl.innerHTML = [`correctas: ${outcomeCounts.correct || 0}`, `corregidas: ${outcomeCounts.corrected || 0}`]
            .map(item => `<span class="dataset-pill neutral">${item}</span>`)
            .join(' ');

        const origins = summary.origins || [];
        originsEl.innerHTML = origins.length
            ? origins.map(origin => {
                const originLabelCounts = origin.label_counts || {};
                return `
                    <div class="dataset-origin-item" title="${origin.source_video || ''}">
                        <div>
                            <strong>${origin.creator || 'Origen desconocido'}</strong>
                            <span>${origin.video_name || 'Video desconocido'}</span>
                        </div>
                        <small>${origin.clip_count || 0} fragmentos · scrum ${originLabelCounts.scrum || 0} · line ${originLabelCounts.line || 0} · other ${originLabelCounts.other || 0}</small>
                    </div>
                `;
            }).join('')
            : '<div class="dataset-empty">No hay videos de origen registrados.</div>';

        const recentClips = summary.recent_clips || [];
        clipOriginsEl.innerHTML = recentClips.length
            ? recentClips.map(clip => `
                <div class="dataset-recent-item" title="${clip.source_video || ''}">
                    <div class="dataset-recent-main">
                        <strong>${clip.clip_name || clip.dataset_id || 'clip'}</strong>
                        <span>${clip.creator || 'Origen desconocido'} · ${clip.video_name || 'Video desconocido'}</span>
                    </div>
                    <div class="dataset-recent-meta">
                        <span>${clip.correct_label || 'other'}</span>
                        <span>${clip.start_time || '?'}s-${clip.end_time || '?'}s</span>
                        <span>${clip.review_outcome || 'reviewed'}</span>
                    </div>
                </div>
            `).join('')
            : '<div class="dataset-empty">Todavía no hay fragmentos.</div>';

        const recentSessions = summary.recent_sessions || [];
        if (!recentSessions.length) {
            recentEl.innerHTML = '<div class="dataset-empty">Todavía no hay exportaciones.</div>';
            return;
        }

        recentEl.innerHTML = recentSessions.map(session => `
            <div class="dataset-recent-item">
                <div class="dataset-recent-main">
                    <strong>${formatSessionDate(session.last_export_at)} · ${session.exported_count || 0} fragmentos</strong>
                    <span>${session.model_name || 'selected model'} · ${session.source_video || 'unknown source'}</span>
                </div>
                <div class="dataset-recent-meta">
                    <span>${session.last_export_at || '—'}</span>
                    <span>${session.correct_count || 0} correctas</span>
                    <span>${session.correct_count || 0} correct · ${session.corrected_count || 0} corrected</span>
                </div>
            </div>
        `).join('');
        recentEl.innerHTML = recentSessions.map(session => `
            <div class="dataset-recent-item">
                <div class="dataset-recent-main">
                    <strong>${formatSessionDate(session.last_export_at)} · ${session.exported_count || 0} fragmentos</strong>
                    <span>${friendlyModelName(session.model_name)} · ${basename(session.source_video)}</span>
                </div>
                <div class="dataset-recent-meta">
                    <span>${session.correct_count || 0} correctas</span>
                    <span>${session.corrected_count || 0} corregidas</span>
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Training dataset summary error:', error);
        totalEl.textContent = '0';
        sessionsEl.textContent = '0';
        correctEl.textContent = '0';
        correctedEl.textContent = '0';
        rootEl.textContent = 'Unavailable';
        manifestEl.textContent = 'Unavailable';
        reviewLogEl.textContent = 'Unavailable';
        labelsEl.innerHTML = '<div class="dataset-empty">Failed to load dataset summary.</div>';
        outcomesEl.innerHTML = '<div class="dataset-empty">Failed to load dataset summary.</div>';
        originsEl.innerHTML = '<div class="dataset-empty">Failed to load dataset summary.</div>';
        recentEl.innerHTML = '<div class="dataset-empty">Failed to load dataset summary.</div>';
        clipOriginsEl.innerHTML = '<div class="dataset-empty">Failed to load dataset summary.</div>';
    }
}

function startLivePolling() {
    stopLivePolling();
    liveStatusPollTimer = setInterval(() => {
        refreshLiveStatus();
    }, 2000);
}

function stopLivePolling() {
    if (liveStatusPollTimer) {
        clearInterval(liveStatusPollTimer);
        liveStatusPollTimer = null;
    }
}

function startLivePlaybackSync() {
    stopLivePlaybackSync();
    syncLivePlaybackTime();
    livePlaybackSyncTimer = setInterval(syncLivePlaybackTime, 250);
}

function stopLivePlaybackSync() {
    if (livePlaybackSyncTimer) {
        clearInterval(livePlaybackSyncTimer);
        livePlaybackSyncTimer = null;
    }
}

async function syncLivePlaybackTime() {
    if (!currentLiveJobId) {
        return;
    }

    const videoEl = document.getElementById('live-video-preview');
    if (!videoEl) {
        return;
    }

    try {
        await fetch(`/api/live_detection/playback/${currentLiveJobId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                current_time: videoEl.currentTime || 0,
                playing: !videoEl.paused && !videoEl.ended,
            }),
            keepalive: true,
        });
    } catch (error) {
        console.warn('Could not sync live playback time:', error);
    }
}

function updateLiveButtons(running) {
    const startBtn = document.getElementById('start-live-detection');
    const stopBtn = document.getElementById('stop-live-detection');
    const refreshBtn = document.getElementById('refresh-live-status');
    if (startBtn) startBtn.disabled = running || !selectedModel || !selectedVideo;
    if (stopBtn) stopBtn.disabled = !running;
    if (refreshBtn) refreshBtn.disabled = !currentLiveJobId;
}

async function startLiveDetection() {
    if (!selectedModel || !selectedVideo) {
        alert('Elegí un modelo y un video.');
        return;
    }

    try {
        stopLivePolling();
        stopLivePlaybackSync();
        currentLiveJobId = null;
        livePredictionHistory = [];
        renderLiveHistory();
        renderLiveContext();
        updateLiveButtons(true);
        setLiveStatus({ status: 'Iniciando...', jobId: '—', progress: '0%', latest: '—' });
        document.getElementById('live-latest-probs').textContent = '—';
        document.getElementById('live-detections-count').textContent = '0';
        document.getElementById('live-processed-time').textContent = '0:00 / 0:00';
        document.getElementById('live-progress-fill').style.width = '0%';

        const windowSeconds = parseFloat(document.getElementById('live-window-seconds').value || '7');
        const predictionInterval = parseFloat(document.getElementById('live-prediction-interval').value || '1');
        const targetFps = parseFloat(document.getElementById('live-target-fps').value || '25');
        const sampledFrameCount = parseInt(document.getElementById('live-sampled-frame-count').value || '0', 10);
        const simulateRealtime = document.getElementById('live-simulate-realtime')?.checked !== false;

        const body = {
            model_name: selectedModel,
            video_path: selectedVideo,
            parameters: {
                confidence_threshold: parseFloat(document.getElementById('confidence-threshold').value),
                line_confidence_threshold: parseFloat(document.getElementById('line-confidence-threshold').value),
                window_seconds: windowSeconds,
                prediction_interval_seconds: predictionInterval,
                target_fps: targetFps,
                sampled_frame_count: sampledFrameCount,
                simulate_realtime: simulateRealtime,
            },
        };

        const response = await fetch('/api/live_detection/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Failed to start live detection');
        }

        currentLiveJobId = data.job_id;
        await restartLivePreviewPlayback();
        startLivePlaybackSync();
        setLiveStatus({
            status: data.status || 'queued',
            jobId: currentLiveJobId,
            progress: '0%',
            latest: '—',
        });
        updateLiveButtons(true);
        startLivePolling();
    } catch (error) {
        console.error('Live detection start error:', error);
        alert(`Live detection failed to start: ${error.message}`);
        updateLiveButtons(false);
        setLiveStatus({ status: 'Error', jobId: '—', progress: '0%', latest: error.message });
    }
}

async function refreshLiveStatus() {
    if (!currentLiveJobId) {
        return;
    }

    try {
        const response = await fetch(`/api/live_detection/status/${currentLiveJobId}`);
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || 'Failed to fetch live status');
        }

        const job = data.job;
        const latest = job.latest_prediction
            ? `${job.latest_prediction.play_type} ${(job.latest_prediction.confidence * 100).toFixed(1)}% @ ${formatTime(job.latest_prediction.time)}`
            : '—';

        const percent = (job.progress?.percent || 0);
        const currentTime = job.progress?.current_time || 0;
        const duration = job.progress?.duration || 0;
        document.getElementById('live-progress-fill').style.width = `${Math.max(0, Math.min(100, percent)).toFixed(1)}%`;
        document.getElementById('live-detections-count').textContent = `${job.detections_count || 0}`;
        document.getElementById('live-processed-time').textContent = `${formatTime(currentTime)} / ${formatTime(duration)}`;

        const latestProbsEl = document.getElementById('live-latest-probs');
        if (job.latest_prediction?.probabilities) {
            const probText = Object.entries(job.latest_prediction.probabilities)
                .sort((a, b) => b[1] - a[1])
                .map(([label, p]) => `${label}: ${(p * 100).toFixed(1)}%`)
                .join(' · ');
            latestProbsEl.textContent = probText;
        } else {
            latestProbsEl.textContent = '—';
        }

        if (job.latest_prediction) {
            appendLivePrediction(job.latest_prediction);
        }

        setLiveStatus({
            status: job.status,
            jobId: job.job_id,
            progress: `${percent.toFixed(1)}%`,
            latest,
        });
        renderLivePredictionForPlayback();

        const finished = ['completed', 'stopped', 'error'].includes(job.status);
        updateLiveButtons(!finished);
        if (finished) {
            stopLivePolling();
            stopLivePlaybackSync();
            currentLiveJobId = null;
            updateLiveButtons(false);
        }
    } catch (error) {
        console.error('Live detection status error:', error);
        setLiveStatus({ status: 'Status error', jobId: currentLiveJobId, progress: '—', latest: error.message });
        document.getElementById('live-latest-probs').textContent = '—';
        updateLiveButtons(false);
    }
}

async function stopLiveDetection() {
    if (!currentLiveJobId) {
        return;
    }

    try {
        const response = await fetch(`/api/live_detection/stop/${currentLiveJobId}`, {
            method: 'POST',
        });
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || 'Failed to stop live detection');
        }
        setLiveStatus({
            status: data.status || 'stopping',
            jobId: currentLiveJobId,
            progress: '—',
            latest: 'Stopping requested',
        });
        document.getElementById('live-video-preview')?.pause();
        stopLivePlaybackSync();
        refreshLiveStatus();
    } catch (error) {
        console.error('Live detection stop error:', error);
        alert(`Failed to stop live detection: ${error.message}`);
    }
}

function resetParameters() {
    document.getElementById('confidence-threshold').value = 0.70;
    document.getElementById('line-confidence-threshold').value = 0.95;
    document.getElementById('batch-size').value = 8;
    document.getElementById('temporal-fps-mode').value = 'training_match_25';
    document.getElementById('custom-effective-fps').value = 25;
    document.getElementById('custom-effective-fps').style.display = 'none';
    
    // Trigger updates
    document.querySelectorAll('input[type="range"]').forEach(slider => {
        slider.dispatchEvent(new Event('input'));
    });
}

// Run Analysis
async function runAnalysis() {
    if (!validateInputs()) return;
    
    const button = document.getElementById('run-analysis');
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = 'Analizando...';
    analysisStartedAt = Date.now();
    setAnalysisProgress({ percent: 0, stage: 'queued', message: 'Iniciando el análisis' });
    
    const temporalMode = document.getElementById('temporal-fps-mode')?.value || 'training_match_25';
    const customFpsRaw = parseFloat(document.getElementById('custom-effective-fps')?.value || '25');
    const parameters = {
        confidence_threshold: parseFloat(document.getElementById('confidence-threshold').value),
        line_confidence_threshold: parseFloat(document.getElementById('line-confidence-threshold').value),
        batch_size: parseInt(document.getElementById('batch-size').value),
        temporal_fps_mode: temporalMode,
        custom_effective_fps: temporalMode === 'custom' ? customFpsRaw : null,
        feature_batch_size: 256,
    };
    
    try {
        const response = await fetch('/api/analyze/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_name: selectedModel,
                video_path: selectedVideo,
                parameters: parameters,
            }),
        });
        
        const startData = await response.json();
        if (!response.ok || !startData.success) {
            throw new Error(startData.error || 'No se pudo iniciar el análisis');
        }

        currentAnalysisJobId = startData.job_id;
        const data = await waitForAnalysisJob(currentAnalysisJobId);
        currentResultId = data.result_id;
        data.results.video_url = data.video_url || `/api/video-for-result/${currentResultId}`;
        displayResults(data.results);
    } catch (error) {
        console.error('Analysis error:', error);
        setAnalysisProgressError(error.message);
        alert('El análisis falló: ' + error.message);
    } finally {
        currentAnalysisJobId = null;
        button.disabled = false;
        button.textContent = originalText;
    }
}

const ANALYSIS_STAGE_LABELS = {
    queued: 'En cola',
    validating: 'Validando entradas',
    model: 'Cargando modelo',
    preprocessing: 'Preparando video',
    features: 'Extrayendo características',
    inference: 'Detectando jugadas',
    postprocessing: 'Agrupando detecciones',
    saving: 'Guardando resultados',
    completed: 'Análisis completo',
};

function formatAnalysisDuration(totalSeconds) {
    const seconds = Math.max(0, Math.round(totalSeconds));
    const minutes = Math.floor(seconds / 60);
    return `${minutes}:${String(seconds % 60).padStart(2, '0')}`;
}

function setAnalysisProgress(progress = {}) {
    const container = document.getElementById('analysis-progress');
    if (!container) return;

    const percent = Math.max(0, Math.min(100, Number(progress.percent) || 0));
    const elapsed = analysisStartedAt ? (Date.now() - analysisStartedAt) / 1000 : 0;
    const remaining = percent >= 3 && percent < 100
        ? elapsed * (100 - percent) / percent
        : null;
    const stage = progress.stage || 'queued';

    container.hidden = false;
    container.classList.toggle('is-complete', percent >= 100);
    container.classList.remove('is-error');
    document.getElementById('analysis-progress-stage').textContent = ANALYSIS_STAGE_LABELS[stage] || stage;
    document.getElementById('analysis-progress-percent').textContent = `${Math.round(percent)}%`;
    document.getElementById('analysis-progress-fill').style.width = `${percent}%`;
    document.getElementById('analysis-progress-message').textContent = progress.message || 'Procesando...';
    document.getElementById('analysis-progress-track').setAttribute('aria-valuenow', String(Math.round(percent)));
    document.getElementById('analysis-progress-time').textContent = remaining === null
        ? `Transcurrido ${formatAnalysisDuration(elapsed)}`
        : `Transcurrido ${formatAnalysisDuration(elapsed)} · Restante estimado ${formatAnalysisDuration(remaining)}`;
}

function setAnalysisProgressError(message) {
    const container = document.getElementById('analysis-progress');
    if (!container) return;
    container.hidden = false;
    container.classList.add('is-error');
    container.classList.remove('is-complete');
    document.getElementById('analysis-progress-stage').textContent = 'El análisis se interrumpió';
    document.getElementById('analysis-progress-message').textContent = message;
}

function analysisPollDelay(milliseconds) {
    return new Promise(resolve => window.setTimeout(resolve, milliseconds));
}

async function waitForAnalysisJob(jobId) {
    while (currentAnalysisJobId === jobId) {
        const response = await fetch(`/api/analyze/status/${encodeURIComponent(jobId)}`, { cache: 'no-store' });
        const job = await response.json();
        if (!response.ok || !job.success) {
            throw new Error(job.error || 'No se pudo consultar el progreso');
        }

        setAnalysisProgress(job.progress);
        if (job.status === 'completed') return job.response;
        if (job.status === 'error') throw new Error(job.error || 'El análisis terminó con un error');
        await analysisPollDelay(750);
    }
    throw new Error('El seguimiento del análisis fue cancelado');
}

// Display Results
let _currentResults = null;  // cache for filter re-renders
let _baseAnalysisResults = null;  // immutable source for post-classification threshold previews

function displayResults(results, { restored = false, thresholdPreview = false } = {}) {
    _currentResults = results;
    if (!thresholdPreview) {
        _baseAnalysisResults = results;
        initializePostClassificationThresholds(results);
    }
    replayModelRoiController?.setInputMode(roiInputModeForModel(results.model_name));
    currentDetectionFilter = 'all';
    const resultsSection = document.getElementById('results-section');
    resultsSection.style.display = 'block';
    if (!thresholdPreview) resultsSection.scrollIntoView({ behavior: 'smooth' });

    // ── Header with meta info ──────────────────────────────────
    const headerEl = document.getElementById('results-meta-header');
    if (headerEl && !thresholdPreview) {
        const videoName = results.video_path ? results.video_path.split(/[\\/]/).pop() : '—';
        const model = results.model_name || '—';
        const icon = restored ? 'Restored analysis' : 'Analysis';
        headerEl.innerHTML = `
            <span style="font-size:13px;">
                ${icon} — <strong>${videoName}</strong> &nbsp;|&nbsp;
                Model: <strong>${model}</strong>
            </span>
            ${results.meta_correction?.applied ? `
                <div style="margin-top:8px;padding:8px 10px;background:#fff7ed;border:1px solid #fed7aa;border-radius:6px;color:#9a3412;font-size:12px;">
                    <strong>Corrector few-shot activo.</strong>
                    Se conserva la predicción primaria y se aplica una corrección calibrada con
                    ${results.meta_correction.support_rows || 401} clips del mismo video.
                    Para otro partido se necesita soporte humano nuevo.
                </div>` : ''}`;
        headerEl.style.display = 'block';
    }

    // ── Summary cards ──────────────────────────────────────────
    const stats = results.statistics || {};
    const byType = stats.by_type || {};
    const summary = document.getElementById('results-summary');
    const temporalDiagnostics = buildTemporalDiagnosticsHtml(results);
    const timing = results.timing || results.metadata?.timing || {};
    const inferenceSeconds = Number(timing.inference_seconds);
    const realtimeFactor = Number(timing.inference_realtime_factor);
    summary.innerHTML = `
        <div class="summary-card">
            <div class="summary-value">${results.detections.length}</div>
            <div class="summary-label">Total Plays</div>
        </div>
        <div class="summary-card">
            <div class="summary-value">${stats.avg_confidence !== undefined ? (stats.avg_confidence * 100).toFixed(1) + '%' : '—'}</div>
            <div class="summary-label">Avg Confidence</div>
        </div>
        <div class="summary-card">
            <div class="summary-value">${stats.avg_duration !== undefined ? stats.avg_duration.toFixed(1) + 's' : '—'}</div>
            <div class="summary-label">Avg Duration</div>
        </div>
        ${Number.isFinite(inferenceSeconds) ? `
            <div class="summary-card">
                <div class="summary-value">${formatAnalysisDuration(inferenceSeconds)}</div>
                <div class="summary-label">Tiempo de inferencia${Number.isFinite(realtimeFactor) ? ` · ${realtimeFactor.toFixed(2)}× tiempo real` : ''}</div>
            </div>
        ` : ''}
        ${Object.entries(byType).map(([type, info]) => `
            <div class="summary-card" style="cursor:pointer;" onclick="filterDetections('${type}')">
                <div class="summary-value">
                    <span class="badge ${type}">${info.count}</span>
                </div>
                <div class="summary-label">${type.charAt(0).toUpperCase() + type.slice(1)} plays</div>
            </div>
        `).join('')}
        ${temporalDiagnostics}
    `;

    // ── Filter bar ─────────────────────────────────────────────
    const playTypes = [...new Set(results.detections.map(d => d.play_type))].sort();
    const filterBar = document.getElementById('detections-filter');
    if (filterBar) {
        filterBar.innerHTML = `
            <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px;">
                <span style="font-size:13px;color:var(--text-secondary);font-weight:500;">Filter:</span>
                <button class="btn btn-primary filter-btn active" data-type="all"
                    onclick="filterDetections('all')" style="padding:4px 14px;font-size:13px;">All (${results.detections.length})</button>
                ${playTypes.map(type => {
                    const cnt = results.detections.filter(d => d.play_type === type).length;
                    return `<button class="btn btn-secondary filter-btn" data-type="${type}"
                        onclick="filterDetections('${type}')" style="padding:4px 14px;font-size:13px;">
                        <span class="badge ${type}">${type.toUpperCase()}</span> ${cnt}
                    </button>`;
                }).join('')}
            </div>`;
        filterBar.style.display = 'block';
    }

    initializeReplayExperience(results);
    renderDetections(results.detections.map((d, i) => ({ ...d, _origIdx: i })));
}

function buildTemporalDiagnosticsHtml(results) {
    if (!results || !results.temporal_standardization) {
        return '';
    }

    const sourceFps = Number.isFinite(results.source_fps) ? `${results.source_fps.toFixed(2)} fps` : '—';
    const effectiveFps = Number.isFinite(results.effective_fps) ? `${results.effective_fps.toFixed(2)} fps` : '—';
    const sourceFrames = Number.isFinite(results.source_total_frames) ? results.source_total_frames : '—';
    const sampledFrames = Number.isFinite(results.sampled_total_frames) ? results.sampled_total_frames : '—';
    const framesPerWindow = Number.isFinite(results.frames_per_window) ? results.frames_per_window : '—';
    const framesPerStride = Number.isFinite(results.frames_per_stride) ? results.frames_per_stride : '—';

    return `
        <div class="temporal-diagnostics-card">
            <div class="temporal-diagnostics-header">
                <div>
                    <h3>Muestreo temporal</h3>
                    <p>${results.temporal_standardization}</p>
                </div>
            </div>
            <div class="temporal-diagnostics-grid">
                <div class="temporal-metric">
                    <span class="label">Source FPS</span>
                    <span class="value">${sourceFps}</span>
                </div>
                <div class="temporal-metric">
                    <span class="label">Effective FPS</span>
                    <span class="value">${effectiveFps}</span>
                </div>
                <div class="temporal-metric">
                    <span class="label">Raw Frames</span>
                    <span class="value">${sourceFrames}</span>
                </div>
                <div class="temporal-metric">
                    <span class="label">Sampled Frames</span>
                    <span class="value">${sampledFrames}</span>
                </div>
                <div class="temporal-metric">
                    <span class="label">Frames / Window</span>
                    <span class="value">${framesPerWindow}</span>
                </div>
                <div class="temporal-metric">
                    <span class="label">Frames / Stride</span>
                    <span class="value">${framesPerStride}</span>
                </div>
            </div>
        </div>
    `;
}

function filterDetections(type) {
    if (!_currentResults) return;
    currentDetectionFilter = type;
    // Update active button
    document.querySelectorAll('.filter-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.type === type);
        b.classList.toggle('btn-primary', b.dataset.type === type);
        b.classList.toggle('btn-secondary', b.dataset.type !== type);
    });
    // Tag each detection with its original index before filtering
    const tagged = _currentResults.detections.map((d, i) => ({ ...d, _origIdx: i }));
    const list = type === 'all'
        ? tagged
        : tagged.filter(d => d.play_type === type);
    renderDetections(list);
    renderReplayExperience();
}

function renderDetections(detections) {
    const detectionsList = document.getElementById('detections-list');

    if (detections.length === 0) {
        detectionsList.innerHTML = '<p style="color:var(--text-secondary);text-align:center;padding:2rem;">No hay detecciones que coincidan con el filtro.</p>';
        return;
    }

    detectionsList.innerHTML = detections.map((det, idx) => `
        <div class="detection-item ${det.play_type}" data-start-time="${det.start_time}" data-end-time="${det.end_time}" data-play-type="${det.play_type}">
            <div class="detection-info">
                <h4>
                    <span class="badge ${det.play_type}">${det.play_type.toUpperCase()}</span>
                    Play #${det._origIdx !== undefined ? det._origIdx + 1 : idx + 1}
                </h4>
                <div class="detection-time">
                    ${formatTime(det.start_time)} – ${formatTime(det.end_time)}
                    (${det.duration ? det.duration.toFixed(1) : (det.end_time - det.start_time).toFixed(1)}s)
                </div>
                ${det.meta_corrector_applied && det.primary_play_type && det.primary_play_type !== det.play_type
                    ? `<div class="detection-probs"><small>Primaria: ${det.primary_play_type.toUpperCase()} (${(Number(det.primary_confidence || 0) * 100).toFixed(1)}%) → corregida: ${det.play_type.toUpperCase()}</small></div>`
                    : ''}
                <div class="detection-probs">
                    ${det.probabilities
                        ? Object.entries(det.probabilities).map(([type, prob]) =>
                            `<small>${type}: ${(prob * 100).toFixed(1)}%</small>`).join(' | ')
                        : `<small>confidence: ${(det.confidence * 100).toFixed(1)}%</small>`}
                </div>
            </div>
            <div class="detection-actions">
                <button class="btn btn-secondary play-clip-btn"
                    data-detection-index="${det._origIdx !== undefined ? det._origIdx : idx}">
                    Play Clip
                </button>
                <div class="detection-confidence">${(det.confidence * 100).toFixed(1)}%</div>
            </div>
        </div>
    `).join('');

    document.querySelectorAll('.play-clip-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            playDetectionClip(parseInt(e.currentTarget.dataset.detectionIndex));
        });
    });
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Export Results
function exportResults(format) {
    if (!currentResultId) return;
    window.open(`/api/export/${currentResultId}?format=${format}`, '_blank');
}

function viewTimeline() {
    if (!currentResultId) return;
    setReplayFocusMode(true);
}

function toggleReplayFocusMode() {
    const shell = document.getElementById('results-replay-shell');
    if (!shell) return;
    setReplayFocusMode(!shell.classList.contains('is-focus-mode'));
}

function setReplayFocusMode(enabled) {
    const shell = document.getElementById('results-replay-shell');
    const toggle = document.getElementById('results-replay-focus-toggle');
    const openButton = document.getElementById('view-timeline');
    if (!shell) return;

    const wasEnabled = shell.classList.contains('is-focus-mode');
    if (wasEnabled !== enabled) {
        replayState.clipLabelsVisible = false;
    }
    shell.classList.toggle('is-focus-mode', enabled);
    document.body.classList.toggle('replay-focus-open', enabled);
    if (toggle) {
        toggle.textContent = enabled ? 'Cerrar vista amplia' : 'Vista amplia';
        toggle.setAttribute('aria-pressed', String(enabled));
    }
    openButton?.setAttribute('aria-pressed', String(enabled));
    renderReplayClipComparison();

    window.requestAnimationFrame(() => {
        resizeReplayTimeline();
        if (!enabled) {
            shell.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
}

function resetReplayClassVisibility() {
    replayClassVisibility.scrum = true;
    replayClassVisibility.line = true;
    replayClassVisibility.other = true;
}

function isReplayClassVisible(playType) {
    if (!(playType in replayClassVisibility)) return true;
    return replayClassVisibility[playType];
}

function toggleReplayClassVisibility(playType) {
    if (!(playType in replayClassVisibility)) return;

    const visibleCount = Object.values(replayClassVisibility).filter(Boolean).length;
    if (replayClassVisibility[playType] && visibleCount <= 1) {
        return;
    }

    replayClassVisibility[playType] = !replayClassVisibility[playType];
    renderReplayExperience();
}

function renderReplayClassToggles() {
    const container = document.getElementById('results-replay-class-toggles');
    if (!container) return;

    const detections = (_currentResults?.detections || []);
    const classCounts = {
        scrum: detections.filter(d => d.play_type === 'scrum').length,
        line: detections.filter(d => d.play_type === 'line').length,
        other: detections.filter(d => d.play_type === 'other').length,
    };

    container.innerHTML = ['scrum', 'line', 'other'].map((playType) => {
        const isOn = !!replayClassVisibility[playType];
        const count = classCounts[playType];
        return `
            <button
                type="button"
                class="replay-class-toggle ${playType} ${isOn ? 'active' : 'inactive'}"
                onclick="toggleReplayClassVisibility('${playType}')"
                title="Show/hide ${playType} in replay timeline"
            >
                <span class="badge ${playType}">${playType.toUpperCase()}</span>
                <span class="replay-class-count">${count}</span>
                <span class="replay-class-state">${isOn ? 'ON' : 'OFF'}</span>
            </button>
        `;
    }).join('');
}

function getReplayDetections() {
    if (!_currentResults) return [];
    const detections = _currentResults.detections || [];
    const filteredByType = currentDetectionFilter === 'all'
        ? detections
        : detections.filter(d => d.play_type === currentDetectionFilter);
    return filteredByType.filter(d => isReplayClassVisible(d.play_type));
}

async function loadReplayClipComparison(resultId) {
    replayState.clipComparisonResultId = resultId;
    replayState.clipComparison = null;
    replayState.clipComparisonLoading = true;
    replayState.clipLabelsVisible = false;
    renderReplayClipComparison();

    try {
        const response = await fetch(`/api/result/${encodeURIComponent(resultId)}/clip-labels`, { cache: 'no-store' });
        const data = await response.json();
        if (!response.ok || !data.success) throw new Error(data.error || 'No se pudieron cargar las etiquetas reales');
        if (replayState.clipComparisonResultId !== resultId) return;
        replayState.clipComparison = data.available ? data.comparison : null;
    } catch (error) {
        if (replayState.clipComparisonResultId === resultId) {
            console.warn('No se pudo cargar el timeline del generador de clips:', error);
            replayState.clipComparison = null;
        }
    } finally {
        if (replayState.clipComparisonResultId === resultId) {
            replayState.clipComparisonLoading = false;
            renderReplayClipComparison();
        }
    }
}

function toggleReplayClipLabels() {
    if (!replayState.clipComparison) return;
    replayState.clipLabelsVisible = !replayState.clipLabelsVisible;
    renderReplayClipComparison();
    window.requestAnimationFrame(resizeReplayTimeline);
}

function renderReplayClipComparison() {
    const shell = document.getElementById('results-replay-shell');
    const toggle = document.getElementById('results-replay-clip-labels-toggle');
    const panel = document.getElementById('results-replay-clip-labels');
    const summary = document.getElementById('results-replay-clip-labels-summary');
    const link = document.getElementById('results-replay-clip-labels-link');
    if (!shell || !toggle || !panel) return;

    const comparison = replayState.clipComparison;
    const available = !!comparison && !replayState.clipComparisonLoading;
    const expanded = available
        && shell.classList.contains('is-focus-mode')
        && replayState.clipLabelsVisible;
    toggle.hidden = !available;
    toggle.textContent = expanded ? 'Ocultar etiquetas reales' : 'Mostrar etiquetas reales';
    toggle.setAttribute('aria-expanded', String(expanded));
    panel.hidden = !expanded;

    if (!available) return;
    const reviewed = Number(comparison.summary?.reviewed) || 0;
    const total = Number(comparison.summary?.total_candidates) || comparison.labels?.length || 0;
    const percentage = comparison.agreement?.percentage;
    const split = comparison.composite_reference ? 'Referencia consolidada' : comparison.dataset_split === 'test' ? 'Test' : 'Training';
    const counts = comparison.summary?.final_counts || {};
    summary.textContent = `${split} · ${reviewed} de ${total} etiquetados · ${Number(counts.line) || 0} line · ${Number(counts.scrum) || 0} scrum${percentage === null || percentage === undefined ? '' : ` · ${percentage}% coincide con el análisis`}`;
    link.href = comparison.clip_generator_url || '/clip-generator';
    link.textContent = comparison.composite_reference ? 'Editar última sesión' : 'Abrir generación';
    link.title = comparison.composite_reference ? 'La referencia reúne las sesiones anteriores y sus correcciones. Este enlace abre la última sesión para editarla.' : '';
    if (expanded) drawReplayClipLabelsTimeline();
}

function initializeReplayExperience(results) {
    const shell = document.getElementById('results-replay-shell');
    const video = document.getElementById('results-replay-video');
    if (!shell || !video || !currentResultId || !results?.video_path) {
        shell && (shell.style.display = 'none');
        return;
    }

    shell.style.display = 'grid';
    bindReplayControls();
    renderReplayStats(results);
    resetReplayStatus();
    renderReplayClassToggles();

    if (replayState.clipComparisonResultId !== currentResultId) {
        loadReplayClipComparison(currentResultId);
    }

    if (video.dataset.resultId !== currentResultId) {
        video.dataset.resultId = currentResultId;
        replayState.duration = 0;
        replayState.initialSeekApplied = false;
        replayState.fallbackAttempted = false;
        replayState.sourcePath = results.video_path;
        resetReplayClassVisibility();
        setReplayVideoMessage('Cargando video...');
        // The result payload can also contain a legacy/preprocessed video_url.
        // Integrated replay must always request the original playback source;
        // the 224x224 representation has its own explicit viewer control.
        video.src = `/api/video-for-result/${encodeURIComponent(currentResultId)}?view=original&v=2`;
        video.load();
    } else if (video.error || video.readyState === 0) {
        setReplayVideoMessage('Reintentando la carga del video...');
        video.load();
    }

    renderReplayExperience();
}

function bindReplayControls() {
    if (replayState.controlsBound) return;
    replayState.controlsBound = true;

    const video = document.getElementById('results-replay-video');
    const seek = document.getElementById('results-replay-seek');
    const timeline = document.getElementById('results-replay-timeline');
    const clipLabelsTimeline = document.getElementById('results-replay-clip-labels-timeline');
    replayModelRoiController = window.ModelRoiOverlay?.attach({
        video,
        stage: document.getElementById('results-replay-roi-stage'),
        overlay: document.getElementById('results-replay-model-roi'),
        toggle: document.getElementById('show-results-replay-model-roi'),
        inputMode: roiInputModeForModel(_currentResults?.model_name),
    });

    video?.addEventListener('loadstart', () => {
        setReplayVideoMessage('Cargando video...');
    });

    video?.addEventListener('loadedmetadata', () => {
        replayState.duration = video.duration || 0;
        seek.max = replayState.duration || 0;
        document.getElementById('results-replay-duration').textContent = formatTime(replayState.duration);
        if (!replayState.initialSeekApplied) {
            const firstDetection = getReplayDetections()[0];
            if (firstDetection && Number.isFinite(firstDetection.start_time)) {
                video.currentTime = Math.min(firstDetection.start_time + 0.01, replayState.duration || firstDetection.start_time);
                document.getElementById('results-replay-current-time').textContent = formatTime(video.currentTime);
            }
            replayState.initialSeekApplied = true;
        }
        setReplayVideoMessage('', 'ready');
        resizeReplayTimeline();
        renderReplayExperience();
    });

    video?.addEventListener('canplay', () => {
        setReplayVideoMessage('', 'ready');
    });

    video?.addEventListener('stalled', () => {
        setReplayVideoMessage('La carga del video está demorando. Esperando datos...');
    });

    video?.addEventListener('error', () => {
        if (!replayState.fallbackAttempted && replayState.sourcePath) {
            replayState.fallbackAttempted = true;
            setReplayVideoMessage('Reintentando desde el archivo de origen...');
            video.src = `/api/video-by-path?video_path=${encodeURIComponent(replayState.sourcePath)}`;
            video.load();
            return;
        }
        const detail = video.error?.message ? ` (${video.error.message})` : '';
        setReplayVideoMessage(`No se pudo cargar el video${detail}.`, 'error');
    });

    video?.addEventListener('timeupdate', () => {
        seek.value = video.currentTime;
        document.getElementById('results-replay-current-time').textContent = formatTime(video.currentTime);
        renderReplayExperience();
    });

    video?.addEventListener('play', startReplayAnimationLoop);
    video?.addEventListener('pause', stopReplayAnimationLoop);
    video?.addEventListener('ended', stopReplayAnimationLoop);

    document.getElementById('results-replay-play-pause')?.addEventListener('click', () => {
        if (!video) return;
        if (video.paused) {
            const playRequest = video.play();
            playRequest?.catch(() => {
                setReplayVideoMessage('El navegador bloqueó la reproducción automática. Usá el control de reproducción del video.');
            });
        } else {
            video.pause();
        }
    });

    document.getElementById('results-replay-fullscreen')?.addEventListener('click', () => {
        const stage = document.getElementById('results-replay-roi-stage');
        if (stage?.requestFullscreen) stage.requestFullscreen();
        else if (stage?.webkitRequestFullscreen) stage.webkitRequestFullscreen();
    });

    seek?.addEventListener('input', (event) => {
        if (!video) return;
        video.currentTime = parseFloat(event.target.value || '0');
        renderReplayExperience();
    });

    const seekFromReplayTimeline = (event) => {
        if (!video || replayState.duration <= 0) return;
        const rect = event.currentTarget.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
        video.currentTime = ratio * replayState.duration;
        renderReplayExperience();
    };
    timeline?.addEventListener('click', seekFromReplayTimeline);
    clipLabelsTimeline?.addEventListener('click', seekFromReplayTimeline);
    document.getElementById('results-replay-clip-labels-toggle')?.addEventListener('click', toggleReplayClipLabels);

    window.addEventListener('resize', resizeReplayTimeline);
}

function renderReplayStats(results) {
    const container = document.getElementById('results-replay-stats');
    if (!container) return;

    const stats = results.statistics || {};
    const byType = stats.by_type || {};
    const visibleDetections = getReplayDetections();
    const videoName = results.video_path ? results.video_path.split(/[\\/]/).pop() : '—';
    container.innerHTML = `
        <div class="results-replay-stat-card">
            <span class="label">Video</span>
            <span class="value small">${videoName}</span>
        </div>
        <div class="results-replay-stat-card">
            <span class="label">Modelo</span>
            <span class="value small">${results.model_name || '—'}</span>
        </div>
        <div class="results-replay-stat-card">
            <span class="label">Detecciones visibles</span>
            <span class="value">${visibleDetections.length}</span>
        </div>
        <div class="results-replay-stat-card">
            <span class="label">Confianza promedio</span>
            <span class="value">${stats.avg_confidence !== undefined ? (stats.avg_confidence * 100).toFixed(1) + '%' : '—'}</span>
        </div>
        ${Object.entries(byType).map(([type, info]) => `
            <div class="results-replay-stat-card ${type}">
                <span class="label">${type}</span>
                <span class="value">${info.count}</span>
            </div>
        `).join('')}
    `;
}

function renderReplayExperience() {
    renderReplayClassToggles();
    drawReplayTimeline();
    updateReplayCurrentState();
    renderReplayStats(_currentResults || {});
}

function setReplayVideoMessage(message, state = '') {
    const element = document.getElementById('results-replay-video-message');
    if (!element) return;
    element.textContent = message;
    element.className = `results-replay-video-message ${state}`.trim();
}

function resetReplayStatus() {
    const status = document.getElementById('results-replay-status');
    if (!status) return;
    status.className = 'results-replay-status empty';
    status.textContent = 'Sin detección activa';
}

function updateReplayCurrentState() {
    const video = document.getElementById('results-replay-video');
    const status = document.getElementById('results-replay-status');
    if (!video || !status) return;

    const currentTime = video.currentTime || 0;
    const activeDetection = getReplayDetections().find(det => currentTime >= det.start_time && currentTime <= det.end_time);
    if (!activeDetection) {
        resetReplayStatus();
        updateActiveDetectionHighlight(null);
        return;
    }

    status.className = `results-replay-status ${activeDetection.play_type}`;
    status.textContent = `${activeDetection.play_type.toUpperCase()} · ${(activeDetection.confidence * 100).toFixed(1)}% · ${formatTime(activeDetection.start_time)}-${formatTime(activeDetection.end_time)}`;
    updateActiveDetectionHighlight(activeDetection);
}

function updateActiveDetectionHighlight(activeDetection) {
    document.querySelectorAll('#detections-list .detection-item').forEach(item => {
        const itemStart = parseFloat(item.dataset.startTime || '0');
        const itemEnd = parseFloat(item.dataset.endTime || '0');
        const itemType = item.dataset.playType || '';
        const isActive = !!activeDetection && itemType === activeDetection.play_type && Math.abs(itemStart - activeDetection.start_time) < 0.01 && Math.abs(itemEnd - activeDetection.end_time) < 0.01;
        item.classList.toggle('active', isActive);
    });
}

function resizeReplayTimeline() {
    const canvas = document.getElementById('results-replay-timeline');
    if (!canvas) return;
    resizeReplayCanvas(canvas);
    const clipLabelsCanvas = document.getElementById('results-replay-clip-labels-timeline');
    if (replayState.clipLabelsVisible && !clipLabelsCanvas?.closest('[hidden]')) {
        resizeReplayCanvas(clipLabelsCanvas);
    }
    drawReplayTimeline();
}

function resizeReplayCanvas(canvas) {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * ratio);
    canvas.height = Math.round(rect.height * ratio);
    const ctx = canvas.getContext('2d');
    ctx?.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function drawReplayTimeline() {
    const canvas = document.getElementById('results-replay-timeline');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.offsetWidth || 0;
    const height = canvas.offsetHeight || 0;
    ctx.clearRect(0, 0, width, height);

    ctx.fillStyle = 'rgba(15, 23, 42, 0.08)';
    ctx.beginPath();
    ctx.roundRect(0, 8, width, Math.max(24, height - 16), 6);
    ctx.fill();

    const duration = replayState.duration || 0;
    if (duration <= 0) {
        drawReplayClipLabelsTimeline();
        return;
    }

    const colors = { scrum: '#ef4444', line: '#10b981', other: '#3b82f6' };
    for (const det of getReplayDetections()) {
        const x = (det.start_time / duration) * width;
        const segmentWidth = Math.max(3, ((det.end_time - det.start_time) / duration) * width);
        ctx.fillStyle = colors[det.play_type] || '#64748b';
        ctx.globalAlpha = 0.45 + (det.confidence || 0) * 0.45;
        ctx.beginPath();
        ctx.roundRect(x, 8, segmentWidth, Math.max(24, height - 16), 6);
        ctx.fill();
    }
    ctx.globalAlpha = 1;

    const video = document.getElementById('results-replay-video');
    const currentTime = video?.currentTime || 0;
    const scrubberX = (currentTime / duration) * width;
    ctx.fillStyle = '#0f172a';
    ctx.beginPath();
    ctx.roundRect(Math.max(0, scrubberX - 1.5), 4, 3, Math.max(30, height - 8), 2);
    ctx.fill();
    drawReplayClipLabelsTimeline();
}

function drawReplayClipLabelsTimeline() {
    const canvas = document.getElementById('results-replay-clip-labels-timeline');
    const comparison = replayState.clipComparison;
    if (!canvas || canvas.closest('[hidden]') || !comparison) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.offsetWidth || 0;
    const height = canvas.offsetHeight || 0;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#cbd5e1';
    ctx.fillRect(0, 0, width, height);

    const duration = replayState.duration || Number(_currentResults?.metadata?.duration) || 0;
    if (duration <= 0) return;
    const colors = { line: '#10b981', scrum: '#ef4444', other: '#3b82f6' };
    for (const block of comparison.labels || []) {
        const start = Math.max(0, Number(block.start_time) || 0);
        const end = Math.min(duration, Number(block.end_time) || start);
        const x = (start / duration) * width;
        const segmentWidth = Math.max(1, ((end - start) / duration) * width);
        ctx.fillStyle = block.final_label ? (colors[block.final_label] || '#64748b') : '#cbd5e1';
        ctx.globalAlpha = block.final_label ? 0.95 : 0.72;
        ctx.fillRect(x, 0, segmentWidth, height);
        if (segmentWidth >= 4) {
            ctx.fillStyle = '#ffffff';
            ctx.globalAlpha = 0.22;
            ctx.fillRect(Math.floor(x + segmentWidth) - 1, 0, 1, height);
        }
    }
    ctx.globalAlpha = 1;

    const video = document.getElementById('results-replay-video');
    const currentTime = video?.currentTime || 0;
    const scrubberX = (currentTime / duration) * width;
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(Math.max(0, scrubberX - 1.5), 0, 3, height);

    const reviewed = Number(comparison.summary?.reviewed) || 0;
    const total = Number(comparison.summary?.total_candidates) || comparison.labels?.length || 0;
    canvas.setAttribute('aria-label', `Etiquetas reales del generador de clips. ${reviewed} de ${total} bloques etiquetados.`);
}

function startReplayAnimationLoop() {
    stopReplayAnimationLoop();
    const tick = () => {
        drawReplayTimeline();
        updateReplayCurrentState();
        replayState.animationFrameId = window.requestAnimationFrame(tick);
    };
    replayState.animationFrameId = window.requestAnimationFrame(tick);
}

function stopReplayAnimationLoop() {
    if (replayState.animationFrameId) {
        window.cancelAnimationFrame(replayState.animationFrameId);
        replayState.animationFrameId = null;
    }
}

async function startActiveReview(reviewMode = 'overlap') {
    if (!currentResultId) {
        alert('Run or restore an analysis before starting review.');
        return;
    }

    const button = reviewMode === 'non_overlapping'
        ? document.getElementById('start-review-non-overlap')
        : reviewMode === 'full_match'
            ? document.getElementById('start-review-full-match')
            : document.getElementById('start-review');
    const originalText = button?.textContent || 'Start Review';
    if (button) {
        button.disabled = true;
        button.textContent = reviewMode === 'non_overlapping' || reviewMode === 'full_match'
            ? 'Preparando bloques...'
            : 'Preparing review...';
    }

    try {
        const response = await fetch('/api/review/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                result_id: currentResultId,
                review_mode: reviewMode,
            })
        });
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || 'Failed to start review');
        }

        window.location.href = `/review?session_id=${data.session.session_id}`;
    } catch (error) {
        alert(`Could not start review: ${error.message}`);
    } finally {
        if (button) {
            button.disabled = false;
            button.textContent = originalText;
        }
    }
}

// Preview Preprocessing
async function previewPreprocessing() {
    if (!selectedVideo) {
        alert('Seleccioná un video primero.');
        return;
    }

    const btn = document.getElementById('preview-preprocessing');
    const originalText = btn?.textContent || 'Previsualizar procesamiento';
    if (btn) {
        btn.textContent = 'Cargando video original...';
        btn.disabled = true;
    }

    const stepsDiv = document.getElementById('preprocessing-steps');
    const videoPlayer = document.getElementById('preview-video');
    const modal = document.getElementById('preprocessing-modal');
    if (!stepsDiv || !videoPlayer || !modal) {
        if (btn) {
            btn.textContent = originalText;
            btn.disabled = false;
        }
        return;
    }

    stepsDiv.innerHTML = `
        <div class="info-box preview-summary">
            <div>
                <h4>Video original</h4>
                <p id="preview-original-resolution">Cargando resolución...</p>
            </div>
            <div>
                <h4>Entrada del modelo</h4>
                <p>La región marcada se normaliza a 224×224</p>
            </div>
        </div>
        <div class="info-box">
            <h4>Procesamiento espacial</h4>
            <ol>
                <li>Se normaliza la altura a la referencia de entrenamiento de 360 px.</li>
                <li>Se conserva la relación de aspecto.</li>
                <li>Se toma el recorte central marcado y se convierte a 224×224.</li>
            </ol>
        </div>
    `;

    const restoreButton = () => {
        if (btn) {
            btn.textContent = originalText;
            btn.disabled = false;
        }
    };
    videoPlayer.onloadedmetadata = () => {
        const resolution = document.getElementById('preview-original-resolution');
        if (resolution) {
            resolution.textContent = `Resolución: ${videoPlayer.videoWidth}×${videoPlayer.videoHeight}`;
        }
        preprocessingPreviewRoiController?.positionOverlay();
        restoreButton();
    };
    videoPlayer.onerror = () => {
        const resolution = document.getElementById('preview-original-resolution');
        if (resolution) resolution.textContent = 'No se pudo cargar el video original.';
        restoreButton();
    };

    videoPlayer.src = `/api/video-by-path?video_path=${encodeURIComponent(selectedVideo)}`;
    modal.style.display = 'flex';
    videoPlayer.load();
}

// Play Detection Clip
async function playDetectionClip(detectionIndex) {
    if (!currentResultId) {
        alert('No analysis results available');
        return;
    }
    
    try {
        // Fetch segment information
        const response = await fetch(`/api/play-segment/${currentResultId}/${detectionIndex}`);
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Failed to get segment information');
        }
        
        // Show modal with video player
        const modal = document.getElementById('clip-modal');
        const videoPlayer = document.getElementById('clip-video');
        const modalTitle = modal.querySelector('.modal-title');
        const clipInfo = document.getElementById('clip-info');
        
        // Update modal title and info
        modalTitle.textContent = `Play #${detectionIndex + 1} - ${data.play_type.toUpperCase()}`;
        clipInfo.textContent = `${formatTime(data.start_time)} - ${formatTime(data.end_time)} (Confidence: ${(data.confidence * 100).toFixed(1)}%)`;
        
        // Use the safe server-side URL (avoids Windows path encoding issues)
        const videoUrl = data.video_url;

        // Set video source
        videoPlayer.src = videoUrl;
        
        // Wait for video to load metadata
        videoPlayer.addEventListener('loadedmetadata', function onLoaded() {
            videoPlayer.removeEventListener('loadedmetadata', onLoaded);
            
            // Set start time and play
            videoPlayer.currentTime = data.start_time;
            videoPlayer.play();
            
            // Add event listener to stop at end time
            const checkTime = () => {
                if (videoPlayer.currentTime >= data.end_time) {
                    videoPlayer.pause();
                    videoPlayer.currentTime = data.start_time; // Reset to start
                    videoPlayer.removeEventListener('timeupdate', checkTime);
                }
            };
            videoPlayer.addEventListener('timeupdate', checkTime);
            
            // Store cleanup function
            videoPlayer._clipCleanup = () => {
                videoPlayer.removeEventListener('timeupdate', checkTime);
            };
        });
        
        // Show modal
        modal.style.display = 'flex';
        
    } catch (error) {
        alert(`Error: ${error.message}`);
        console.error('Play clip error:', error);
    }
}

function closePreprocessingModal() {
    const modal = document.getElementById('preprocessing-modal');
    const videoPlayer = document.getElementById('preview-video');
    
    // Stop and clear video
    videoPlayer.pause();
    videoPlayer.src = '';
    
    modal.style.display = 'none';
}

// ─────────────────────────── History ───────────────────────────

function escapeHistoryText(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function openHistoryModal() {
    const modal = document.getElementById('history-modal');
    modal.style.display = 'flex';
    loadHistory();
}

function closeHistoryModal() {
    document.getElementById('history-modal').style.display = 'none';
}

async function loadHistory() {
    const container = document.getElementById('history-list');
    container.innerHTML = '<div class="loading">Cargando historial…</div>';

    try {
        const response = await fetch('/api/results');
        const data = await response.json();

        if (!data.success) throw new Error(data.error);

        historyResults = data.results || [];
        historyPage = 1;
        renderHistory();
        return;

        if (data.results.length === 0) {
            container.innerHTML = '<p style="color:#666;text-align:center;padding:2rem;">No past analyses found.</p>';
            return;
        }

        container.innerHTML = `
            <table style="width:100%;border-collapse:collapse;font-size:14px;">
                <thead>
                    <tr style="border-bottom:2px solid #e2e8f0;text-align:left;">
                        <th style="padding:10px 8px;">Date</th>
                        <th style="padding:10px 8px;">Video</th>
                        <th style="padding:10px 8px;">Model</th>
                        <th style="padding:10px 8px;text-align:center;">Detections</th>
                        <th style="padding:10px 8px;text-align:center;">Line</th>
                        <th style="padding:10px 8px;text-align:center;">Scrum</th>
                        <th style="padding:10px 8px;text-align:center;">Other</th>
                        <th style="padding:10px 8px;"></th>
                    </tr>
                </thead>
                <tbody>
                    ${data.results.map(r => `
                        <tr style="border-bottom:1px solid #e2e8f0;" class="history-row">
                            <td style="padding:10px 8px;white-space:nowrap;">${r.date}</td>
                            <td style="padding:10px 8px;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${r.video}">${r.video}</td>
                            <td style="padding:10px 8px;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${r.model}">${r.model}</td>
                            <td style="padding:10px 8px;text-align:center;font-weight:600;">${r.total_detections}</td>
                            <td style="padding:10px 8px;text-align:center;"><span class="badge line">${r.counts.line || 0}</span></td>
                            <td style="padding:10px 8px;text-align:center;"><span class="badge scrum">${r.counts.scrum || 0}</span></td>
                            <td style="padding:10px 8px;text-align:center;"><span class="badge other">${r.counts.other || 0}</span></td>
                            <td style="padding:10px 8px;">
                                <button class="btn btn-primary" style="padding:4px 12px;font-size:13px;"
                                    onclick="restoreResult('${r.result_id}')">
                                    Load
                                </button>
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    } catch (err) {
        container.innerHTML = `<p style="color:red;">No se pudo cargar el historial: ${err.message}</p>`;
    }
}

function renderHistory() {
    const container = document.getElementById('history-list');
    if (!container) return;

    const searchHadFocus = document.activeElement?.id === 'history-search';
    const caretPosition = document.getElementById('history-search')?.selectionStart ?? null;
    const currentQuery = document.getElementById('history-search')?.value || '';
    const hideEmpty = document.getElementById('history-hide-empty')?.checked ?? true;
    const query = currentQuery.trim().toLowerCase();
    const filtered = historyResults.filter(result => {
        const haystack = `${result.video || ''} ${result.model || ''}`.toLowerCase();
        return (!query || haystack.includes(query)) && (!hideEmpty || Number(result.total_detections || 0) > 0);
    });

    const totalPages = Math.max(1, Math.ceil(filtered.length / HISTORY_PAGE_SIZE));
    historyPage = Math.min(historyPage, totalPages);
    const start = (historyPage - 1) * HISTORY_PAGE_SIZE;
    const rows = filtered.slice(start, start + HISTORY_PAGE_SIZE);

    container.innerHTML = `
        <div class="history-toolbar">
            <input id="history-search" type="search" value="${escapeHistoryText(currentQuery)}" placeholder="Buscar por video o modelo..." aria-label="Buscar por video o modelo en el historial">
            <label><input id="history-hide-empty" type="checkbox" ${hideEmpty ? 'checked' : ''}> Ocultar análisis sin detecciones</label>
            <span>${filtered.length} resultados</span>
        </div>
        ${rows.length ? `
            <div class="history-cards" role="list">${rows.map(result => {
                const video = escapeHistoryText(result.video || 'Video sin nombre');
                const model = escapeHistoryText(result.model || 'Modelo no informado');
                const resultId = escapeHistoryText(result.result_id);
                const total = Number(result.total_detections || 0);
                const clipGenerator = result.clip_generator?.available ? result.clip_generator : null;
                return `
                    <article class="history-card" role="listitem">
                        <div class="history-card-identifiers">
                            <div class="history-card-field history-card-video">
                                <span class="history-card-label">Video</span>
                                <strong title="${video}">${video}</strong>
                            </div>
                            <div class="history-card-field history-card-model">
                                <span class="history-card-label">Modelo</span>
                                <strong title="${model}">${model}</strong>
                            </div>
                        </div>
                        <div class="history-card-summary">
                            <div class="history-card-total"><strong>${total}</strong><span>detecciones</span></div>
                            <div class="history-card-breakdown" aria-label="Detecciones por tipo">
                                <span class="history-count line"><b>${result.counts.line || 0}</b> line</span>
                                <span class="history-count scrum"><b>${result.counts.scrum || 0}</b> scrum</span>
                            </div>
                            <button class="btn btn-primary history-load" onclick="restoreResult('${resultId}')">Cargar análisis</button>
                        </div>
                        ${clipGenerator ? `
                            <div class="history-card-real" title="Etiquetas finales de la sesión de generación de clips">
                                <span class="history-card-label">Real · clips revisados</span>
                                <span><b>${clipGenerator.line}</b> line · <b>${clipGenerator.scrum}</b> scrum</span>
                            </div>` : ''}
                    </article>`;
            }).join('')}</div>
            <div class="history-pagination">
                <button id="history-prev" class="btn btn-secondary" ${historyPage === 1 ? 'disabled' : ''}>Anterior</button>
                <span>Página ${historyPage} de ${totalPages}</span>
                <button id="history-next" class="btn btn-secondary" ${historyPage === totalPages ? 'disabled' : ''}>Siguiente</button>
            </div>`
            : '<div class="dataset-empty">No hay análisis que coincidan con los filtros.</div>'}
    `;

    document.getElementById('history-search')?.addEventListener('input', () => { historyPage = 1; renderHistory(); });
    document.getElementById('history-hide-empty')?.addEventListener('change', () => { historyPage = 1; renderHistory(); });
    document.getElementById('history-prev')?.addEventListener('click', () => { historyPage -= 1; renderHistory(); });
    document.getElementById('history-next')?.addEventListener('click', () => { historyPage += 1; renderHistory(); });
    if (searchHadFocus) {
        const nextSearch = document.getElementById('history-search');
        nextSearch?.focus();
        if (caretPosition !== null) nextSearch?.setSelectionRange(caretPosition, caretPosition);
    }
}

async function restoreResult(resultId) {
    try {
        const response = await fetch(`/api/result/${resultId}`);
        const data = await response.json();
        if (!data.success) throw new Error(data.error);

        currentResultId = resultId;
        data.results.video_url = data.video_url || `/api/video-for-result/${resultId}`;
        closeHistoryModal();
        displayResults(data.results, { restored: true });

        // Scroll to results
        document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
    } catch (err) {
        alert('No se pudo cargar el resultado: ' + err.message);
    }
}

// ─────────────────────────── Process Video ───────────────────────────

/**
 * Send a raw/uploaded video to the server to be split without spatial preprocessing.
 * Shows a floating status toast that polls every 3 s until the job completes.
 */
async function processVideo(videoPath) {
    if (!confirm(`Dividir el video en bloques conservando su resoluci\u00f3n original?\n${videoPath}\n\nPuede tardar varios minutos.`)) return;

    // --- toast helper ---
    function makeToast(msg) {
        const el = document.createElement('div');
        el.className = 'process-toast';
        el.innerHTML = `<span class="process-toast-msg">${msg}</span>
            <button onclick="this.parentElement.remove()" style="margin-left:12px;background:none;border:none;cursor:pointer;font-size:16px;">×</button>`;
        document.body.appendChild(el);
        return el;
    }

        const toast = makeToast('Starting processing...');    try {
        const resp = await fetch('/api/process_video', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_path: videoPath })
        });

        const data = await resp.json();
        if (!data.success) throw new Error(data.error || 'Failed to start processing');

        const jobId = data.job_id;
        toast.querySelector('.process-toast-msg').textContent = `Processing job ${jobId} started...`;

        const poll = setInterval(async () => {
            try {
                const sresp = await fetch(`/api/process_status/${jobId}`);
                const sdata = await sresp.json();
                if (!sdata.success) { clearInterval(poll); return; }

                const job = sdata.job;
                if (job.status === 'running') {
                    toast.querySelector('.process-toast-msg').textContent = `Job ${jobId}: processing...`;
                } else if (job.status === 'done') {
                    clearInterval(poll);
                    toast.querySelector('.process-toast-msg').textContent =
                        `Job ${jobId} complete — ${job.out_files.length} chunk(s) ready.`;
                    toast.style.background = 'var(--success-color, #10b981)';
                    // Refresh both video lists
                    loadVideos();
                    loadProcessedVideos();
                    setTimeout(() => toast.remove(), 8000);
                } else if (job.status === 'error') {
                    clearInterval(poll);
                    toast.querySelector('.process-toast-msg').textContent = `Job ${jobId} failed: ${job.error}`;
                    toast.style.background = 'var(--error-color, #ef4444)';
                    setTimeout(() => toast.remove(), 12000);
                }
            } catch (_) { /* ignore transient network errors */ }
        }, 3000);

    } catch (err) {
        toast.querySelector('.process-toast-msg').textContent = `Error: ${err.message}`;
        toast.style.background = 'var(--error-color, #ef4444)';
        setTimeout(() => toast.remove(), 8000);
    }
}
