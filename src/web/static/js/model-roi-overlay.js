(function () {
    'use strict';

    const MODEL_INPUT_SIZE = 224;
    const YOLO_INPUT_SIZE = 640;
    const TRAINING_REFERENCE_HEIGHT = 360;

    function normalizationGeometry(sourceWidth, sourceHeight) {
        const prepared = sourceWidth === MODEL_INPUT_SIZE && sourceHeight === MODEL_INPUT_SIZE;
        const scale = prepared ? 1 : Math.max(
            TRAINING_REFERENCE_HEIGHT / sourceHeight,
            MODEL_INPUT_SIZE / sourceWidth,
        );
        const resizedWidth = Math.max(MODEL_INPUT_SIZE, Math.round(sourceWidth * scale));
        const resizedHeight = Math.max(MODEL_INPUT_SIZE, Math.round(sourceHeight * scale));
        const cropX = Math.floor((resizedWidth - MODEL_INPUT_SIZE) / 2);
        const cropY = Math.floor((resizedHeight - MODEL_INPUT_SIZE) / 2);
        return {
            sourceX: cropX * sourceWidth / resizedWidth,
            sourceY: cropY * sourceHeight / resizedHeight,
            sourceWidth: MODEL_INPUT_SIZE * sourceWidth / resizedWidth,
            sourceHeight: MODEL_INPUT_SIZE * sourceHeight / resizedHeight,
        };
    }

    function formatSourceDimension(value) {
        const rounded = Math.round(value);
        return Math.abs(value - rounded) < 0.05 ? String(rounded) : value.toFixed(1);
    }

    function attach({ video, stage, overlay, toggle, inputMode = 'rgb_224' }) {
        if (!video || !stage || !overlay || !toggle) return null;

        let available = true;
        let currentInputMode = inputMode;

        function formatRoiLabel(geometry) {
            const source = `${formatSourceDimension(geometry.sourceWidth)}×${formatSourceDimension(geometry.sourceHeight)}`;
            if (currentInputMode === 'source_equivalent_roi') {
                return `Área clasificada: ${source} → RGB ${MODEL_INPUT_SIZE}×${MODEL_INPUT_SIZE} · YOLO ${YOLO_INPUT_SIZE}×${YOLO_INPUT_SIZE}`;
            }
            return `Área clasificada: ${source} → ${MODEL_INPUT_SIZE}×${MODEL_INPUT_SIZE}`;
        }

        function positionOverlay() {
            const videoWidth = video.videoWidth;
            const videoHeight = video.videoHeight;
            const elementWidth = video.clientWidth;
            const elementHeight = video.clientHeight;
            const visible = available && toggle.checked
                && videoWidth > 0 && videoHeight > 0
                && elementWidth > 0 && elementHeight > 0;

            overlay.classList.toggle('is-visible', visible);
            overlay.setAttribute('aria-hidden', visible ? 'false' : 'true');
            if (!visible) return;

            const mediaRatio = videoWidth / videoHeight;
            const elementRatio = elementWidth / elementHeight;
            let contentWidth;
            let contentHeight;
            let contentLeft = video.offsetLeft;
            let contentTop = video.offsetTop;

            if (elementRatio > mediaRatio) {
                contentHeight = elementHeight;
                contentWidth = contentHeight * mediaRatio;
                contentLeft += (elementWidth - contentWidth) / 2;
            } else {
                contentWidth = elementWidth;
                contentHeight = contentWidth / mediaRatio;
                contentTop += (elementHeight - contentHeight) / 2;
            }

            const geometry = normalizationGeometry(videoWidth, videoHeight);
            const roiLabel = formatRoiLabel(geometry);

            overlay.style.left = `${contentLeft + (geometry.sourceX / videoWidth) * contentWidth}px`;
            overlay.style.top = `${contentTop + (geometry.sourceY / videoHeight) * contentHeight}px`;
            overlay.style.width = `${(geometry.sourceWidth / videoWidth) * contentWidth}px`;
            overlay.style.height = `${(geometry.sourceHeight / videoHeight) * contentHeight}px`;
            overlay.querySelector('span')?.replaceChildren(roiLabel);
            overlay.setAttribute('aria-label', roiLabel);
            overlay.dataset.sourceRoi = `${geometry.sourceX},${geometry.sourceY},${geometry.sourceWidth},${geometry.sourceHeight}`;
        }

        function setAvailable(nextAvailable) {
            available = Boolean(nextAvailable);
            toggle.disabled = !available;
            toggle.closest('label')?.classList.toggle('is-disabled', !available);
            positionOverlay();
        }

        function setInputMode(nextInputMode) {
            currentInputMode = nextInputMode === 'source_equivalent_roi'
                ? 'source_equivalent_roi'
                : 'rgb_224';
            positionOverlay();
        }

        toggle.addEventListener('change', positionOverlay);
        video.addEventListener('loadedmetadata', positionOverlay);
        video.addEventListener('emptied', positionOverlay);
        document.addEventListener('fullscreenchange', () => requestAnimationFrame(positionOverlay));
        window.addEventListener('resize', positionOverlay);

        if (typeof ResizeObserver === 'function') {
            const resizeObserver = new ResizeObserver(positionOverlay);
            resizeObserver.observe(stage);
            resizeObserver.observe(video);
        }

        positionOverlay();
        return { positionOverlay, setAvailable, setInputMode };
    }

    window.ModelRoiOverlay = {
        attach,
        normalizationGeometry,
        formatSourceDimension,
        modelInputSize: MODEL_INPUT_SIZE,
        yoloInputSize: YOLO_INPUT_SIZE,
        referenceHeight: TRAINING_REFERENCE_HEIGHT,
    };
})();
