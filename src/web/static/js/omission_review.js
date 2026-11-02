/* Missed human-reference segments for a frozen saved comparison. */
window.RugbyOmissionReview = (() => {
    let selectedModel = 'all', selectedClass = 'all', activeRange = null, scrolled = false;
    const boundVideos = new WeakSet();
    const escape = value => String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    const time = seconds => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;

    function bindVideo(video) {
        if (!video || boundVideos.has(video)) return;
        boundVideos.add(video);
        video.addEventListener('timeupdate', () => {
            if (activeRange?.video === video && video.currentTime >= activeRange.end) {
                video.pause();
                activeRange = null;
            }
        });
        // Normal player and timeline interaction exits the bounded omission preview.
        video.addEventListener('pointerdown', () => { activeRange = null; });
        document.addEventListener('click', event => {
            if (event.target.closest('.compare-timeline-segment')) activeRange = null;
        }, true);
    }

    function render({results, host, video}) {
        if (!host) return;
        const review = Object.values(results || {}).map(r => r.omission_review).find(Boolean);
        let panel = document.getElementById('compare-omissions');
        if (!review) {
            panel?.remove();
            activeRange = null;
            return;
        }
        if (!panel) {
            panel = document.createElement('section');
            panel.id = 'compare-omissions';
            panel.className = 'compare-omissions';
            panel.setAttribute('aria-labelledby', 'omission-title');
            host.before(panel);
        }
        const valid = review.models.every(model => {
            const result = results[model.title];
            const signature = (result?.detections || []).map(p => [p.play_type,p.start_time,p.end_time]);
            return result?.manual_comparison?.reference_sha256 === review.reference_sha256 &&
                JSON.stringify(signature) === JSON.stringify(model.prediction_signature);
        });
        const heading = '<span id="omisiones"></span><h2 id="omission-title">Jugadas omitidas</h2>';
        if (!valid) {
            activeRange = null;
            panel.innerHTML = heading+'<p id="omission-status" role="status">Las detecciones cambiaron. Restablecé los umbrales originales para ver las omisiones de esta evaluación.</p>';
            return;
        }
        if (!review.models.some(m => m.key === selectedModel)) selectedModel = 'all';
        const items = review.segments.filter(s => (selectedModel === 'all' || !s.detected[selectedModel]) && (selectedClass === 'all' || s.play_type === selectedClass));
        const modelOptions = '<option value="all">Cualquier versión · '+review.segments.length+' segmentos</option>'+review.models.map(m => `<option value="${escape(m.key)}">${escape(m.title)} · ${m.missed} omitidos</option>`).join('');
        panel.innerHTML = heading+`<p>Comparación con tus etiquetas corregidas del partido Argentina–Nueva Zelanda, 16/08/2025. Elegí una versión para revisar qué se pierde.</p>
            <div class="omission-controls"><label>Omitidas por<select id="omission-model">${modelOptions}</select></label>
            <label>Tipo<select id="omission-class"><option value="all">Line y scrum</option><option value="line">Line</option><option value="scrum">Scrum</option></select></label></div>
            <p id="omission-status" role="status">${items.length} segmentos para revisar. «Ver» reproduce el segmento con ${review.context_s} segundos de contexto a cada lado.</p>
            <div class="omission-table-scroll" tabindex="0" aria-label="Estado por versión; desplazable horizontalmente"><table><thead><tr><th>Segmento · tiempo del video</th>${review.models.map(m => `<th>${escape(m.title)}</th>`).join('')}<th>Revisar</th></tr></thead><tbody>
            ${items.map(s => `<tr data-omission-id="${escape(s.id)}"><th scope="row"><span class="omission-play">${s.play_type === 'line' ? 'Line' : 'Scrum'}</span><span class="omission-time">${time(s.start_s)}–${time(s.end_s)}</span><small>Clips ${s.blocks.map(b => b+1).join(', ')}</small></th>${review.models.map(m => `<td><span class="omission-badge ${s.detected[m.key] ? 'omission-found' : 'omission-missed'}">${s.detected[m.key] ? 'Detectada' : 'Omitida'}</span></td>`).join('')}<td><button type="button" class="omission-play-button" data-start="${s.start_s}" data-end="${s.end_s}" aria-label="Ver ${escape(s.play_type)} de ${time(s.start_s)} a ${time(s.end_s)}">▶ Ver</button></td></tr>`).join('')}
            </tbody></table></div>
            <p class="omission-note">Cada fila es un segmento de tus clips contiguos de la misma clase; puede incluir repeticiones o rearmados. Se considera detectado con una coincidencia de la misma clase, al menos 1 segundo de solapamiento y una pareja por segmento, igual que en las estadísticas.</p>`;
        const modelFilter = panel.querySelector('#omission-model'), classFilter = panel.querySelector('#omission-class');
        modelFilter.value = selectedModel;
        classFilter.value = selectedClass;
        modelFilter.addEventListener('change', event => { selectedModel = event.target.value; render({results,host,video}); panel.querySelector('#omission-model').focus(); });
        classFilter.addEventListener('change', event => { selectedClass = event.target.value; render({results,host,video}); panel.querySelector('#omission-class').focus(); });
        bindVideo(video);
        panel.querySelectorAll('.omission-play-button').forEach(button => button.addEventListener('click', () => {
            if (!video) return;
            const start = Number(button.dataset.start), end = Number(button.dataset.end);
            activeRange = {video,end:Math.min(Number.isFinite(video.duration) ? video.duration : Infinity,end+review.context_s)};
            video.currentTime = Math.max(0,start-review.context_s);
            const play = video.play();
            if (play?.catch) play.catch(() => { panel.querySelector('#omission-status').textContent = 'Segmento seleccionado. Presioná reproducir en el video para verlo.'; });
            panel.querySelector('#omission-status').textContent = `Revisando ${time(start)}–${time(end)} con contexto. El video se pausa al finalizar.`;
            video.scrollIntoView({behavior:'smooth',block:'center'});
        }));
        if (!scrolled && location.hash === '#omisiones') {
            scrolled = true;
            window.setTimeout(() => panel.scrollIntoView({block:'start'}),200);
        }
    }
    return {render};
})();
