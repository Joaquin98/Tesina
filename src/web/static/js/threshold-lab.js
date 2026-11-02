/* Exact lookup over the exhaustive, saved VF threshold intervals. */
'use strict';
const ThresholdLab = {
  stateAt(states, threshold) {
    // Rule p >= t: upper endpoints are inclusive, lower endpoints exclusive.
    let lo = 0, hi = states.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (threshold <= states[mid].upper_inclusive) hi = mid;
      else lo = mid + 1;
    }
    return states[lo];
  },
  metrics(tp, fp, fn) {
    return {tp, fp, fn, candidates: tp + fp, precision: tp + fp ? tp / (tp + fp) : 0,
      recall: tp / (tp + fn), f1: 2 * tp / (2 * tp + fp + fn)};
  },
};
if (typeof module !== 'undefined') module.exports = ThresholdLab;
if (typeof document !== 'undefined') document.addEventListener('DOMContentLoaded', async () => {
  const status = document.getElementById('load-status');
  try {
    const response = await fetch('/static/data/m1-vf-threshold-sweep.json');
    if (!response.ok) throw new Error('No se pudieron cargar los resultados guardados.');
    const data = await response.json();
    if (data.states.length !== 657 || data.states.at(-1).upper_inclusive !== 1) throw new Error('El barrido está incompleto.');
    const slider = document.getElementById('threshold');
    const number = document.getElementById('threshold-number');
    const dial = document.getElementById('threshold-dial');
    const baseline = data.summary.baseline;
    const scrum = {...ThresholdLab.metrics(10, 2, 0), temporal_f1: data.summary.scrum.weighted_f1};
    const percent = n => (100 * n).toLocaleString('es-AR', {minimumFractionDigits: 1, maximumFractionDigits: 1}) + '%';
    const decimal = n => n.toLocaleString('es-AR', {maximumFractionDigits: 4});
    const row = (label, m, cls) => `<tr class="${cls}"><th scope="row">${label}</th><td>${m.tp}</td><td>${m.fp}</td><td>${m.fn}</td><td>${percent(m.precision)}</td><td>${percent(m.recall)}</td><td>${percent(m.f1)}</td><td>${m.temporal_f1 == null ? '—' : percent(m.temporal_f1)}</td></tr>`;
    function update(raw, syncNumber = true) {
      if (!Number.isFinite(raw)) return;
      const t = Math.min(1, Math.max(0, raw));
      const m = ThresholdLab.stateAt(data.states, t);
      slider.value = t;
      slider.setAttribute('aria-valuetext', `${decimal(t)}: ${m.tp} aciertos, ${m.fp} falsas alarmas y ${m.fn} omitidas`);
      if (syncNumber) number.value = t;
      document.getElementById('dial-value').textContent = decimal(t);
      dial.style.setProperty('--turn', `${t * 270}deg`);
      dial.style.setProperty('--angle', `${t * 270 - 135}deg`);
      document.getElementById('selected-rule').textContent = `Lineout con p(line) ≥ ${decimal(t)}. Scrum fijo.`;
      const total = ThresholdLab.metrics(m.tp + scrum.tp, m.fp + scrum.fp, m.fn + scrum.fn);
      document.getElementById('metrics-body').innerHTML = row('Line · umbral actual', m, 'current') + row('Line · argmax', baseline, 'reference') + row('Scrum · fijo', scrum, '') + row('Total · umbral actual', total, 'total');
      const tpDelta = m.tp - baseline.tp, fpDelta = m.fp - baseline.fp;
      document.getElementById('comparison-note').textContent = `Frente a argmax: ${tpDelta > 0 ? tpDelta + ' aciertos más' : tpDelta < 0 ? -tpDelta + ' aciertos menos' : 'los mismos aciertos'} y ${fpDelta > 0 ? fpDelta + ' falsas alarmas más' : fpDelta < 0 ? -fpDelta + ' falsas alarmas menos' : 'las mismas falsas alarmas'}.`;
      document.getElementById('coverage-label').textContent = `${m.tp} de 30 lineouts detectados`;
      document.getElementById('candidate-count').textContent = `${m.candidates} detecciones de lineout para revisar`;
      document.getElementById('coverage-fill').style.width = `${100 * m.recall}%`;
      document.getElementById('state-note').textContent = `Este valor activa ${m.active_blocks} de los 688 bloques como lineout. La tabla usa las decisiones exactas del barrido, sin interpolar métricas.`;
      document.querySelectorAll('[data-threshold]').forEach(button => button.setAttribute('aria-pressed', String(Math.abs(Number(button.dataset.threshold) - t) < 1e-9)));
      history.replaceState(null, '', `${location.pathname}?umbral=${t}`);
    }
    slider.addEventListener('input', () => update(Number(slider.value)));
    number.addEventListener('input', () => {if (number.value !== '' && number.validity.valid) update(number.valueAsNumber, false);});
    number.addEventListener('change', () => {if (number.value !== '' && Number.isFinite(number.valueAsNumber)) update(number.valueAsNumber); else update(Number(slider.value));});
    document.querySelectorAll('[data-threshold]').forEach(button => button.addEventListener('click', () => update(Number(button.dataset.threshold))));
    let dragging = false;
    const point = event => {
      const r = dial.getBoundingClientRect();
      const angle = (Math.atan2(event.clientY - r.top - r.height / 2, event.clientX - r.left - r.width / 2) * 180 / Math.PI + 90 + 360) % 360;
      const relative = (angle - 225 + 360) % 360;
      const t = relative > 270 ? (relative > 315 ? 0 : 1) : relative / 270;
      update(Math.round(t * 10000) / 10000);
    };
    dial.addEventListener('pointerdown', event => {if (event.button !== 0) return; dragging = true; dial.setPointerCapture(event.pointerId); point(event);});
    dial.addEventListener('pointermove', event => {if (dragging) point(event);});
    dial.addEventListener('pointerup', () => {dragging = false;});
    dial.addEventListener('pointercancel', () => {dragging = false;});
    const requested = new URLSearchParams(location.search).get('umbral');
    update(requested !== null && requested.trim() !== '' && Number.isFinite(Number(requested)) ? Number(requested) : .35);
    document.getElementById('lab-content').hidden = false;
    status.textContent = '';
  } catch (error) {
    status.textContent = `No se pudo abrir el explorador: ${error.message} Recargá la página para reintentar.`;
  }
});
