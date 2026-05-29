'use strict';

const slider    = document.getElementById('epsilon-slider');
const epsVal    = document.getElementById('epsilon-val');
const runBtn    = document.getElementById('run-btn');
const results   = document.getElementById('results');
const sampleSel = document.getElementById('sample-select');

// Live ε display
slider.addEventListener('input', () => { epsVal.textContent = parseFloat(slider.value).toFixed(2); });

// Run attack
runBtn.addEventListener('click', async () => {
  runBtn.disabled = true;
  runBtn.textContent = 'Running…';
  runBtn.classList.add('loading');

  const payload = {
    sample_id: parseInt(sampleSel.value),
    epsilon:   parseFloat(slider.value),
  };

  try {
    const res  = await fetch('/api/attack', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.error) { alert('Error: ' + data.error); return; }
    renderResults(data);
  } catch (e) {
    alert('Request failed: ' + e.message);
  } finally {
    runBtn.disabled  = false;
    runBtn.textContent = 'Run Attack';
    runBtn.classList.remove('loading');
  }
});

function renderResults(d) {
  results.classList.remove('hidden');

  // Verdicts
  setBadge('v-original',      d.original_pred);
  setBadge('v-constrained',   d.con_pred);
  setBadge('v-unconstrained', d.uncon_pred);

  // Gap callout
  const callout = document.getElementById('gap-callout');
  if (d.uncon_evaded && !d.con_evaded) {
    callout.classList.remove('hidden');
    callout.innerHTML =
      `<strong>Constraint inflation demonstrated.</strong> ` +
      `Unconstrained PGD evaded the detector. Constrained PGD did not. ` +
      `The difference: the unconstrained attack exploited ` +
      `<strong>${d.n_violations} physically impossible feature value${d.n_violations !== 1 ? 's' : ''}</strong>. ` +
      `At ε = ${d.epsilon.toFixed(2)}, this gap reaches up to <strong>+72 pp</strong> ` +
      `across the full test set.`;
  } else if (d.uncon_evaded && d.con_evaded) {
    callout.classList.remove('hidden');
    callout.innerHTML =
      `Both attacks evaded the detector at ε = ${d.epsilon.toFixed(2)}. ` +
      `Try a smaller ε to see the gap emerge — the constraint inflation effect is ` +
      `largest between ε = 0.05 and ε = 0.20.`;
  } else if (!d.uncon_evaded && !d.con_evaded) {
    callout.classList.remove('hidden');
    callout.innerHTML =
      `Neither attack evaded at ε = ${d.epsilon.toFixed(2)}. ` +
      `Try ε = 0.20 or higher on this sample, or select a different flow.`;
  } else {
    callout.classList.add('hidden');
  }

  // Feature table
  const tbody = document.getElementById('feature-tbody');
  tbody.innerHTML = '';
  for (const row of d.feature_rows) {
    const tr = document.createElement('tr');
    const range = formatRange(row.valid_lo, row.valid_hi);

    tr.innerHTML = `
      <td>${row.feature}</td>
      <td class="cell-ok">${fmt(row.original)}</td>
      <td class="${row.con_violated ? 'cell-viol' : 'cell-ok'}">
        ${fmt(row.constrained)}${row.con_violated ? '<span class="violation-tag">⚠ impossible</span>' : ''}
      </td>
      <td class="${row.uncon_violated ? 'cell-viol' : 'cell-ok'}">
        ${fmt(row.unconstrained)}${row.uncon_violated ? '<span class="violation-tag">⚠ impossible</span>' : ''}
      </td>
      <td class="cell-ok muted">${range}</td>
    `;
    tbody.appendChild(tr);
  }

  // Violations card
  const vCard = document.getElementById('violations-card');
  const vList = document.getElementById('violations-list');
  if (d.violations.length > 0) {
    vCard.classList.remove('hidden');
    vList.innerHTML = '';
    for (const v of d.violations) {
      const li = document.createElement('li');
      const rangeStr = formatRange(v.valid_lo, v.valid_hi);
      li.innerHTML =
        `<span class="viol-name">${v.feature}</span>: ` +
        `${fmt(v.value)} ` +
        `<span class="viol-range">(valid: ${rangeStr})</span>`;
      vList.appendChild(li);
    }
  } else {
    vCard.classList.add('hidden');
  }

  // Explainer
  document.getElementById('explainer-text').innerHTML = buildExplainer(d);

  // Scroll into view
  results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function setBadge(id, pred) {
  const el = document.getElementById(id);
  el.textContent = pred;
  el.className = 'v-badge ' + (pred === 'ATTACK' ? 'attack' : 'benign');
}

function fmt(n) {
  if (n === null || n === undefined) return '—';
  const abs = Math.abs(n);
  if (abs >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (abs >= 1)    return n.toFixed(2);
  return n.toFixed(4);
}

function formatRange(lo, hi) {
  const loStr = lo !== null && lo !== undefined ? fmt(lo) : '−∞';
  const hiStr = hi !== null && hi !== undefined ? fmt(hi) : '+∞';
  return `[${loStr}, ${hiStr}]`;
}

function buildExplainer(d) {
  if (d.uncon_evaded && !d.con_evaded) {
    return `
      The unconstrained adversarial example evaded the NIDS classifier — but
      ${d.n_violations} of its features have values that <strong>cannot appear
      in real network traffic</strong>. A physical network sensor would never
      record these flows. The "evasion" is not a property of the attack; it is
      a measurement artefact from running the optimizer outside the physically
      valid feature space.
      <br><br>
      Constrained PGD — which projects features back onto valid bounds after
      every gradient step — did not evade at this budget. That is the true
      attack success rate. The difference between the two numbers is the
      <strong>constraint inflation gap</strong>.
    `;
  }
  if (d.uncon_evaded && d.con_evaded) {
    return `
      Both attacks evaded the detector at ε = ${d.epsilon.toFixed(2)}. This
      means the constrained adversary can evade the classifier without resorting
      to impossible feature values — a genuine finding. The constraint inflation
      gap is visible at lower ε for this sample.
    `;
  }
  return `
    At ε = ${d.epsilon.toFixed(2)}, neither attack evaded this particular flow.
    This is expected: at small budgets, the constrained adversary has limited room
    to maneuver within valid bounds. The gap between constrained and unconstrained
    attacks is largest around ε = 0.10–0.20 on UNSW-NB15.
  `;
}
