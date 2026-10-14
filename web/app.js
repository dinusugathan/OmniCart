document.addEventListener('DOMContentLoaded', () => {
  // Navigation Tabs
  const navTabs = document.querySelectorAll('.nav-tab');
  const viewPanels = document.querySelectorAll('.view-panel');
  
  navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      navTabs.forEach(t => t.classList.remove('active'));
      viewPanels.forEach(p => p.classList.remove('active-panel'));
      
      tab.classList.add('active');
      const targetPanel = document.getElementById(tab.dataset.tab);
      if (targetPanel) targetPanel.classList.add('active-panel');
    });
  });

  // State
  let catalogData = [];

  // Chat Elements
  const chatMessages = document.getElementById('chat-messages');
  const chatForm = document.getElementById('chat-form');
  const queryInput = document.getElementById('query-input');
  const quickChips = document.querySelectorAll('.quick-chip');

  // Modal Elements
  const citationModal = document.getElementById('citation-modal');
  const modalCitationTag = document.getElementById('modal-citation-tag');
  const modalCitationText = document.getElementById('modal-citation-text');
  const modalCloseBtn = document.getElementById('modal-close-btn');

  // Quick Chips Click
  quickChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const q = chip.dataset.query;
      queryInput.value = q;
      handleSubmitQuery(q);
    });
  });

  // Chat Submit
  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;
    handleSubmitQuery(query);
  });

  async function handleSubmitQuery(query) {
    // Append User Message
    appendUserMessage(query);
    queryInput.value = '';

    // Append Assistant Loading Skeleton
    const loadingId = 'loading-' + Date.now();
    appendLoadingSkeleton(loadingId);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const data = await response.json();
      removeLoadingSkeleton(loadingId);
      appendAssistantMessage(data);
    } catch (err) {
      removeLoadingSkeleton(loadingId);
      appendErrorMessage(err.message);
    }
  }

  function appendUserMessage(text) {
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble user-bubble';
    bubble.innerHTML = `<div class="bubble-body">${escapeHtml(text)}</div>`;
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function appendLoadingSkeleton(id) {
    const skeleton = document.createElement('div');
    skeleton.id = id;
    skeleton.className = 'chat-bubble assistant-bubble';
    skeleton.innerHTML = `
      <div class="bubble-header">
        <span class="bubble-avatar">✨</span>
        <span class="bubble-name">OmniCart Clinical Advisor</span>
        <span class="bubble-badge">Reranking candidates...</span>
      </div>
      <div class="bubble-body">
        <p style="color: var(--text-muted); font-style: italic;">Analyzing query intent, filtering catalog, and running cross-encoder reranker...</p>
      </div>
    `;
    chatMessages.appendChild(skeleton);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function removeLoadingSkeleton(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function appendErrorMessage(errText) {
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble assistant-bubble';
    bubble.innerHTML = `
      <div class="bubble-body" style="border-color: rgba(239, 68, 68, 0.4); background: rgba(239, 68, 68, 0.1);">
        <p style="color: #fca5a5;">⚠️ Error connecting to RAG pipeline: ${escapeHtml(errText)}</p>
      </div>
    `;
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function appendAssistantMessage(data) {
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble assistant-bubble';

    let guardrailHtml = '';
    if (data.guardrail && data.guardrail.triggered) {
      const alertClass = data.guardrail.alert_type === 'CONTRAINDICATION' ? 'contraindication' : 'blocked';
      const icon = data.guardrail.alert_type === 'CONTRAINDICATION' ? '⚠️' : '🛡️';
      guardrailHtml = `
        <div class="guardrail-banner ${alertClass}">
          <div class="guardrail-icon">${icon}</div>
          <div class="guardrail-text"><strong>Safety Protocol:</strong> ${escapeHtml(data.guardrail.message)}</div>
        </div>
      `;
    }

    let productsHtml = '';
    if (data.recommended_products && data.recommended_products.length > 0) {
      productsHtml = `
        <div class="products-cards-grid">
          ${data.recommended_products.map(prod => `
            <div class="product-card">
              <div>
                <div class="card-top">
                  <span class="card-category">${escapeHtml(prod.category)}</span>
                  <span class="card-price">$${prod.price.toFixed(2)}</span>
                </div>
                <h4 class="card-title">${escapeHtml(prod.title)}</h4>
                <p class="card-reason">${escapeHtml(prod.reason)}</p>
              </div>
              <div>
                <div class="card-actives">
                  ${prod.key_ingredients.map(ing => `<span class="active-tag">${escapeHtml(ing)}</span>`).join('')}
                </div>
                <div class="card-footer">
                  <span style="font-size: 0.72rem; color: var(--text-muted);">${escapeHtml(prod.brand)}</span>
                  <button class="citation-btn" data-citation="${escapeHtml(prod.doc_citation)}">
                    📜 Citation
                  </button>
                </div>
              </div>
            </div>
          `).join('')}
        </div>
      `;
    }

    let bundleHtml = '';
    if (data.bundle_offer) {
      const b = data.bundle_offer;
      bundleHtml = `
        <div class="bundle-callout-card">
          <div class="bundle-header">
            <span class="bundle-title">🎁 ${escapeHtml(b.bundle_title)}</span>
            <span class="bundle-savings-badge">Save ${b.discount_percent.toFixed(0)}%</span>
          </div>
          <div class="bundle-pricing">
            <span class="bundle-discounted-price">$${b.discounted_total.toFixed(2)}</span>
            <span class="bundle-original-price">$${b.original_total.toFixed(2)}</span>
            <span style="font-size: 0.85rem; color: #6ee7b7; font-weight: 600;">(Save $${b.savings_amount.toFixed(2)})</span>
          </div>
          <p class="bundle-pitch">${escapeHtml(b.aov_pitch)}</p>
        </div>
      `;
    }

    let citationsHtml = '';
    if (data.policy_citations && data.policy_citations.length > 0) {
      citationsHtml = `
        <div style="margin-top: 14px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
          <span style="font-size: 0.75rem; color: var(--text-muted); font-weight: 600;">VERIFIED STORE POLICIES:</span>
          ${data.policy_citations.map(c => `
            <button class="citation-btn" data-citation="${escapeHtml(c)}">${escapeHtml(c)}</button>
          `).join('')}
        </div>
      `;
    }

    const latency = data.retrieval_stats ? data.retrieval_stats.latency_ms : 0;
    const groundedness = data.retrieval_stats ? data.retrieval_stats.groundedness_score : 1.0;

    bubble.innerHTML = `
      <div class="bubble-header">
        <span class="bubble-avatar">✨</span>
        <span class="bubble-name">OmniCart Clinical Advisor</span>
        <span class="bubble-badge">RAG Latency: ${latency}ms</span>
        <span class="bubble-badge" style="border-color: rgba(16, 185, 129, 0.4); color: #6ee7b7;">Grounded: ${(groundedness * 100).toFixed(0)}%</span>
      </div>
      <div class="bubble-body">
        ${guardrailHtml}
        <div style="white-space: pre-line; line-height: 1.5;">${formatMarkdown(data.grounded_summary)}</div>
        ${productsHtml}
        ${bundleHtml}
        ${citationsHtml}
      </div>
    `;

    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Attach Citation click events
    bubble.querySelectorAll('.citation-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        openCitationModal(btn.dataset.citation);
      });
    });
  }

  // Citation Modal
  function openCitationModal(citationText) {
    modalCitationTag.textContent = citationText.split('(')[0].trim();
    modalCitationText.textContent = citationText;
    citationModal.classList.add('active');
  }

  modalCloseBtn.addEventListener('click', () => {
    citationModal.classList.remove('active');
  });

  citationModal.addEventListener('click', (e) => {
    if (e.target === citationModal) citationModal.classList.remove('active');
  });

  // Load Catalog View
  async function loadCatalog() {
    try {
      const res = await fetch('/api/catalog');
      catalogData = await res.json();
      renderCatalog(catalogData);
      document.getElementById('cat-count-all').textContent = catalogData.length;
    } catch (err) {
      console.error('Failed to load catalog', err);
    }
  }

  function renderCatalog(items) {
    const grid = document.getElementById('catalog-grid');
    grid.innerHTML = items.map(p => `
      <div class="cat-item-card">
        <div class="cat-item-header">
          <div>
            <span class="card-category">${escapeHtml(p.category)}</span>
            <h3 style="font-size: 1.05rem; font-weight: 700; margin-top: 2px;">${escapeHtml(p.title)}</h3>
            <span style="font-size: 0.8rem; color: var(--text-muted);">${escapeHtml(p.brand)}</span>
          </div>
          <span class="stock-badge ${p.in_stock ? 'in-stock' : 'out-of-stock'}">
            ${p.in_stock ? 'In Stock' : 'Sold Out'}
          </span>
        </div>
        <p style="font-size: 0.84rem; color: var(--text-secondary); line-height: 1.4;">
          ${escapeHtml(p.benefits[0])}
        </p>
        <div class="card-actives">
          ${p.key_ingredients.map(ing => `<span class="active-tag">${escapeHtml(ing)}</span>`).join('')}
        </div>
        <div class="card-footer" style="margin-top: auto;">
          <span class="card-price">$${p.price.toFixed(2)}</span>
          <button class="citation-btn" data-citation="${escapeHtml(p.doc_citation)}">
            Evidence 📜
          </button>
        </div>
      </div>
    `).join('');

    grid.querySelectorAll('.citation-btn').forEach(btn => {
      btn.addEventListener('click', () => openCitationModal(btn.dataset.citation));
    });
  }

  // Catalog Filter Buttons
  const filterBtns = document.querySelectorAll('.filter-btn');
  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const category = btn.dataset.filter;
      if (category === 'All') {
        renderCatalog(catalogData);
      } else {
        renderCatalog(catalogData.filter(p => p.category.toLowerCase().includes(category.toLowerCase())));
      }
    });
  });

  // Load Evals Benchmark View
  async function loadEvals() {
    try {
      const res = await fetch('/api/evals');
      const evalData = await res.json();
      renderEvals(evalData);
    } catch (err) {
      console.error('Failed to load evals', err);
    }
  }

  function renderEvals(data) {
    document.getElementById('kpi-pass-rate').textContent = `${data.pass_rate_percentage}%`;
    document.getElementById('kpi-mrr').textContent = data.mean_reciprocal_rank_mrr.toFixed(2);
    document.getElementById('kpi-groundedness').textContent = data.mean_groundedness_score.toFixed(2);

    const tbody = document.getElementById('evals-table-body');
    tbody.innerHTML = data.results.map(r => `
      <tr>
        <td style="font-family: var(--font-mono); font-weight: 600; color: white;">${escapeHtml(r.test_id)}</td>
        <td style="max-width: 320px; color: #cbd5e1;">${escapeHtml(r.query)}</td>
        <td>
          <span class="${r.passed ? 'badge-pass' : 'badge-fail'}">
            ${r.passed ? 'PASS' : 'FAIL'}
          </span>
        </td>
        <td style="font-family: var(--font-mono);">${r.mrr_score.toFixed(2)}</td>
        <td style="font-family: var(--font-mono);">${(r.groundedness_score * 100).toFixed(0)}%</td>
        <td>${r.guardrail_correct ? '✅ Safe' : '⚠️ Alert'}</td>
        <td>${r.bundle_calculated ? '🎁 Bundle Tool' : '—'}</td>
      </tr>
    `).join('');
  }

  // Run Live Evals Button
  const runEvalsBtn = document.getElementById('run-evals-btn');
  const evalSpinner = document.getElementById('eval-spinner');
  runEvalsBtn.addEventListener('click', async () => {
    runEvalsBtn.disabled = true;
    runEvalsBtn.innerHTML = '<span>⚡ Running Benchmark Suite...</span>';
    try {
      const res = await fetch('/api/evals/run', { method: 'POST' });
      const freshData = await res.json();
      renderEvals(freshData);
    } catch (err) {
      alert('Error running benchmark: ' + err.message);
    } finally {
      runEvalsBtn.disabled = false;
      runEvalsBtn.innerHTML = '<span>⚡ Run Live Evaluation Benchmark</span>';
    }
  });

  // Helpers
  function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/[&<>'"]/g, tag => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      "'": '&#39;',
      '"': '&quot;'
    }[tag] || tag));
  }

  function formatMarkdown(text) {
    if (!text) return '';
    return escapeHtml(text)
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>');
  }

  // Initial Data Loads
  loadCatalog();
  loadEvals();
});
