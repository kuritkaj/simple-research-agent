const chatLog = document.getElementById('chat-log');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const submitButton = chatForm.querySelector('button[type="submit"]');

function escapeHtml(text) {
  return String(text ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function renderMarkdown(text) {
  const raw = String(text ?? '').trim();
  if (!raw) {
    return '';
  }

  if (window.marked) {
    const rendered = window.marked.parse(raw, {
      mangle: false,
      headerIds: false,
      smartypants: true,
    });
    if (window.DOMPurify) {
      return window.DOMPurify.sanitize(rendered, { ALLOW_DATA_ATTR: false });
    }
    return rendered;
  }

  return `<p>${escapeHtml(raw)}</p>`;
}

function addMessage(author, html, role = 'agent') {
  const wrapper = document.createElement('div');
  wrapper.className = `message ${role}`;
  wrapper.innerHTML = `
    <div class="message__author">${escapeHtml(author)}</div>
    <div class="message__content">${html}</div>
  `;
  chatLog.appendChild(wrapper);
  chatLog.scrollTop = chatLog.scrollHeight;
  return wrapper;
}

function updateMessageContent(messageElement, html) {
  const content = messageElement.querySelector('.message__content');
  if (content) {
    content.innerHTML = html;
  }
  chatLog.scrollTop = chatLog.scrollHeight;
}

function formatResults(data) {
  const queries = (data.queries || []).map((query) => `<li>${escapeHtml(query)}</li>`).join('');

  const results = (data.results || []).map((item, index) => `
      <li class="results__item">
        <div class="results__rank">${index + 1}</div>
        <div class="results__body">
          <a href="${escapeHtml(item.url)}" target="_blank" rel="noopener">
            ${escapeHtml(item.title || 'Untitled result')}
          </a>
          <p>${escapeHtml(item.snippet || 'No description available.')}</p>
          <div class="results__meta">From query: ${escapeHtml(item.query || '')}</div>
        </div>
      </li>
    `).join('');

  const warnings = (data.warnings || []).map((warning) => `<div>${escapeHtml(warning)}</div>`).join('');

  const sources = (data.answer_sources || [])
    .filter((source) => source && source.url)
    .map((source) => `
        <li>
          <a href="${escapeHtml(source.url)}" target="_blank" rel="noopener">
            ${escapeHtml(source.title || source.url)}
          </a>
        </li>
      `)
    .join('');

  const answerHtml = renderMarkdown(data.answer);

  const answer = answerHtml ? `
      <div class="results__summary">
        <h4>Research summary</h4>
        <div class="results__summary-body">${answerHtml}</div>
        ${sources ? `<div class="results__summary-sources"><span>Sources</span><ul>${sources}</ul></div>` : ''}
      </div>
    ` : '';

  return `
    <div class="results">
      ${answer}
      <div class="results__queries">
        <h4>Search queries</h4>
        <ol>${queries || '<li>No queries generated.</li>'}</ol>
      </div>
      <div class="results__list">
        <h4>Top results</h4>
        <ol>${results || '<li>No results returned.</li>'}</ol>
      </div>
    
      ${warnings ? `<div class="results__warnings">${warnings}</div>` : ''}
    </div>
  `;
}

chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();

  const message = userInput.value.trim();
  if (!message) {
    return;
  }

  addMessage('You', escapeHtml(message), 'user');
  userInput.value = '';
  userInput.focus();

  submitButton.disabled = true;
  const placeholder = addMessage('Research Agent', '<span class="loading">Researching…</span>');

  try {
    const response = await fetch('/research', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message }),
    });

    const data = await response.json();

    if (!response.ok) {
      const errorMessage = escapeHtml(data.error || 'The research agent encountered an error.');
      updateMessageContent(placeholder, `<p>${errorMessage}</p>`);
      return;
    }

    const formatted = formatResults(data);
    updateMessageContent(placeholder, formatted);
  } catch (error) { // eslint-disable-line no-unused-vars
    updateMessageContent(placeholder, '<p>Network request failed. Please try again.</p>');
  } finally {
    submitButton.disabled = false;
  }
});
