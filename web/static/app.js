const session = window.__RAG_SESSION__ || {};
const form = document.getElementById("ask-form");
const input = document.getElementById("question");
const chatLog = document.getElementById("chat-log");
const retrievalLog = document.getElementById("retrieval-log");

function appendMessage(text, type) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${type}`;
  wrapper.innerHTML = `<p>${text}</p>`;
  chatLog.appendChild(wrapper);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function renderRetrieval(items) {
  retrievalLog.innerHTML = "";
  if (!items || items.length === 0) {
    retrievalLog.innerHTML = '<p class="helper">No evidence retrieved.</p>';
    return;
  }
  items.forEach((item) => {
    const el = document.createElement("div");
    el.className = "retrieval-item";
    el.innerHTML = `
      <h3>[${item.citation}] score=${item.score} bm25=${item.bm25} cosine=${item.cosine}</h3>
      <p>${item.snippet}</p>
    `;
    retrievalLog.appendChild(el);
  });
}

async function askQuestion(question) {
  const response = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: session.sessionId, question }),
  });

  if (!response.ok) {
    const payload = await response.json();
    throw new Error(payload.error || "Request failed");
  }

  return response.json();
}

if (form) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question) {
      return;
    }

    appendMessage(question, "user");
    input.value = "";
    appendMessage("Thinking...", "assistant");

    try {
      const payload = await askQuestion(question);
      const last = chatLog.lastElementChild;
      if (last && last.classList.contains("assistant")) {
        last.innerHTML = `<p>${payload.answer}</p>`;
      } else {
        appendMessage(payload.answer, "assistant");
      }
      renderRetrieval(payload.retrieved);
    } catch (err) {
      const last = chatLog.lastElementChild;
      if (last && last.classList.contains("assistant")) {
        last.innerHTML = `<p>Error: ${err.message}</p>`;
      } else {
        appendMessage(`Error: ${err.message}`, "assistant");
      }
    }
  });
}
