// Streams an assistant reply into its placeholder via Server-Sent Events.
//
// The server renders `<article data-stream-url=...>`; we open an EventSource to it and
// append `token` events as plain text. LLM output and document names are untrusted, so
// they are only ever inserted as text nodes / textContent, never innerHTML.
//
// Citations: `sources` lists the chunks sent to the model; each `citation` event marks
// the text streamed so far as supported by one of them. Footnotes are numbered in order
// of first citation — the same rule as Message.citation_numbers on the server.
(function () {
  "use strict";

  function formatMeta(data) {
    const parts = [];
    if (data.status === "interrupted") parts.push("Interrupted.");
    if (data.status === "error") parts.push("Failed to get an answer.");
    if (data.stop_reason === "max_tokens") parts.push("Answer truncated (max tokens).");
    if (data.stop_reason === "refusal") parts.push("The model declined to answer.");
    if (data.input_tokens != null) {
      let usage = `${data.model} · ${data.prompt_tokens} in`;
      if (data.cache_read_tokens) usage += ` (${data.cache_read_tokens} cached)`;
      usage += ` / ${data.output_tokens} out`;
      if (data.cost_usd != null) usage += ` · $${Number(data.cost_usd).toFixed(6)}`;
      parts.push(usage);
    }
    if (data.retrieval_cost_usd != null) {
      parts.push(`· search $${Number(data.retrieval_cost_usd).toFixed(6)}`);
    }
    if (data.rewrite_cost_usd != null) {
      parts.push(`· rewrite $${Number(data.rewrite_cost_usd).toFixed(6)}`);
    }
    if (data.search_query) parts.push(`· searched for: ${data.search_query}`);
    return parts.join(" ");
  }

  function scrollToBottom() {
    window.scrollTo({ top: document.body.scrollHeight });
  }

  function startStream(el) {
    if (el.dataset.started) return;
    el.dataset.started = "1";

    const content = el.querySelector(".content");
    const footnotes = el.querySelector(".sources");
    const meta = el.querySelector(".meta");
    const sources = new Map(); // position -> {document, location, url}
    const numbers = new Map(); // position -> footnote number
    const source = new EventSource(el.dataset.streamUrl);
    let finished = false;

    function finish(text) {
      finished = true;
      // Must close explicitly: EventSource reconnects automatically when the server
      // ends the response, which would request (and pay for) another generation.
      source.close();
      el.classList.remove("streaming");
      meta.textContent = text;
    }

    source.addEventListener("sources", (event) => {
      for (const s of JSON.parse(event.data).sources) sources.set(s.position, s);
      meta.textContent = sources.size
        ? `Found ${sources.size} relevant passages…`
        : "No relevant passages found…";
    });

    source.addEventListener("citation", (event) => {
      const data = JSON.parse(event.data);
      const src = sources.get(data.source);
      if (!numbers.has(data.source)) {
        numbers.set(data.source, numbers.size + 1);
        addFootnote(src);
      }
      const marker = document.createElement(src ? "a" : "span");
      marker.className = "cite";
      if (src) marker.href = src.url;
      marker.title = data.cited_text;
      marker.textContent = `[${numbers.get(data.source)}]`;
      content.append(marker);
    });

    function addFootnote(src) {
      const li = document.createElement("li");
      if (src) {
        const link = document.createElement("a");
        link.href = src.url;
        link.textContent = src.document;
        li.append(link);
        if (src.location) li.append(` · ${src.location}`);
      } else {
        li.textContent = "Unknown source";
      }
      footnotes.append(li);
      footnotes.hidden = false;
    }

    source.addEventListener("thinking", () => {
      meta.textContent = "Thinking…";
    });

    source.addEventListener("token", (event) => {
      if (!content.textContent) meta.textContent = "Writing…";
      // append() adds a text node; `textContent +=` would wipe the citation markers.
      content.append(JSON.parse(event.data).text);
      scrollToBottom();
    });

    source.addEventListener("done", (event) => {
      finish(formatMeta(JSON.parse(event.data)));
    });

    source.addEventListener("failed", (event) => {
      const data = JSON.parse(event.data);
      el.classList.add("error");
      finish([data.error, formatMeta(data)].filter(Boolean).join(" "));
    });

    // Network errors, and also a 204 from the server ("nothing to generate").
    source.onerror = () => {
      if (finished) return;
      el.classList.add("error");
      finish(
        content.textContent
          ? "Connection lost. Reload the page to see the saved answer."
          : "No answer is being streamed. Reload the page in a moment."
      );
    };
  }

  // htmx.onLoad runs for the initial page and for every swapped-in fragment.
  htmx.onLoad((root) => {
    const nodes = root.matches?.("[data-stream-url]")
      ? [root]
      : root.querySelectorAll("[data-stream-url]");
    nodes.forEach(startStream);
    scrollToBottom();
  });
})();
