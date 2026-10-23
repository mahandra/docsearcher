// Streams an assistant reply into its placeholder via Server-Sent Events.
//
// The server renders `<article data-stream-url=...>`; we open an EventSource to it and
// append `token` events as plain text. LLM output is untrusted, so it is only ever
// inserted with textContent, never innerHTML.
(function () {
  "use strict";

  function formatMeta(data) {
    const parts = [];
    if (data.status === "interrupted") parts.push("Interrupted.");
    if (data.status === "error") parts.push("Failed to get an answer.");
    if (data.stop_reason === "max_tokens") parts.push("Answer truncated (max tokens).");
    if (data.stop_reason === "refusal") parts.push("The model declined to answer.");
    if (data.input_tokens != null) {
      let usage = `${data.model} · ${data.input_tokens} in / ${data.output_tokens} out`;
      if (data.cost_usd != null) usage += ` · $${Number(data.cost_usd).toFixed(6)}`;
      parts.push(usage);
    }
    return parts.join(" ");
  }

  function scrollToBottom() {
    window.scrollTo({ top: document.body.scrollHeight });
  }

  function startStream(el) {
    if (el.dataset.started) return;
    el.dataset.started = "1";

    const content = el.querySelector(".content");
    const meta = el.querySelector(".meta");
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

    source.addEventListener("thinking", () => {
      meta.textContent = "Thinking…";
    });

    source.addEventListener("token", (event) => {
      if (!content.textContent) meta.textContent = "Writing…";
      content.textContent += JSON.parse(event.data).text;
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
