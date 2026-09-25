// MMH3 Prompt Load: picking a saved prompt copies it into the editable 'text' box,
// and 'save text over file' writes the box back to that saved prompt.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

async function fetchSaved(name) {
  const r = await api.fetchApi(`/mmh3/api/prompt?name=${encodeURIComponent(name)}`);
  return r.ok ? (await r.json()).prompt : null;
}

app.registerExtension({
  name: "dihan.mmh3.prompt.load",
  nodeCreated(node) {
    if (node.comfyClass !== "MMH3PromptLoad") return;
    const pick = node.widgets.find((w) => w.name === "prompt_name");
    const text = node.widgets.find((w) => w.name === "text");

    const fill = async (name, replace) => {
      const prompt = await fetchSaved(name);
      // A workflow restores its edited text after nodeCreated; never overwrite that.
      if (prompt === null || (!replace && text.value)) return;
      text.value = prompt;
      app.graph.setDirtyCanvas(true, false);
    };

    const callback = pick.callback;
    pick.callback = function (value) {
      callback?.apply(this, arguments);
      fill(value, true);
    };
    fill(pick.value, false);

    const save = node.addWidget("button", "save text over file", null, async () => {
      const name = pick.value;
      if (!confirm(`Overwrite the saved prompt '${name}' with this text?`)) return;
      const r = await api.fetchApi("/mmh3/api/prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, prompt: text.value }),
      });
      const out = await r.json();
      app.extensionManager.toast.add(r.ok
        ? { severity: "success", summary: `Saved '${name}'`, detail: out.report.split("\n")[0], life: 4000 }
        : { severity: "error", summary: "Not saved", detail: out.error, life: 6000 });
    });
    save.serialize = false;
    save.options = { ...(save.options || {}), serialize: false };
  },
});
