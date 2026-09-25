// Shows the written prompt + check report inside the MMH3 prompt nodes after they run.
import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const SHOW = new Set(["MMH3PromptWriter", "MMH3PromptRefine", "MMH3PromptCheck", "MMH3PromptSave"]);

app.registerExtension({
  name: "dihan.mmh3.prompt.preview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!SHOW.has(nodeData.name)) return;
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const text = (message?.text || []).join("\n");
      let w = this.widgets?.find((x) => x.name === "mmh3_preview");
      if (!w) {
        w = ComfyWidgets["STRING"](this, "mmh3_preview", ["STRING", { multiline: true }], app).widget;
        w.serialize = false; // never saved into the workflow or sent to the server
        w.options = { ...(w.options || {}), serialize: false };
        if (w.inputEl) {
          w.inputEl.readOnly = true;
          w.inputEl.style.opacity = 0.9;
          w.inputEl.style.fontFamily = "monospace";
          w.inputEl.style.fontSize = "11px";
        }
      }
      w.value = text;
      requestAnimationFrame(() => {
        const sz = this.computeSize();
        this.setSize([Math.max(this.size[0], sz[0], 420), Math.max(this.size[1], sz[1])]);
        app.graph.setDirtyCanvas(true, false);
      });
    };
  },
});
