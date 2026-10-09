// Shows the saved prompt file on the "Save Unique Prompt" node.
import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

const NODE_NAME = "KreaSaveUniquePrompt";
const WIDGET_NAME = "saved_prompts";

function showText(node, text) {
	let widget = node.widgets?.find((w) => w.name === WIDGET_NAME);
	if (!widget) {
		widget = ComfyWidgets["STRING"](node, WIDGET_NAME, ["STRING", { multiline: true }], app).widget;
		widget.inputEl.readOnly = true;
		widget.inputEl.style.opacity = 0.7;
		// Display only: never sent to the backend as an input.
		widget.options = { ...(widget.options ?? {}), serialize: false };
		const [w, h] = node.size;
		node.setSize([Math.max(w, 400), Math.max(h, 300)]);
	}
	widget.value = text ?? "";
	if (widget.inputEl) {
		widget.inputEl.value = widget.value;
		widget.inputEl.scrollTop = widget.inputEl.scrollHeight;
	}
	app.graph?.setDirtyCanvas?.(true, false);
}

app.registerExtension({
	name: "KreaPromptGenerator.SaveUniquePrompt",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== NODE_NAME) return;

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);
			const text = message?.krea_file_text;
			if (text?.length) showText(this, String(text[0]));
		};

		// Restore the last view when a saved workflow is reopened.
		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function (info) {
			onConfigure?.apply(this, arguments);
			// The display widget doesn't exist yet, so its saved value is the one left over at the end.
			const values = info?.widgets_values ?? [];
			const saved = values.length > (this.widgets?.length ?? 0) ? values[values.length - 1] : null;
			if (typeof saved === "string" && saved) {
				requestAnimationFrame(() => showText(this, saved));
			}
		};
	},
});
