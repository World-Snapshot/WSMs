import { ControlWavePlugin } from '../../plugin-loader.js';
import { WSMTaskGraph } from './task-graph.js?v=20260903-chain-of-visual';



const TURBO_GRADIENT = "linear-gradient(90deg,#30123b,#3e9bfe,#46f884,#f9e721,#7a0403)";
const GRAYSCALE_GRADIENT = "linear-gradient(90deg,#000,#fff)";
const RGB_INPUT_TOKENS = new Set([
    "0-2",
    "Reference_IMG",
    "FirstFrame_RGB",
    "CurrentFrame_RGB_Reference",
    "CurrentFrame_RGB_Condition",
]);

// Fixed WSG Reader visualization contracts. These are semantic channel ranges,
// never per-image percentile statistics and never evaluation scores.
const OUTPUT_ENCODING_RANGES = {
    "3": ["0 · transparent", "255 · opaque", "Rendered alpha", GRAYSCALE_GRADIENT],
    "32-33": ["-3.60", "1.86", "Relative scene depth", TURBO_GRADIENT],
    "34-35": ["0.584 m", "67.8 m", "Metric depth", TURBO_GRADIENT],
    "40": ["0 · no edge", "255 · edge", "Edge response", GRAYSCALE_GRADIENT],
    "126": ["0.308", "100", "Relative solid volume", "linear-gradient(90deg,#182044 0%,#2b4c9a 3%,#168aad 15%,#35b779 35%,#d6cf3f 55%,#e76f51 75%,#f5f0b8 100%)"],
    "127": ["0.426", "14.27", "Relative surface volume", "linear-gradient(90deg,#25113f 0%,#54318c 2%,#8f3c9f 8%,#c4477d 18%,#e66a45 35%,#f3b33d 60%,#fff0a6 100%)"],
    "128": ["0.363", "132.59", "Relative object weight", "linear-gradient(90deg,#101f52 0%,#2456a4 3%,#1c91c0 15%,#38b977 35%,#b7d52b 55%,#f3ad32 75%,#ef4b3e 100%)"],
    "129": ["0.071", "2.378", "Support-surface area", "linear-gradient(90deg,#102b26 0%,#16685c 1.5%,#199b83 5%,#53bd67 10%,#b7d53c 20%,#f0c541 40%,#f07a4a 100%)"],
    "130": ["0.132", "50", "Relative static-friction force", "linear-gradient(90deg,#211047 0%,#52318f 3%,#8450b4 15%,#bb4b91 35%,#e45d62 55%,#f2a43a 75%,#f8df72 100%)"],
    "132": ["18.08", "100", "Material confidence", "linear-gradient(90deg,#5a1422 0%,#9e2529 3%,#dc4b32 15%,#edae32 40%,#c5d63d 65%,#63bd52 85%,#1d9d67 100%)"],
    "133": ["0.60", "2.74", "Relative material density", "linear-gradient(90deg,#111b45 0%,#203f83 2%,#276ca8 8%,#3296ad 20%,#59b69c 40%,#a6c879 65%,#eee6bd 100%)"],
    "134": ["0.30", "0.68", "Static-friction coefficient", "linear-gradient(90deg,#261448 0%,#57368e 2%,#7d5eb6 10%,#4b91c4 30%,#44ba9a 50%,#a4d052 70%,#f1dc55 100%)"],
    "150": ["0", "0.498", "Opacity", "linear-gradient(90deg,#161616 0%,#303846 2%,#52677c 10%,#7898ae 30%,#a8becb 50%,#d2dce2 70%,#fff 100%)"],
    "154": ["1", "1", "Specular shininess (constant in the released encoding)", "linear-gradient(90deg,#30123b,#325a9b,#27828e,#6cbe64,#f2dc54)"],
    "167": ["0.153", "0.886", "Surface height", "linear-gradient(90deg,#30123b,#325a9b,#27828e,#6cbe64,#f2dc54)"],
    "168": ["0 · black", "1 · white", "Roughness", GRAYSCALE_GRADIENT],
    "169": ["0 · black", "1 · white", "Metallicity", GRAYSCALE_GRADIENT],
    "173": ["0 · transparent", "1 · opaque", "Opacity map", GRAYSCALE_GRADIENT],
};

const OUTPUT_ENCODING_SWATCHES = {
    "103": [["#000000","background"],["#ffff00","hair"],["#00ffff","body skin"],["#00ff00","face skin"],["#6e906e","clothes"],["#ff00ff","accessories"]],
    "90-92": [["#000000","empty"],["#bdbdbd","surface"],["#ffffff","highlight"]],
    "122-124": [["#000000","empty"],["#bdbdbd","decoded edge"],["#ffffff","joints / links"]],
    "111": [["#000000","background"],["#b55239","class ID"],["#4d82b8","class ID"],["#57a66d","class ID"]],
    "131": [["#000000","unknown"],["#b55239","brick"],["#e7e2d3","ceramic"],["#d85f8a","fabric"],["#73c9d9","glass"],["#aeb8c2","metal"],["#4d8ed8","plastic"],["#9a6a3a","wood"]],
    "135": [["#4c78a8","false"],["#54a24b","true"]],
    "136": [["#000000","background"],["#b55239","category ID"],["#4d82b8","category ID"],["#57a66d","category ID"]],
    "143": [["#000000","background"],["#b55239","region ID"],["#4d82b8","region ID"],["#57a66d","region ID"]],
    "195": [["#000000","background"],["#ff0000","person"],["#00ff00","violin"],["#0000ff","guitar"],["#ffff00","cello"],["#ff00ff","flute"],["#00ffff","piano"],["#ff8000","ukulele"]],
    "99-101": [["#ff8a00","left"],["#00d9e7","right"],["#e0e0e0","links"],["#000000","empty"]],
};

function outputEncodingSpec(token, label) {
    if (OUTPUT_ENCODING_SWATCHES[token]) {
        let note = "Fixed categorical palette · color identifies class ID";
        if (token === "103") note = "MediaPipe Seg semantic regions";
        if (token === "90-92") note = "Stored SMPLer-X surface rendering · grayscale intensity is not a class ID";
        if (token === "122-124") note = "Stored X-Pose skeleton rendering · grayscale is not categorical";
        if (token === "195") note = "Fixed 26-class AVIS palette · representative entries shown; colors are categorical, not scalar";
        if (token === "131") note = "Fixed 20-class material palette · representative entries shown";
        if (["111", "136", "143"].includes(token)) note = "Deterministic WSG Reader category palette · representative class colors shown";
        return { type: "swatches", items: OUTPUT_ENCODING_SWATCHES[token], note };
    }
    if (token === "36-38") return { type: "swatches", items: [["#e34a4a","R / x"],["#3aaa55","G / y"],["#3977d5","B / z"],["#000000","invalid"]], note: "Unit XYZ direction mapped to RGB; black = invalid padding" };
    if (token === "39") return { type: "gradient", gradient: "linear-gradient(90deg,#000004,#51127c,#b73779,#fc8961,#fcfdbf)", low: "negative", high: "positive", note: "Signed curvature · WSG Reader symmetric normalization" };
    if (token === "41-42") return { type: "gradient", gradient: "linear-gradient(90deg,#ffd8d8,#fff3ce,#dfffdc,#dceeff,#efdfff,#ffd8e9)", low: "direction", high: "magnitude", note: "Median-centered flow: hue = residual direction; strength = residual magnitude; white = dominant motion" };
    if (token === "67-69") {
        return { type: "gradient", gradient: "linear-gradient(90deg,#f00,#fa0,#af0,#0fa,#0af,#a0f)", low: "body", high: "hands / face", note: "DWPose joint/link palette; black = empty" };
    }
    if (["0-2","144-146","147-149","151-153","155-157","161-163","164-166","170-172"].includes(token)) {
        const detail = {
            "0-2": "Direct RGB intensity",
            "144-146": "Ambient RGB coefficients; each channel in [0, 1]",
            "147-149": "Diffuse color; RGB in [0, 1]",
            "151-153": "Specular RGB coefficients; each channel in [0, 1]",
            "155-157": "Irradiance RGB channels in the released WSG encoding",
            "161-163": "Base RGB; channel maxima 0.906 / 0.694 / 0.643",
            "164-166": "Diffuse RGB; channel maxima 0.933 / 0.765 / 0.761",
            "170-172": "Specular RGB; channel maxima 0.396 / 0.369 / 0.357",
        }[token];
        return { type: "swatches", items: [["#e34a4a","R"],["#3aaa55","G"],["#3977d5","B"]], note: detail };
    }
    if (OUTPUT_ENCODING_RANGES[token]) {
        const [low, high, note, gradient] = OUTPUT_ENCODING_RANGES[token];
        return { type: "gradient", gradient, low, high, note };
    }
    return { type: "text", note: `No fixed WSG Reader legend is registered for ${label || token || "this output"}.` };
}


export default class WSMInferencePlugin extends ControlWavePlugin {
    constructor() {
        super({
            id: 'wsm-v1-inference',
            name: 'WSM v1',
            version: '1.0.0',
            author: 'World Snapshot',
            description: 'Select trained Channel Heads by input and output modality.',
            needsViser: false,
            styleMode: 'inherit',
        });
        this.catalog = null;
        this.fullCatalog = null;
        this.demoManifest = null;
        this.demoManifestUrl = null;
        this.demoEnabled = false;
        this.activeDemoRecord = null;
        this.demoAssetCache = new Map();
        this.signature = null;
        this.task = null;
        this.compositeRoute = null;
        this.uploads = new Map();
        this.uploadFilenames = new Map();
        this.inputValidation = new Map();
        this.demoLatentInputs = new Set();
        this.selfPreprocessorInputs = new Set();
        this.taskGraph = null;
        this.graphSelectionMode = 'input';
        this.inferenceJobId = null;
        this.graphCanvasResize = null;
    }

    async render(container) {
        this.container = container;
        container.classList.add('tab-content', 'active');
        container.style.flexDirection = 'column';
        container.style.overflowY = 'auto';
        container.innerHTML = `
            <style>
                .wsm-shell {
                    min-height: 100%;
                    width: 100%;
                    flex: 1 1 auto;
                    background: #f4f6f8;
                    color: #18202a;
                    padding: 22px;
                    box-sizing: border-box;
                }
                .wsm-topbar {
                    width: 100%;
                    margin: 0 0 14px;
                    display: grid;
                    grid-template-columns: minmax(0, 1fr) auto;
                    align-items: start;
                    column-gap: 16px;
                    row-gap: 7px;
                }
                .wsm-title { margin: 0; font-size: 22px; font-weight: 700; }
                .wsm-subtitle { margin: 3px 0 0; color: #647181; font-size: 13px; }
                .wsm-research-note {
                    grid-column: 1 / -1;
                    width: 100%;
                    margin: 0;
                    color: #176b4d;
                    font-size: 12px;
                    line-height: 1.5;
                }
                .wsm-research-note strong { color: #0f5138; font-weight: 700; }
                .wsm-status { color: #647181; font-size: 12px; text-align: right; }
                .wsm-topbar-actions { display:flex; align-items:center; justify-content:flex-end; gap:10px; flex-wrap:wrap; }
                .wsm-demo-toggle { border:1px solid #b8c3cf; border-radius:999px; background:#fff; color:#425164; padding:7px 12px; font:inherit; font-size:12px; font-weight:700; cursor:pointer; }
                .wsm-demo-toggle:hover { border-color:#4f83cc; color:#245f9f; }
                .wsm-demo-toggle.active { border-color:#26735a; background:#e9f7f1; color:#176b4d; box-shadow:0 0 0 2px rgba(38,115,90,.10); }
                .wsm-demo-bar { grid-column:1 / -1; display:grid; grid-template-columns:minmax(280px,.72fr) minmax(420px,1.28fr); align-items:center; gap:14px; border:1px solid #bad8cb; border-left:4px solid #26735a; border-radius:7px; background:#f1faf6; padding:11px 12px; }
                .wsm-demo-bar[hidden] { display:none; }
                .wsm-demo-copy strong { display:block; color:#145a42; font-size:12px; }
                .wsm-demo-copy span { display:block; margin-top:2px; color:#547064; font-size:10px; line-height:1.4; }
                .wsm-demo-controls { display:grid; grid-template-columns:minmax(0,1fr) minmax(145px,.34fr); gap:8px; align-items:center; min-width:0; }
                .wsm-demo-select { width:100%; box-sizing:border-box; border:1px solid #9dbdaf; border-radius:5px; background:#fff; color:#25382f; padding:8px 9px; font:inherit; font-size:12px; }
                .wsm-demo-search { width:100%; box-sizing:border-box; border:1px solid #9dbdaf; border-radius:5px; background:#fff; color:#25382f; padding:8px 9px; font:inherit; font-size:12px; }
                .wsm-demo-search::placeholder { color:#718078; }
                .wsm-head-search { width:190px; border-color:#b8c3cf; color:#263747; }
                .wsm-upload.demo-locked { border-color:#9fc8b6; background:#f8fcfa; }
                .wsm-upload.demo-locked .wsm-dropzone { cursor:zoom-in; border-style:solid; border-color:#8cb7a4; background:#eef8f3; }
                .wsm-demo-lock { color:#176b4d; font-size:10px; font-weight:700; }
                .wsm-lightbox { position:fixed; inset:0; z-index:10050; display:flex; align-items:center; justify-content:center; padding:28px; background:rgba(5,10,15,.88); cursor:zoom-out; }
                .wsm-lightbox[hidden] { display:none; }
                .wsm-lightbox img { display:block; max-width:min(92vw,1100px); max-height:90vh; object-fit:contain; border:1px solid rgba(255,255,255,.35); box-shadow:0 20px 70px rgba(0,0,0,.45); background:#000; }
                .wsm-lightbox-close { position:absolute; top:18px; right:22px; border:0; background:transparent; color:#fff; font-size:34px; line-height:1; cursor:pointer; }
                #wsm-v1-inference .wsm-workspace {
                    width: 100%;
                    box-sizing: border-box;
                    margin: 0;
                    display: grid;
                    grid-template-columns: minmax(0, 1fr);
                    align-items: stretch;
                    min-height: 680px;
                    border: 1px solid #d8dee6;
                    border-radius: 8px;
                    background: #fff;
                    overflow: hidden;
                }
                #wsm-v1-inference .wsm-pane {
                    width: auto;
                    box-sizing: border-box;
                    padding: 20px;
                    min-width: 0;
                }
                #wsm-v1-inference .wsm-pane + .wsm-pane { border-top: 1px solid #d8dee6; }

                .wsm-output-selector-grid {
                    display: grid;
                    grid-template-columns: minmax(0, 1.45fr) minmax(300px, 0.75fr);
                    gap: 20px;
                    align-items: stretch;
                    margin-bottom: 16px;
                }
                .wsm-output-controls { min-width: 0; }
                .wsm-encoding-panel {
                    min-width: 0; box-sizing: border-box; border: 1px solid #aeb4bb;
                    border-radius: 2px; background: #fff; color: #222; padding: 14px 15px 12px;
                    box-shadow: 0 1px 0 rgba(0,0,0,.025);
                }
                .wsm-encoding-kicker { margin-bottom: 4px; color: #555b62; font-size: 9px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
                .wsm-encoding-title { margin: 0; color: #202326; font-size: 15px; font-weight: 700; line-height: 1.25; }
                .wsm-encoding-rule { height: 1px; margin: 9px 0 11px; background: #c8ccd1; }
                .wsm-encoding-role { margin-bottom: 9px; color: #333; font-size: 11px; font-weight: 700; }
                .wsm-encoding-gradient { height: 18px; border: 1px solid #444; background: var(--encoding-gradient, linear-gradient(90deg,#30123b,#3e9bfe,#46f884,#f9e721,#7a0403)); }
                .wsm-encoding-scale { display: flex; justify-content: space-between; gap: 8px; margin-top: 4px; color: #30343a; font-size: 9px; }
                .wsm-encoding-note { margin-top: 8px; color: #353a40; font-size: 10px; line-height: 1.35; }
                .wsm-encoding-swatches { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 8px 12px; }
                .wsm-encoding-swatch { display: flex; align-items: center; min-width: 0; gap: 7px; color: #222; font-size: 10px; }
                .wsm-encoding-swatch i { width: 18px; height: 13px; flex: 0 0 auto; box-sizing: border-box; border: 1px solid #555; background: var(--swatch); }
                .wsm-encoding-placeholder { color: #70757b; font-size: 10px; font-style: italic; line-height: 1.45; }

                .wsm-pane-head {
                    display: flex;
                    align-items: baseline;
                    justify-content: space-between;
                    margin-bottom: 16px;
                }
                .wsm-pane-head h3 { margin: 0; font-size: 16px; }
                .wsm-step { color: #7b8795; font-size: 12px; }
                .wsm-field { margin-bottom: 15px; }
                .wsm-field label {
                    display: block;
                    margin-bottom: 6px;
                    color: #394655;
                    font-size: 12px;
                    font-weight: 650;
                }
                .wsm-model-filter {
                    display: flex !important;
                    align-items: center;
                    gap: 7px;
                    margin: 0 0 8px;
                    font-weight: 500 !important;
                    cursor: pointer;
                }
                .wsm-model-filter input { width: auto; margin: 0; }
                .wsm-model-filter {
                    display: flex !important;
                    align-items: center;
                    gap: 7px;
                    margin: 0 0 8px;
                    font-weight: 500 !important;
                    cursor: pointer;
                }
                .wsm-model-filter input { width: auto; margin: 0; }
                .wsm-input, .wsm-select, .wsm-textarea {
                    width: 100%;
                    box-sizing: border-box;
                    border: 1px solid #c9d1dc;
                    border-radius: 6px;
                    background: #fff;
                    color: #18202a;
                    padding: 9px 10px;
                    font: inherit;
                    font-size: 13px;
                }
                .wsm-textarea { min-height: 78px; resize: vertical; }
                .wsm-input:focus, .wsm-select:focus, .wsm-textarea:focus {
                    outline: 2px solid #b9d2f7;
                    border-color: #4f83cc;
                }
                .wsm-upload-list { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; align-items: start; }
                .wsm-upload {
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                    align-items: stretch;
                    border: 1px solid #d8dee6;
                    border-radius: 7px;
                    padding: 12px;
                }
                .wsm-text-input-card { min-height: 246px; box-sizing: border-box; }
                .wsm-text-input-body { display: flex; flex: 1 1 auto; min-height: 180px; }
                .wsm-text-input-card .wsm-textarea { min-height: 180px; height: 100%; resize: vertical; }
                .wsm-text-input-card .wsm-textarea[readonly] { background: #f1f4f7; color: #536171; cursor: default; }
                .wsm-preview-stage { position:relative; width:100%; }
                .wsm-preview-notice { position:absolute; top:7px; right:7px; z-index:2; display:grid; place-items:center; width:19px; height:19px; border:1px solid rgba(255,255,255,.8); border-radius:50%; background:rgba(18,28,38,.82); color:#fff; font-size:12px; font-weight:800; cursor:help; box-shadow:0 1px 5px rgba(0,0,0,.22); }
                .wsm-preview-notice[hidden], .wsm-preview-caption[hidden], .wsm-variable-file[hidden] { display:none; }
                .wsm-preview-caption { margin-top:6px; color:#617181; font-size:10px; line-height:1.35; text-align:left; }
                .wsm-variable-file { display:grid; grid-template-columns:auto minmax(0,1fr); gap:9px; align-items:center; min-height:48px; box-sizing:border-box; padding:8px 9px; border:1px solid #c8d3df; border-radius:5px; background:#f7f9fb; text-align:left; }
                .wsm-variable-extension { min-width:37px; padding:6px 5px; border-radius:4px; background:#344b63; color:#fff; font-size:9px; font-weight:800; letter-spacing:.05em; text-align:center; }
                .wsm-variable-copy { min-width:0; }
                .wsm-variable-name { overflow:hidden; color:#2f3d4c; font-size:10px; font-weight:700; text-overflow:ellipsis; white-space:nowrap; }
                .wsm-variable-role { margin-top:2px; color:#6c7987; font-size:9px; line-height:1.3; }
                .wsm-upload-preview {
                    width: 100%;
                    aspect-ratio: 16 / 9;
                    height: auto;
                    max-height: 320px;
                    object-fit: contain;
                    background: #eef1f4;
                    border-radius: 4px;
                }
                .wsm-dropzone {
                    position: relative;
                    width: 100%;
                    min-height: 180px;
                    box-sizing: border-box;
                    display: grid;
                    place-items: center;
                    border: 1px dashed #aeb9c6;
                    border-radius: 6px;
                    background: #f7f9fb;
                    color: #647181;
                    cursor: pointer;
                    text-align: center;
                    padding: 20px;
                }
                .wsm-dropzone:hover, .wsm-dropzone.dragging {
                    border-color: #4f83cc;
                    background: #eef5ff;
                    color: #285f9f;
                }
                .wsm-dropzone.has-image { padding: 0; border-style: solid; background: #eef1f4; }
                .wsm-dropzone-copy { line-height: 1.5; }
                .wsm-dropzone-copy strong { display: block; color: #394655; font-size: 13px; }
                .wsm-dropzone-copy span { display: block; margin-top: 4px; font-size: 11px; }
                .wsm-upload-title-row {
                    display: grid;
                    grid-template-columns: minmax(0, 1fr);
                    align-items: start;
                    gap: 6px;
                }
                .wsm-text-input-card .wsm-upload-title-row { grid-template-columns:minmax(0,1fr) auto; align-items:center; }
                .wsm-upload-title-actions { display:flex; width:100%; min-width:0; align-items:center; justify-content:space-between; gap:7px; }
                .wsm-demo-input-toggle { display:inline-flex; align-items:center; gap:5px; color:#536474; font-size:9px; font-weight:700; cursor:pointer; white-space:nowrap; }
                .wsm-demo-input-toggle input { width:13px; height:13px; margin:0; accent-color:#26735a; cursor:pointer; }
                .wsm-latent-source-toggle { position:relative; display:inline-grid; grid-template-columns:auto auto; gap:0; padding:2px; border:1px solid #b8c3cf; border-radius:999px; background:#eef1f4; }
                .wsm-latent-source-toggle input { position:absolute; width:1px; height:1px; opacity:0; pointer-events:none; }
                .wsm-latent-source-option { padding:3px 7px; border-radius:999px; color:#6a7785; line-height:1.15; transition:background .16s ease,color .16s ease,box-shadow .16s ease; }
                .wsm-latent-source-toggle input:not(:checked) + .wsm-latent-source-option,
                .wsm-latent-source-toggle input:checked + .wsm-latent-source-option + .wsm-latent-source-option { background:#fff; color:#245f9f; box-shadow:0 1px 3px rgba(26,38,50,.18); }
                .wsm-latent-source-toggle input:focus-visible + .wsm-latent-source-option { outline:2px solid #4f83cc; outline-offset:1px; }
                .wsm-latent-source-toggle:has(input:disabled) { cursor:wait; opacity:.68; }
                .wsm-self-preprocessor-toggle { position:relative; gap:7px; }
                .wsm-self-preprocessor-toggle input { position:absolute; width:1px; height:1px; opacity:0; pointer-events:none; }
                .wsm-self-preprocessor-track { position:relative; width:31px; height:17px; flex:0 0 auto; border:1px solid #aeb9c4; border-radius:999px; background:#dfe4e9; box-sizing:border-box; transition:background .16s ease,border-color .16s ease,box-shadow .16s ease; }
                .wsm-self-preprocessor-track::after { content:""; position:absolute; top:2px; left:2px; width:11px; height:11px; border-radius:50%; background:#fff; box-shadow:0 1px 3px rgba(26,38,50,.28); transition:transform .16s ease; }
                .wsm-self-preprocessor-toggle input:checked + .wsm-self-preprocessor-track { border-color:#26735a; background:#2f8b69; box-shadow:0 0 0 2px rgba(47,139,105,.10); }
                .wsm-self-preprocessor-toggle input:checked + .wsm-self-preprocessor-track::after { transform:translateX(14px); }
                .wsm-self-preprocessor-toggle input:focus-visible + .wsm-self-preprocessor-track { outline:2px solid #4f83cc; outline-offset:2px; }
                .wsm-upload-title { min-width:0; min-height:30px; font-size:12px; font-weight:700; line-height:1.25; display:-webkit-box; -webkit-box-orient:vertical; -webkit-line-clamp:2; overflow:hidden; overflow-wrap:anywhere; }
                .wsm-role-badge {
                    flex: 0 0 auto;
                    border: 1px solid #b8c8dc;
                    border-radius: 999px;
                    background: #edf4fc;
                    color: #315e91;
                    padding: 3px 7px;
                    font-size: 9px;
                    font-weight: 750;
                    text-transform: uppercase;
                }
                .wsm-file { display: none; }
                .wsm-hint { color: #748191; font-size: 11px; line-height: 1.45; margin-top: 5px; }
                .wsm-upload > .wsm-hint { min-height:48px; max-height:48px; display:-webkit-box; -webkit-box-orient:vertical; -webkit-line-clamp:3; overflow:hidden; overflow-wrap:anywhere; }
                .wsm-input-validation { border:1px solid transparent; border-radius:6px; padding:8px 9px; font-size:11px; line-height:1.45; white-space:pre-line; }
                .wsm-input-validation[hidden] { display:none; }
                .wsm-input-validation[data-severity="checking"] { background:#f3f5f7; border-color:#d8dee6; color:#596675; }
                .wsm-input-validation[data-severity="green"] { background:#edf8f1; border-color:#a9d5b8; color:#245b36; }
                .wsm-input-validation[data-severity="orange"] { background:#fff6e8; border-color:#edc77e; color:#80510d; }
                .wsm-input-validation[data-severity="red"] { background:#fff0f0; border-color:#e3a1a1; color:#8b2929; }
                .wsm-frame-guide {
                    margin-top: 9px;
                    border-left: 3px solid #4f83cc;
                    background: #f2f7fd;
                    padding: 9px 11px;
                    color: #46576a;
                    font-size: 11px;
                    line-height: 1.5;
                }
                .wsm-frame-guide strong { display: block; margin-bottom: 3px; color: #25384c; }
                .wsm-frame-guide span { display: block; }
                .wsm-signature-list, .wsm-task-list {
                    width: 100%;
                    box-sizing: border-box;
                }
                .wsm-signature-picker { position: relative; }
                .wsm-signature-trigger {
                    width: 100%;
                    min-height: 42px;
                    box-sizing: border-box;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 12px;
                    border: 1px solid #c9d1dc;
                    border-radius: 6px;
                    background: #fff;
                    color: #18202a;
                    padding: 9px 10px;
                    text-align: left;
                    white-space: normal;
                    overflow-wrap: anywhere;
                    line-height: 1.4;
                    cursor: pointer;
                }
                .wsm-signature-arrow { flex: 0 0 auto; color: #718091; font-size: 11px; }
                .wsm-signature-panel {
                    display: none;
                    position: absolute;
                    z-index: 30;
                    top: calc(100% + 5px);
                    left: 0;
                    right: 0;
                    padding: 8px;
                    border: 1px solid #c9d1dc;
                    border-radius: 7px;
                    background: #fff;
                    box-shadow: 0 12px 28px rgba(25, 38, 53, 0.16);
                }
                .wsm-signature-picker.open .wsm-signature-panel { display: block; }
                .wsm-signature-panel .wsm-input { margin-bottom: 8px; }
                .wsm-signature-list { max-height: 300px; overflow-y: auto; border: 1px solid #e1e6ec; border-radius: 5px; }
                .wsm-signature {
                    width: 100%;
                    display: block;
                    border: 0;
                    border-bottom: 1px solid #e7ebf0;
                    background: #fff;
                    padding: 10px 11px;
                    color: #18202a;
                    text-align: left;
                    white-space: normal;
                    overflow-wrap: anywhere;
                    line-height: 1.45;
                    cursor: pointer;
                }
                .wsm-signature:last-child { border-bottom: 0; }
                .wsm-signature:hover { background: #f3f7fc; }
                .wsm-signature.active { background: #e8f1ff; box-shadow: inset 3px 0 #3478c8; }
                .wsm-signature-label { font-size: 12px; font-weight: 650; }
                .wsm-signature-count { margin-top: 3px; color: #8994a1; font-size: 10px; }
                .wsm-task-list {
                    height: 300px;
                    overflow-y: auto;
                    border: 1px solid #d8dee6;
                    border-radius: 6px;
                }
                .wsm-task {
                    width: 100%;
                    border: 0;
                    border-bottom: 1px solid #e7ebf0;
                    background: #fff;
                    padding: 10px 11px;
                    text-align: left;
                    cursor: pointer;
                }
                .wsm-task:last-child { border-bottom: 0; }
                .wsm-task:hover { background: #f3f7fc; }
                .wsm-task.active { background: #e8f1ff; box-shadow: inset 3px 0 #3478c8; }
                .wsm-task:disabled {
                    background: #f7f8fa;
                    color: #7b8795;
                    cursor: not-allowed;
                    opacity: 0.75;
                }
                .wsm-task-name-row {
                    display: flex;
                    align-items: center;
                    flex-wrap: wrap;
                    gap: 5px 7px;
                }
                .wsm-task-name { font-size: 12px; font-weight: 700; }
                .wsm-task-badge {
                    display: inline-flex;
                    align-items: center;
                    border-radius: 999px;
                    padding: 2px 6px;
                    font-size: 9px;
                    font-weight: 750;
                    line-height: 1.25;
                }
                .wsm-task-badge.recommended {
                    border: 1px solid #aac5e6;
                    background: #edf5ff;
                    color: #245d9c;
                }
                .wsm-task-badge.warning {
                    border: 1px solid #e0b95b;
                    background: #fff7dc;
                    color: #775510;
                }
                .wsm-task-target { margin-top: 3px; font-size: 12px; color: #4d5b6a; }
                .wsm-task-type { margin-top: 3px; font-size: 10px; color: #8994a1; }
                .wsm-output-stage {
                    min-height: 318px;
                    background: #11161d;
                    display: grid;
                    place-items: center;
                    border-radius: 7px;
                    overflow: hidden;
                    position: relative;
                }
                .wsm-result { display:block; width:min(100%,500px); aspect-ratio:1 / 1; max-height:500px; object-fit:contain; background:#000; }
                .wsm-result[hidden] { display: none; }
                .wsm-empty { color: #9ca7b4; font-size: 13px; padding: 28px; text-align: center; }
                .wsm-output-list {
                    margin-top: 10px;
                    padding: 10px;
                    border: 1px solid #d8dee6;
                    border-radius: 7px;
                    background: #f8fafc;
                }
                .wsm-output-list[hidden] { display: none; }
                .wsm-output-list-title {
                    margin-bottom: 8px;
                    color: #4a5868;
                    font-size: 11px;
                    font-weight: 700;
                }
                .wsm-output-items {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
                    gap: 8px;
                    max-height: 240px;
                    overflow-y: auto;
                }
                .wsm-output-item {
                    min-width: 0;
                    overflow: hidden;
                    padding: 0;
                    border: 1px solid #d5dde7;
                    border-radius: 6px;
                    background: #ffffff;
                    color: #344253;
                    cursor: pointer;
                    text-align: left;
                }
                .wsm-output-item:hover { border-color: #79a7dc; }
                .wsm-output-item.active {
                    border-color: #2768b2;
                    box-shadow: 0 0 0 2px rgba(39, 104, 178, 0.15);
                }
                .wsm-output-item img {
                    display: block;
                    width: 100%;
                    height: 82px;
                    object-fit: contain;
                    background: #11161d;
                }
                .wsm-output-item span {
                    display: block;
                    overflow: hidden;
                    padding: 7px 8px;
                    font-size: 10px;
                    line-height: 1.35;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                }
                .wsm-runbar {
                    display: grid;
                    grid-template-columns: 100px 100px 120px 1fr;
                    gap: 9px;
                    margin-top: 14px;
                    align-items: end;
                }
                .wsm-button {
                    min-height: 38px;
                    border: 0;
                    border-radius: 6px;
                    background: #2768b2;
                    color: #fff;
                    font-size: 13px;
                    font-weight: 700;
                    cursor: pointer;
                }
                .wsm-button:hover { background: #1e5898; }
                .wsm-button:disabled { background: #9aa7b5; cursor: not-allowed; }
                .wsm-log-panel {
                    margin-top: 12px;
                    overflow: hidden;
                    border: 1px solid #263241;
                    border-radius: 7px;
                    background: #11161d;
                    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.035);
                }
                .wsm-log-panel[hidden] { display: none; }
                .wsm-log-head {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 12px;
                    padding: 8px 10px;
                    border-bottom: 1px solid #263241;
                    color: #dce5ef;
                    font-size: 11px;
                    font-weight: 700;
                }
                .wsm-log-status {
                    color: #88b9eb;
                    font-size: 10px;
                    font-weight: 650;
                    text-transform: uppercase;
                    letter-spacing: 0.055em;
                }
                .wsm-log-panel[data-status="completed"] .wsm-log-status { color: #73d49b; }
                .wsm-log-panel[data-status="failed"] .wsm-log-status { color: #ff8d8d; }
                .wsm-log-output {
                    min-height: 92px;
                    max-height: 250px;
                    margin: 0;
                    overflow: auto;
                    padding: 10px 11px;
                    color: #c7d2df;
                    font: 11px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
                    white-space: pre-wrap;
                    overflow-wrap: anywhere;
                    tab-size: 4;
                }
                .wsm-error {
                    display: none;
                    margin-top: 12px;
                    padding: 9px 10px;
                    border-left: 3px solid #b93a3a;
                    background: #fff0f0;
                    color: #7d2828;
                    font-size: 12px;
                    white-space: pre-wrap;
                }
                .wsm-quality-modal-backdrop {
                    position: fixed;
                    z-index: 10000;
                    inset: 0;
                    display: grid;
                    place-items: center;
                    padding: 20px;
                    background: rgba(19, 27, 37, 0.48);
                    backdrop-filter: blur(2px);
                }
                .wsm-quality-modal-backdrop[hidden] { display: none; }
                .wsm-quality-modal {
                    width: min(520px, 100%);
                    box-sizing: border-box;
                    border: 1px solid #dfbd67;
                    border-radius: 12px;
                    background: #fffdf7;
                    box-shadow: 0 24px 70px rgba(18, 27, 38, 0.30);
                    overflow: hidden;
                }
                .wsm-quality-modal-head {
                    display: flex;
                    align-items: flex-start;
                    justify-content: space-between;
                    gap: 18px;
                    padding: 18px 20px 12px;
                }
                .wsm-quality-modal-title {
                    margin: 0;
                    color: #5d430d;
                    font-size: 17px;
                    line-height: 1.3;
                }
                .wsm-quality-modal-close {
                    border: 0;
                    background: transparent;
                    color: #7d6b45;
                    padding: 0 2px;
                    font-size: 22px;
                    line-height: 1;
                    cursor: pointer;
                }
                .wsm-quality-modal-body {
                    padding: 0 20px 16px;
                    color: #4d4535;
                    font-size: 13px;
                    line-height: 1.6;
                }
                .wsm-quality-modal-head-id {
                    margin-bottom: 9px;
                    color: #72520f;
                    font-size: 12px;
                    font-weight: 750;
                }
                .wsm-quality-modal-message { margin: 0; }
                .wsm-quality-modal-actions {
                    display: flex;
                    justify-content: flex-end;
                    padding: 12px 20px 18px;
                    border-top: 1px solid #eee0bd;
                }
                .wsm-quality-modal-continue {
                    border: 0;
                    border-radius: 6px;
                    background: #986f19;
                    color: #fff;
                    padding: 9px 15px;
                    font: inherit;
                    font-size: 12px;
                    font-weight: 700;
                    cursor: pointer;
                }
                .wsm-quality-modal-continue:hover { background: #79560f; }
                .wsm-graph-row {
                    width: 70%;
                    justify-self: center;
                    box-sizing: border-box;
                    order: -1;
                    grid-column: 1 / -1;
                    min-width: 0;
                    padding: 20px 20px 22px;
                    border-bottom: 1px solid #d8dee6;
                    background: linear-gradient(180deg, #fbfcfe 0%, #f5f7fa 100%);
                }
                .wsm-graph-head {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 22px;
                    margin-bottom: 12px;
                }
                .wsm-graph-title {
                    margin: 0;
                    color: #172230;
                    font-size: 18px;
                    letter-spacing: 0.01em;
                }
                .wsm-graph-summary { margin-top: 4px; color: #5c6978; font-size: 12px; }
                .wsm-graph-actions {
                    display: inline-flex;
                    flex: 0 0 auto;
                    gap: 4px;
                    padding: 4px;
                    border: 1px solid #d6dee8;
                    border-radius: 9px;
                    background: #edf1f5;
                }
                .wsm-graph-mode {
                    border: 0;
                    border-radius: 6px;
                    background: transparent;
                    color: #5b6877;
                    padding: 7px 11px;
                    font: inherit;
                    font-size: 11px;
                    font-weight: 680;
                    cursor: pointer;
                    transition: background 150ms ease, color 150ms ease, box-shadow 150ms ease;
                }
                .wsm-graph-mode:hover { color: #1f5f9f; }
                .wsm-graph-mode.active {
                    background: #ffffff;
                    color: #1f5f9f;
                    box-shadow: 0 1px 4px rgba(32, 53, 76, 0.14);
                }
                .wsm-graph-canvas {
                    --canvas-handle-color: #8091a5;
                    position: relative;
                    left: 50%;
                    transform: translateX(-50%);
                    width: 100%;
                    aspect-ratio: 1480 / 680;
                    min-width: min(560px, 100%);
                    min-height: 260px;
                    margin: 0 auto;
                    box-sizing: border-box;
                }
                .wsm-graph-viewport {
                    width: 100%;
                    height: 100%;
                    overflow: hidden;
                    border: 1px solid #d9e0e8;
                    border-radius: 12px;
                    background:
                        radial-gradient(circle at 50% 44%, rgba(234, 242, 255, 0.42), transparent 34%),
                        linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
                    box-shadow:
                        0 10px 28px rgba(31, 47, 68, 0.06),
                        inset 0 1px 0 rgba(255, 255, 255, 0.9);
                    box-sizing: border-box;
                }
                .wsm-graph-svg {
                    display: block;
                    width: 100%;
                    min-width: 0;
                    height: 100%;
                    overflow: visible;
                }
                .wsm-canvas-handle {
                    position: absolute;
                    z-index: 8;
                    border: 0;
                    padding: 0;
                    margin: 0;
                    background: transparent;
                    touch-action: none;
                }
                .wsm-canvas-handle::after {
                    content: "";
                    position: absolute;
                    border-radius: 999px;
                    background: var(--canvas-handle-color);
                    opacity: .45;
                    transition: opacity 140ms ease, background 140ms ease;
                }
                .wsm-canvas-handle:hover::after,
                .wsm-canvas-handle.resizing::after { opacity: .9; background: #2f6db3; }
                .wsm-canvas-handle.top,
                .wsm-canvas-handle.bottom { left: 28px; right: 28px; height: 12px; cursor: ns-resize; }
                .wsm-canvas-handle.top { top: -6px; }
                .wsm-canvas-handle.bottom { bottom: -6px; }
                .wsm-canvas-handle.top::after,
                .wsm-canvas-handle.bottom::after { left: 45%; right: 45%; height: 3px; top: 4px; }
                .wsm-canvas-handle.left,
                .wsm-canvas-handle.right { top: 28px; bottom: 28px; width: 12px; cursor: ew-resize; }
                .wsm-canvas-handle.left { left: -6px; }
                .wsm-canvas-handle.right { right: -6px; }
                .wsm-canvas-handle.left::after,
                .wsm-canvas-handle.right::after { top: 45%; bottom: 45%; width: 3px; left: 4px; }
                .wsm-canvas-handle.corner { width: 22px; height: 22px; }
                .wsm-canvas-handle.corner::after {
                    width: 10px;
                    height: 10px;
                    background: transparent;
                    border-color: var(--canvas-handle-color);
                    border-style: solid;
                    border-radius: 2px;
                }
                .wsm-canvas-handle.top-left { left: -7px; top: -7px; cursor: nwse-resize; }
                .wsm-canvas-handle.top-right { right: -7px; top: -7px; cursor: nesw-resize; }
                .wsm-canvas-handle.bottom-right { right: -7px; bottom: -7px; cursor: nwse-resize; }
                .wsm-canvas-handle.bottom-left { left: -7px; bottom: -7px; cursor: nesw-resize; }
                .wsm-canvas-handle.top-left::after { left: 5px; top: 5px; border-width: 2px 0 0 2px; }
                .wsm-canvas-handle.top-right::after { right: 5px; top: 5px; border-width: 2px 2px 0 0; }
                .wsm-canvas-handle.bottom-right::after { right: 5px; bottom: 5px; border-width: 0 2px 2px 0; }
                .wsm-canvas-handle.bottom-left::after { left: 5px; bottom: 5px; border-width: 0 0 2px 2px; }
                .wsm-graph-plate { pointer-events: none; }
                .wsm-graph-plate.shadow {
                    fill: #f4f6f8;
                    fill-opacity: 0.55;
                    stroke: none;
                    filter: url(#wsm-graph-soft-shadow);
                }
                .wsm-graph-plate.inner {
                    fill: #ffffff;
                    fill-opacity: 0.94;
                    stroke: #d1d5db;
                    stroke-width: 1.35;
                }
                .wsm-graph-plate.outer {
                    fill: none;
                    stroke: #c1c7d0;
                    stroke-width: 1.55;
                    stroke-dasharray: 7 6;
                }
                .wsm-graph-space-label,
                .wsm-graph-audio-label {
                    fill: #172230;
                    font-weight: 500;
                    pointer-events: none;
                }
                .wsm-graph-space-label { font-size: 18px; }
                .wsm-graph-audio-label { font-size: 15px; }
                .wsm-graph-edge {
                    fill: none;
                    stroke: var(--edge-color);
                    stroke-width: var(--edge-width, 0.85);
                    stroke-opacity: var(--edge-opacity, 0.1);
                    marker-end: url(#wsm-graph-arrow);
                    vector-effect: non-scaling-stroke;
                    transition: stroke-opacity 170ms ease, stroke-width 170ms ease, filter 170ms ease;
                    pointer-events: none;
                }
                .wsm-graph-edge.chain {
                    stroke-width: 0.75;
                    stroke-opacity: 0.46;
                    stroke-dasharray: 10 9;
                }
                .wsm-graph-edge.sparse-task { stroke-dasharray: 5 4; }
                .wsm-graph-edge.primary {
                    stroke-width: 4.8;
                    stroke-opacity: 0.9;
                }
                .wsm-graph-edge.candidate {
                    stroke-width: 2px;
                    stroke-opacity: 0.34;
                }
                .wsm-graph-edge.primary.candidate { stroke-width: 5.2; stroke-opacity: 0.94; }
                .wsm-graph-edge.active {
                    stroke-width: 4.8;
                    stroke-opacity: 1;
                    filter: drop-shadow(0 0 3px color-mix(in srgb, var(--edge-color) 55%, transparent));
                    animation: wsm-edge-pulse 1.15s ease-in-out infinite;
                }
                .wsm-graph-node { cursor: default; }
                .wsm-graph-node .node-shadow {
                    fill: #101828;
                    fill-opacity: 0.1;
                    stroke: none;
                    pointer-events: none;
                }
                .wsm-graph-node .node-disc {
                    fill: #ffffff;
                    stroke: #2f6db3;
                    stroke-width: 1.35;
                    vector-effect: non-scaling-stroke;
                    transition: fill 170ms ease, stroke 170ms ease, stroke-width 170ms ease, transform 170ms ease;
                }
                .wsm-graph-node.sparse .node-disc {
                    fill: #f8fafc;
                    stroke: #d9902f;
                }
                .wsm-graph-node.rgb .node-disc {
                    fill: #eaf2ff;
                    stroke: #2f6db3;
                    stroke-width: 2.1;
                }
                .wsm-graph-node text {
                    fill: #172230;
                    font-size: 10.8px;
                    font-weight: 620;
                    pointer-events: none;
                }
                .wsm-graph-node.sparse text { font-size: 10px; }
                .wsm-graph-node.rgb text { font-size: 21px; font-weight: 760; }
                .wsm-graph-node:hover .node-disc {
                    stroke-width: 2.2;
                    transform: scale(1.055);
                }
                .wsm-graph-node.input-active .node-disc {
                    fill: #e8f2ff;
                    stroke: #2878c7;
                    stroke-width: 3;
                }
                .wsm-graph-node.target-active .node-disc {
                    fill: #fff1df;
                    stroke: #e57f16;
                    stroke-width: 3.2;
                }
                .wsm-graph-text-node .text-hitbox {
                    fill: transparent;
                    stroke: none;
                    pointer-events: all;
                }
                .wsm-graph-text-node text {
                    fill: #000000;
                    font-size: 22px;
                    font-weight: 400;
                    pointer-events: none;
                    transition: fill 170ms ease, font-weight 170ms ease, filter 170ms ease;
                }
                .wsm-graph-text-node.input-active text {
                    fill: #2878c7;
                    font-weight: 600;
                }
                .wsm-graph-node.task-active .node-disc,
                .wsm-graph-text-node.task-active text {
                    animation: wsm-node-pulse 1.15s ease-in-out infinite;
                }
                .wsm-graph-edge-hit {
                    fill: none;
                    stroke: transparent;
                    stroke-width: 14;
                    pointer-events: stroke;
                    cursor: pointer;
                }
                .wsm-graph-edge-hit:hover + .wsm-graph-edge.chain {
                    stroke-width: 3.2;
                    stroke-opacity: 0.82;
                    filter: drop-shadow(0 0 2px color-mix(in srgb, var(--edge-color) 50%, transparent));
                }
                .wsm-graph-edge:is([data-graph-task], [data-graph-composite-task]) {
                    pointer-events: stroke;
                    cursor: pointer;
                }
                .wsm-graph-edge:is([data-graph-task], [data-graph-composite-task]):hover {
                    stroke-width: 3.2;
                    stroke-opacity: 0.82;
                    filter: drop-shadow(0 0 2px color-mix(in srgb, var(--edge-color) 50%, transparent));
                }
                .wsm-graph-node { cursor: grab; outline: none; touch-action: none; user-select: none; }
                .wsm-graph-node.dragging,
                .wsm-graph-viewport.dragging-node .wsm-graph-node { cursor: grabbing; }
                .wsm-graph-node.dragging .node-disc { stroke-width: 3px; filter: drop-shadow(0 5px 7px rgba(31,47,68,.26)); }
                .wsm-graph-text-node { cursor: pointer; outline: none; }
                .wsm-graph-node:focus .node-disc,
                .wsm-graph-text-node:focus text {
                    fill: #0b6dcc;
                    text-decoration: underline;
                }
                .wsm-graph-inside-legend { pointer-events: none; }
                .wsm-graph-legend-panel rect {
                    fill: #ffffff;
                    fill-opacity: 0.9;
                    stroke: #dce2e9;
                    stroke-width: 1;
                }
                .wsm-graph-legend-line {
                    stroke: var(--legend-color);
                    stroke-width: 2.8;
                    stroke-linecap: round;
                }
                .wsm-graph-legend-line.primary { stroke-width: 4.4; }
                .wsm-graph-legend-line.chain { stroke-dasharray: 9 7; }
                .wsm-graph-legend-line.sparse-task { stroke-dasharray: 5 4; }
                .wsm-graph-legend-text {
                    fill: #435264;
                    font-size: 11px;
                }
                .wsm-graph-empty { padding: 48px; color: #7b8795; text-align: center; }
                @keyframes wsm-edge-pulse {
                    0%, 100% { stroke-opacity: 0.58; }
                    50% { stroke-opacity: 1; }
                }
                @keyframes wsm-node-pulse {
                    0%, 100% { filter: drop-shadow(0 0 1px rgba(39, 104, 178, 0.2)); }
                    50% { filter: drop-shadow(0 0 8px rgba(39, 104, 178, 0.86)); }
                }
                @media (prefers-reduced-motion: reduce) {
                    .wsm-graph-edge.active,
                    .wsm-graph-node.task-active .node-disc,
                    .wsm-graph-text-node.task-active text { animation: none; }
                }
                @media (max-width: 1100px) {
                    .wsm-graph-row { width: 100%; }
                    .wsm-graph-svg { min-width: 900px; }
                    .wsm-upload-list { grid-template-columns: repeat(2, minmax(0, 1fr)); }
                }
                @media (max-width: 680px) {
                    .wsm-upload-list { grid-template-columns: minmax(0, 1fr); }
                }
                @media (max-width: 900px) {
                    .wsm-shell { padding: 12px; }
                    #wsm-v1-inference .wsm-workspace { grid-template-columns: minmax(0, 1fr); }
                    #wsm-v1-inference .wsm-pane + .wsm-pane {
                        border-top: 1px solid #d8dee6;
                        border-left: 0;
                    }
                    .wsm-runbar { grid-template-columns: 1fr 1fr; }
                    .wsm-output-selector-grid { grid-template-columns: minmax(0, 1fr); }
                    .wsm-graph-head { display: block; }
                    .wsm-graph-actions { margin-top: 9px; }
                }
                /* ControlWorld dark-theme bridge. */
                body[data-theme="dark"] #wsm-v1-inference .wsm-shell { background:#0d1218; color:#dbe3ec; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-subtitle,
                body[data-theme="dark"] #wsm-v1-inference .wsm-status,
                body[data-theme="dark"] #wsm-v1-inference .wsm-step,
                body[data-theme="dark"] #wsm-v1-inference .wsm-hint { color:#95a4b4; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-input-validation[data-severity="checking"] { background:#1a222b; border-color:#44515f; color:#aab6c2; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-input-validation[data-severity="green"] { background:#14281c; border-color:#396a49; color:#91d5a6; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-input-validation[data-severity="orange"] { background:#2d2414; border-color:#80662f; color:#e8c475; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-input-validation[data-severity="red"] { background:#32191b; border-color:#804247; color:#eda0a4; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-demo-toggle { background:#19222c; border-color:#516171; color:#cbd6e1; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-demo-toggle.active { background:#17372b; border-color:#4da47e; color:#8bd8b5; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-demo-bar { background:#12271f; border-color:#356d56; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-demo-copy strong,
                body[data-theme="dark"] #wsm-v1-inference .wsm-demo-lock { color:#75d0a8; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-demo-copy span { color:#9cbcad; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-demo-select,
                body[data-theme="dark"] #wsm-v1-inference .wsm-demo-search { background:#19222c; border-color:#4f7564; color:#e4eee9; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-upload.demo-locked { background:#14251e; border-color:#3d705a; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-upload.demo-locked .wsm-dropzone { background:#101e18; border-color:#4c8069; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-variable-file { background:#111a23; border-color:#405264; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-variable-name { color:#dbe3ec; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-variable-role,
                body[data-theme="dark"] #wsm-v1-inference .wsm-preview-caption { color:#96a6b6; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-demo-input-toggle { color:#b9c7d5; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-latent-source-toggle { border-color:#566574; background:#26313c; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-latent-source-toggle input:not(:checked) + .wsm-latent-source-option,
                body[data-theme="dark"] #wsm-v1-inference .wsm-latent-source-toggle input:checked + .wsm-latent-source-option + .wsm-latent-source-option { background:#17222d; color:#8fc1ff; box-shadow:none; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-self-preprocessor-track { border-color:#667482; background:#3b4651; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-self-preprocessor-toggle input:checked + .wsm-self-preprocessor-track { border-color:#5fb892; background:#347f62; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-research-note,
                body[data-theme="dark"] #wsm-v1-inference .wsm-research-note strong { color:#66c89c; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-workspace { background:#141b23; border-color:#344150; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-pane + .wsm-pane { border-color:#344150; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-field label,
                body[data-theme="dark"] #wsm-v1-inference .wsm-upload-title,
                body[data-theme="dark"] #wsm-v1-inference .wsm-dropzone-copy strong { color:#dbe3ec; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-input,
                body[data-theme="dark"] #wsm-v1-inference .wsm-select,
                body[data-theme="dark"] #wsm-v1-inference .wsm-textarea,
                body[data-theme="dark"] #wsm-v1-inference .wsm-signature-trigger,
                body[data-theme="dark"] #wsm-v1-inference .wsm-signature-panel { background:#19222c; border-color:#435160; color:#e4ebf3; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-upload,
                body[data-theme="dark"] #wsm-v1-inference .wsm-signature,
                body[data-theme="dark"] #wsm-v1-inference .wsm-task { background:#151d26; border-color:#344150; color:#dbe3ec; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-dropzone { background:#111820; border-color:#526170; color:#9cabb9; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-dropzone:hover,
                body[data-theme="dark"] #wsm-v1-inference .wsm-dropzone.dragging,
                body[data-theme="dark"] #wsm-v1-inference .wsm-signature.active,
                body[data-theme="dark"] #wsm-v1-inference .wsm-task.active { background:#17304a; border-color:#58a8ff; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-frame-guide { background:#162636; border-left-color:#58a8ff; color:#b8c7d7; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-frame-guide strong,
                body[data-theme="dark"] #wsm-v1-inference .wsm-encoding-title { color:#f0f4f8; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-task-target,
                body[data-theme="dark"] #wsm-v1-inference .wsm-encoding-kicker,
                body[data-theme="dark"] #wsm-v1-inference .wsm-encoding-scale,
                body[data-theme="dark"] #wsm-v1-inference .wsm-encoding-note,
                body[data-theme="dark"] #wsm-v1-inference .wsm-encoding-role,
                body[data-theme="dark"] #wsm-v1-inference .wsm-encoding-swatch { color:#c5d0dc; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-task-type { color:#8998a8; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-encoding-panel,
                body[data-theme="dark"] #wsm-v1-inference .wsm-output-list { background:#111820; border-color:#566372; color:#e3eaf2; box-shadow:none; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-encoding-rule { background:#465361; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-output-item { background:#17202a; border-color:#344150; color:#d0dae5; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-graph-row { background:linear-gradient(180deg,#121a23,#0f151d); border-color:#344150; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-graph-title,
                body[data-theme="dark"] #wsm-v1-inference .wsm-graph-space-label,
                body[data-theme="dark"] #wsm-v1-inference .wsm-graph-audio-label { color:#e5edf5; fill:#e5edf5; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-graph-summary { color:#9baaba; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-graph-actions { background:#111820; border-color:#344150; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-graph-mode { color:#9dabb9; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-graph-mode.active { background:#22303e; color:#80baff; box-shadow:none; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-graph-viewport { border-color:#344150; background:radial-gradient(circle at 50% 44%,rgba(34,72,111,.25),transparent 34%),linear-gradient(180deg,#101720,#0d131a); box-shadow:none; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-graph-plate.inner { fill:#151e28; stroke:#4a5867; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-graph-plate.outer { stroke:#596675; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-graph-node .node-disc { fill:#17222d; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-graph-node.rgb .node-disc { fill:#17304a; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-graph-node.sparse .node-disc { fill:#2a251a; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-graph-node text,
                body[data-theme="dark"] #wsm-v1-inference .wsm-graph-text-node text { fill:#e1e9f1; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-graph-legend-panel rect { fill:#121a23; stroke:#3b4856; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-graph-legend-text { fill:#bdc9d5; }
                body[data-theme="dark"] #wsm-v1-inference .wsm-empty { color:#8d9baa; }
            </style>
            <main class="wsm-shell" lang="en">
                <header class="wsm-topbar">
                    <div>
                        <h2 class="wsm-title">WSM V1 — Augustus</h2>
                        <p class="wsm-subtitle">Choose an input signature, then select an available trained output head.</p>
                    </div>
                    <div class="wsm-topbar-actions">
                        <input class="wsm-demo-search wsm-head-search" data-role="head-search" type="search"
                               inputmode="numeric" placeholder="Find Head 38 / 000038"
                               aria-label="Find any trained Head by number">
                        <button class="wsm-demo-toggle" data-role="demo-toggle" type="button" hidden>Paper Demo</button>
                        <div class="wsm-status" data-role="catalog-status">Scanning Channel Heads...</div>
                    </div>
                    <div class="wsm-demo-bar" data-role="demo-bar" hidden>
                        <div class="wsm-demo-copy">
                            <strong>You are currently in Paper Demo mode · Figures 14–39</strong>
                            <span>Each example loads the scoped raw WSG/NPZ variables used by the selected Head and encodes them again at run time. Reader images are display-only. Exit Paper Demo to use your own inputs.</span>
                        </div>
                        <div class="wsm-demo-controls">
                            <select class="wsm-demo-select" data-role="demo-record" aria-label="Paper qualitative example"></select>
                            <input class="wsm-demo-search" data-role="demo-head-search" type="search"
                                   inputmode="numeric" placeholder="Head 38 / 000038"
                                   aria-label="Find paper example by Head number">
                        </div>
                    </div>
                    <p class="wsm-research-note"><strong>Note:</strong> Each task-specific head was fine-tuned on only a few hundred samples for about 3–5 hours. These heads should be treated as zero-/few-shot models. They may fail on unseen samples or unfamiliar data distributions. For example, some density-estimation heads were trained only on a limited subset of OBJ data and do not yet work on real-world inputs. This is an academic proof-of-concept project.</p>
                </header>
                <section class="wsm-workspace">
                    <div class="wsm-pane">
                        <div class="wsm-pane-head"><h3>Input</h3><span class="wsm-step">02</span></div>
                        <div class="wsm-field">
                            <label>Input configuration</label>
                            <div class="wsm-signature-picker" data-role="signature-picker">
                                <button class="wsm-signature-trigger" data-role="signature-trigger" type="button">
                                    <span data-role="signature-current">Select an input configuration</span>
                                    <span class="wsm-signature-arrow">▼</span>
                                </button>
                                <div class="wsm-signature-panel">
                                    <input class="wsm-input" data-role="signature-search" placeholder="Filter input modalities">
                                    <div class="wsm-signature-list" data-role="signature-list"></div>
                                </div>
                            </div>
                            <div class="wsm-frame-guide" data-role="frame-guide" hidden></div>
                        </div>
                        <div class="wsm-upload-list" data-role="uploads"></div>
                        <div data-role="text-input-pool" hidden>
                            <textarea class="wsm-textarea" data-role="modality-prompt" readonly
                                      placeholder="Select an output Head to inspect its automatic task semantics."></textarea>
                            <textarea class="wsm-textarea" data-role="prompt" placeholder="Describe the requested output"></textarea>
                        </div>
                    </div>
                    <div class="wsm-pane">
                        <div class="wsm-pane-head"><h3>Output</h3><span class="wsm-step">03</span></div>
                        <div class="wsm-output-selector-grid">
                            <div class="wsm-output-controls">
                        <div class="wsm-field">
                            <label>Available output</label>
                            <label class="wsm-model-filter">
                                <input type="checkbox" data-role="trained-only" checked>
                                Only show outputs with trained models
                            </label>
                            <input class="wsm-input" data-role="task-search" placeholder="Filter by channel, modality, or description">
                            <div class="wsm-task-list" data-role="task-list"></div>
                        </div>
                        <div class="wsm-field">
                            <label>Checkpoint</label>
                            <select class="wsm-select" data-role="checkpoint"></select>
                            <div class="wsm-hint" data-role="task-description">Select an output Channel Head.</div>
                        </div>
                            </div>
                            <aside class="wsm-encoding-panel" data-role="output-encoding-legend" aria-live="polite"></aside>
                        </div>
                        <div class="wsm-output-stage">
                            <div class="wsm-empty" data-role="empty">The generated output will appear here.</div>
                            <img class="wsm-result" data-role="result" alt="WSM inference result" hidden>
                        </div>
                        <div class="wsm-output-list" data-role="output-list" hidden></div>
                        <div class="wsm-runbar">
                            <div class="wsm-field"><label>Steps</label><input class="wsm-input" data-role="steps" type="number" min="1" max="50" value="10"></div>
                            <div class="wsm-field"><label>Size</label><select class="wsm-select" data-role="size"><option>256</option><option>480</option><option>512</option><option>768</option></select></div>
                            <div class="wsm-field"><label>Seed</label><input class="wsm-input" data-role="seed" type="number" value="20260812"></div>
                            <button class="wsm-button" data-role="run" disabled>Run inference</button>
                        </div>
                        <section class="wsm-log-panel" data-role="inference-log-panel" data-status="idle" hidden>
                            <div class="wsm-log-head">
                                <span>Inference log</span>
                                <span class="wsm-log-status" data-role="inference-log-status">Idle</span>
                            </div>
                            <pre class="wsm-log-output" data-role="inference-log"></pre>
                        </section>
                        <div class="wsm-error" data-role="error"></div>
                    </div>
                    <div class="wsm-graph-row">
                        <div class="wsm-graph-head">
                            <div>
                                <h3 class="wsm-graph-title">WSG Task Graph <span class="wsm-step">01</span></h3>
                                <div class="wsm-graph-summary" data-role="graph-summary">Loading the WSG task graph...</div>
                            </div>
                            <div class="wsm-graph-actions" aria-label="Task graph selection mode">
                                <button class="wsm-graph-mode active" data-role="graph-input-mode" type="button" aria-pressed="true">1 · Select input</button>
                                <button class="wsm-graph-mode" data-role="graph-output-mode" type="button" aria-pressed="false">2 · Select output</button>
                                <button class="wsm-graph-mode" data-role="graph-reset-layout" type="button" title="Restore the original node arrangement">Reset view</button>
                            </div>
                        </div>
                        <div class="wsm-graph-canvas" data-role="graph-canvas">
                            <div class="wsm-graph-viewport" data-role="task-graph"></div>
                            <button class="wsm-canvas-handle top" data-canvas-resize="top" type="button" aria-label="Resize graph canvas height from top"></button>
                            <button class="wsm-canvas-handle right" data-canvas-resize="right" type="button" aria-label="Resize graph canvas width from right"></button>
                            <button class="wsm-canvas-handle bottom" data-canvas-resize="bottom" type="button" aria-label="Resize graph canvas height from bottom"></button>
                            <button class="wsm-canvas-handle left" data-canvas-resize="left" type="button" aria-label="Resize graph canvas width from left"></button>
                            <button class="wsm-canvas-handle top-left corner" data-canvas-resize="top-left" type="button" aria-label="Scale graph canvas from top left"></button>
                            <button class="wsm-canvas-handle top-right corner" data-canvas-resize="top-right" type="button" aria-label="Scale graph canvas from top right"></button>
                            <button class="wsm-canvas-handle bottom-right corner" data-canvas-resize="bottom-right" type="button" aria-label="Scale graph canvas from bottom right"></button>
                            <button class="wsm-canvas-handle bottom-left corner" data-canvas-resize="bottom-left" type="button" aria-label="Scale graph canvas from bottom left"></button>
                        </div>
                    </div>
                </section>
            </main>
            <div class="wsm-lightbox" data-role="image-lightbox" hidden role="dialog" aria-modal="true" aria-label="Input image preview">
                <button class="wsm-lightbox-close" data-role="image-lightbox-close" type="button" aria-label="Close preview">×</button>
                <img data-role="image-lightbox-image" alt="Full-size input preview">
            </div>
            <div class="wsm-quality-modal-backdrop" data-role="quality-warning-modal" hidden>
                <section class="wsm-quality-modal" role="dialog" aria-modal="true"
                         aria-labelledby="wsm-quality-warning-title">
                    <div class="wsm-quality-modal-head">
                        <h3 class="wsm-quality-modal-title" id="wsm-quality-warning-title"
                            data-role="quality-warning-title">Further improvement recommended</h3>
                        <button class="wsm-quality-modal-close" data-role="quality-warning-close"
                                type="button" aria-label="Close warning">×</button>
                    </div>
                    <div class="wsm-quality-modal-body">
                        <div class="wsm-quality-modal-head-id" data-role="quality-warning-head"></div>
                        <p class="wsm-quality-modal-message" data-role="quality-warning-message"></p>
                    </div>
                    <div class="wsm-quality-modal-actions">
                        <button class="wsm-quality-modal-continue" data-role="quality-warning-continue"
                                type="button">Continue with this Head</button>
                    </div>
                </section>
            </div>
        `;
        this.taskGraph = new WSMTaskGraph(
            this.$('task-graph'),
            this.$('graph-summary'),
            selection => this.handleGraphSelection(selection),
        );
        this.bindEvents();
        await this.loadCatalog();
    }

    $(role) {
        return this.container.querySelector(`[data-role="${role}"]`);
    }

    bindEvents() {
        this.$('demo-toggle').addEventListener('click', () => this.setDemoMode(!this.demoEnabled));
        this.$('head-search').addEventListener('input', () => this.handleHeadSearch());
        this.$('head-search').addEventListener('keydown', event => {
            if (event.key === 'Enter') this.handleHeadSearch(true);
        });
        this.$('demo-record').addEventListener('change', event => this.activateDemoRecord(event.target.value));
        this.$('demo-head-search').addEventListener('input', () => this.handleDemoHeadSearch());
        this.$('image-lightbox').addEventListener('click', event => {
            if (event.target === this.$('image-lightbox') || event.target === this.$('image-lightbox-close')) this.closeImageLightbox();
        });
        this.$('signature-search').addEventListener('input', () => this.renderSignatures());
        this.$('signature-trigger').addEventListener('click', () => {
            this.$('signature-picker').classList.toggle('open');
            if (this.$('signature-picker').classList.contains('open')) this.$('signature-search').focus();
        });
        this.container.addEventListener('click', (event) => {
            if (!event.target.closest('[data-role="signature-picker"]')) {
                this.$('signature-picker').classList.remove('open');
            }
        });
        this.$('task-search').addEventListener('input', () => this.renderTasks());
        this.$("trained-only").addEventListener("change", () => {
            this.task = null;
            this.compositeRoute = null;
            this.renderTasks();
            this.renderCheckpoint();
        });
        this.$('graph-input-mode').addEventListener('click', () => this.setGraphSelectionMode('input'));
        this.$('graph-output-mode').addEventListener('click', () => this.setGraphSelectionMode('output'));
        this.$('graph-reset-layout').addEventListener('click', () => this.resetGraphView());
        this.container.querySelectorAll('[data-canvas-resize]').forEach(handle => {
            handle.addEventListener('pointerdown', event => this.beginGraphCanvasResize(event));
        });
        this.container.addEventListener('pointermove', event => this.resizeGraphCanvas(event));
        this.container.addEventListener('pointerup', event => this.endGraphCanvasResize(event));
        this.container.addEventListener('pointercancel', event => this.endGraphCanvasResize(event));
        this.$('prompt').addEventListener('input', () => this.updateRunState());
        this.$('checkpoint').addEventListener('change', () => {
            this.showQualityWarning(this.task);
        });
        this.$('run').addEventListener('click', () => this.runInference());
        this.$('quality-warning-close').addEventListener('click', () => {
            this.closeQualityWarning();
        });
        this.$('quality-warning-continue').addEventListener('click', () => {
            this.closeQualityWarning();
        });
        this.$('quality-warning-modal').addEventListener('click', event => {
            if (event.target === this.$('quality-warning-modal')) {
                this.closeQualityWarning();
            }
        });
        this.container.addEventListener('keydown', event => {
            if (event.key === 'Escape') {
                this.closeQualityWarning();
                this.closeImageLightbox();
            }
        });
    }

    async waitForWebSocket(timeoutMs = 10000) {
        if (window.controlWaveWebSocket?.isConnected) return;
        await new Promise((resolve, reject) => {
            const timeout = window.setTimeout(() => {
                window.removeEventListener('websocket-connected', handleConnected);
                reject(new Error('ControlWorld WebSocket connection timed out.'));
            }, timeoutMs);
            const handleConnected = () => {
                window.clearTimeout(timeout);
                resolve();
            };
            window.addEventListener('websocket-connected', handleConnected, { once: true });
            if (window.controlWaveWebSocket?.isConnected) handleConnected();
        });
    }

    async sendRequest(payload) {
        await this.waitForWebSocket();
        const response = await window.controlWaveWebSocket.sendRequest(this.id, payload);
        if (response?.error) throw new Error(response.error);
        return response;
    }


    renderInputValidationStatus(element, report) {
        if (!element) return;
        const severity = report?.severity || "red";
        element.hidden = false;
        element.dataset.severity = severity;
        const labels = {
            green: "Compatible",
            orange: "Compatible with notice",
            red: report?.runnable === false ? "Cannot map for inference" : "Strong warning",
            checking: "Checking input",
        };
        const messages = severity === "red"
            ? (report?.red_reasons || [])
            : (report?.warnings || []);
        const summary = report?.summary ? " · " + report.summary : "";
        element.textContent = (labels[severity] || labels.red) + summary
            + (messages.length ? "\n" + messages.join("\n") : "")
            + (severity === "red"
                ? (report?.runnable === false
                    ? "\nThe upload is retained, but this file cannot be mapped for inference."
                    : "\nThe upload is retained; you may continue if this representation is intentional.")
                : "");
    }

    async validateInputUpload(token, dataUrl, filename, statusElement, generatePreview = true, options = {}) {
        const taskId = this.task?.id;
        if (!taskId) return;
        const checking = {
            severity: "checking",
            summary: "Reading format, channels, layout, and value ranges",
            runnable: false,
        };
        this.inputValidation.set(token, checking);
        this.renderInputValidationStatus(statusElement, checking);
        this.updateRunState();
        let report;
        try {
            const checkpoint = this.container.querySelector("[data-role=checkpoint]")?.value || "";
            report = await this.sendRequest({
                action: "validate_input",
                data: {
                    task_id: taskId, checkpoint, token, data_url: dataUrl,
                    filename: filename || "", generate_preview: generatePreview,
                    paper_demo: Boolean(options.paperDemo),
                    latent_asset: String(options.latentAsset || ""),
                    self_preprocess: Boolean(options.selfPreprocess),
                },
            });
        } catch (error) {
            report = {
                severity: "red",
                runnable: false,
                summary: "Validation service could not inspect this input",
                red_reasons: [error.message || String(error)],
                warnings: [],
            };
        }
        if (this.task?.id !== taskId || this.uploads.get(token) !== dataUrl) return null;
        this.inputValidation.set(token, report);
        this.renderInputValidationStatus(statusElement, report);
        this.updateRunState();
        return report;
    }

    async loadDemoManifest() {
        try {
            const url = new URL("./demo_assets/manifest.json", import.meta.url);
            const response = await fetch(url, {cache: "no-cache"});
            if (!response.ok) throw new Error("HTTP " + response.status);
            const manifest = await response.json();
            const taskIds = new Set((this.fullCatalog?.tasks || []).map(task => task.id));
            manifest.records = (manifest.records || []).filter(record => taskIds.has(record.task_id));
            manifest.record_count = manifest.records.length;
            manifest.task_count = new Set(manifest.records.map(record => record.task_id)).size;
            this.demoManifest = manifest;
            this.demoManifestUrl = url;
            this.container.querySelector("[data-role=demo-toggle]").hidden = !manifest.records.length;
        } catch (error) {
            this.demoManifest = null;
            this.demoManifestUrl = null;
            this.container.querySelector("[data-role=demo-toggle]").hidden = true;
            console.warn("WSM Paper Demo manifest is unavailable:", error);
        }
    }

    buildDemoCatalog() {
        if (!this.fullCatalog || !this.demoManifest) return this.fullCatalog;
        const allowed = new Set(this.demoManifest.records.map(record => record.task_id));
        const tasks = this.fullCatalog.tasks.filter(task => allowed.has(task.id));
        const signatures = new Set(tasks.map(task => task.input_signature));
        const inputSignatures = this.fullCatalog.input_signatures
            .filter(signature => signatures.has(signature.id))
            .map(signature => {
                const members = tasks.filter(task => task.input_signature === signature.id);
                return {
                    ...signature,
                    configured_count: members.length,
                    available_count: members.filter(task => task.trained).length,
                };
            });
        return {
            ...this.fullCatalog,
            tasks,
            input_signatures: inputSignatures,
            counts: {
                ...this.fullCatalog.counts,
                configured: tasks.length,
                trained: tasks.filter(task => task.trained).length,
                inference_available: tasks.filter(task => task.supported).length,
            },
        };
    }

    refreshCatalogStatus() {
        if (!this.fullCatalog) return;
        const counts = this.fullCatalog.counts;
        if (this.demoEnabled && this.demoManifest) {
            this.$('catalog-status').textContent =
                this.demoManifest.record_count + " paper examples · "
                + this.demoManifest.task_count + " qualitative Heads";
        } else {
            this.$('catalog-status').textContent =
                counts.trained + " trained heads (" + (counts.custom_trained || 0) + " custom, " + counts.sparse_trained + " sparse) · "
                + counts.inference_available + " WebUI-ready · " + counts.configured + " configured · "
                + (counts.quality_needs_improvement || 0) + " flagged for improvement";
        }
        const toggle = this.$('demo-toggle');
        toggle.classList.toggle("active", this.demoEnabled);
        toggle.setAttribute("aria-pressed", String(this.demoEnabled));
        toggle.textContent = this.demoEnabled ? "Exit Paper Demo" : "Paper Demo";
        this.$('demo-bar').hidden = !this.demoEnabled;
        this.$('head-search').hidden = this.demoEnabled;
    }

    handleHeadSearch(reportMissing = false) {
        if (this.demoEnabled || !this.fullCatalog) return;
        const query = this.$('head-search').value.trim();
        const matches = query.match(/\d+/g);
        if (!matches?.length) return;
        const headIndex = Number(matches[matches.length - 1]);
        const sourceOrder = {dense: 0, custom: 1, sparse: 2};
        const candidates = this.fullCatalog.tasks.filter(task =>
            Number(task.index) === headIndex
        ).sort((left, right) =>
            Number(right.supported) - Number(left.supported)
            || Number(right.official_recommended) - Number(left.official_recommended)
            || (sourceOrder[left.source] ?? 9) - (sourceOrder[right.source] ?? 9)
        );
        const task = candidates[0];
        if (!task) {
            if (reportMissing) this.showError(`Head ${headIndex} is not registered.`);
            return;
        }
        this.selectTaskFromGraph(task.id);
    }

    demoHeadSearchRecords() {
        const records = this.demoManifest?.records || [];
        const search = this.container.querySelector("[data-role=\"demo-head-search\"]");
        const query = search?.value.trim() || "";
        if (!query) return records;
        const matches = query.match(/\d+/g);
        if (!matches?.length) return [];
        const headIndex = Number(matches[matches.length - 1]);
        return records.filter(record => Number(record.head_index) === headIndex);
    }

    renderDemoSelector() {
        const select = this.$("demo-record");
        select.innerHTML = "";
        if (!this.demoManifest) return;
        const records = this.demoHeadSearchRecords();
        if (!records.length) {
            const option = document.createElement("option");
            option.textContent = "No paper example matches this Head";
            option.disabled = true;
            option.selected = true;
            select.appendChild(option);
            return;
        }
        const groups = new Map();
        records.forEach(record => {
            const key = "Part " + record.part + " · " + record.part_title;
            if (!groups.has(key)) groups.set(key, []);
            groups.get(key).push(record);
        });
        groups.forEach((groupRecords, label) => {
            const group = document.createElement("optgroup");
            group.label = label;
            groupRecords.forEach(record => {
                const option = document.createElement("option");
                option.value = record.id;
                option.textContent = record.label + " · Head " + String(record.head_index).padStart(6, "0");
                group.appendChild(option);
            });
            select.appendChild(group);
        });
        if (this.activeDemoRecord && records.some(record => record.id === this.activeDemoRecord.id)) {
            select.value = this.activeDemoRecord.id;
        }
    }

    async handleDemoHeadSearch() {
        if (!this.demoEnabled) return;
        const records = this.demoHeadSearchRecords();
        this.renderDemoSelector();
        if (!records.length) return;
        const activeStillVisible = records.some(record => record.id === this.activeDemoRecord?.id);
        if (!activeStillVisible) await this.activateDemoRecord(records[0].id);
    }

    demoRecordsForTask(taskId) {
        return (this.demoManifest?.records || []).filter(record => record.task_id === taskId);
    }

    async setDemoMode(enabled) {
        if (enabled && !this.demoManifest?.records?.length) return;
        this.demoEnabled = Boolean(enabled);
        this.catalog = this.demoEnabled ? this.buildDemoCatalog() : this.fullCatalog;
        this.signature = null;
        this.task = null;
        this.compositeRoute = null;
        this.activeDemoRecord = null;
        this.uploads.clear();
        this.uploadFilenames.clear();
        this.inputValidation.clear();
        this.demoLatentInputs.clear();
        this.selfPreprocessorInputs.clear();
        this.$('signature-search').value = "";
        this.$('task-search').value = "";
        this.$('demo-head-search').value = "";
        this.$('trained-only').checked = true;
        this.refreshCatalogStatus();
        this.renderDemoSelector();
        if (this.demoEnabled) {
            await this.activateDemoRecord(this.demoManifest.records[0].id);
        } else {
            this.$('prompt').readOnly = false;
            this.$('prompt').value = "";
            this.renderSignatures();
            const first = this.catalog.input_signatures[0];
            if (first) this.selectSignature(first.id);
        }
    }

    async activateDemoRecord(recordId) {
        if (!this.demoEnabled || !this.demoManifest) return;
        const record = this.demoManifest.records.find(item => item.id === recordId);
        const task = this.catalog.tasks.find(item => item.id === record?.task_id);
        const signature = this.catalog.input_signatures.find(item => item.id === task?.input_signature);
        if (!record || !task || !signature) {
            this.showError("This paper example is not available in the current task registry.");
            return;
        }
        const headSearch = this.$("demo-head-search");
        if (headSearch?.value.trim()) {
            const selectedIsVisible = this.demoHeadSearchRecords().some(item => item.id === record.id);
            if (!selectedIsVisible) headSearch.value = "";
        }
        this.activeDemoRecord = record;
        this.compositeRoute = null;
        this.signature = signature;
        this.task = task;
        this.uploads.clear();
        this.uploadFilenames.clear();
        this.inputValidation.clear();
        this.demoLatentInputs.clear();
        this.selfPreprocessorInputs.clear();
        this.$('signature-search').value = "";
        this.$('task-search').value = "";
        this.$('signature-picker').classList.remove("open");
        this.$('steps').value = record.steps;
        this.$('size').value = record.size;
        this.$('seed').value = record.seed;
        this.$('prompt').value = record.prompt || "";
        this.$('prompt').readOnly = false;
        this.renderDemoSelector();
        this.renderSignatures();
        this.renderUploads();
        this.renderTasks(false);
        this.renderCheckpoint();
        this.setGraphSelectionMode("output");
        this.showQualityWarning(task);
    }

    demoAssetUrl(asset) {
        return new URL(asset, this.demoManifestUrl).href;
    }

    async demoDataUrl(assetUrl) {
        if (!this.demoAssetCache.has(assetUrl)) {
            this.demoAssetCache.set(assetUrl, fetch(assetUrl).then(response => {
                if (!response.ok) throw new Error("Demo asset failed to load: HTTP " + response.status);
                return response.blob();
            }).then(blob => this.readFile(blob)));
        }
        return this.demoAssetCache.get(assetUrl);
    }

    openImageLightbox(source, label) {
        this.$('image-lightbox-image').src = source;
        this.$('image-lightbox-image').alt = label || "Full-size input preview";
        this.$('image-lightbox').hidden = false;
    }

    closeImageLightbox() {
        const lightbox = this.$('image-lightbox');
        if (lightbox) lightbox.hidden = true;
    }

    async loadCatalog(refresh = false) {
        try {
            this.fullCatalog = await this.sendRequest({ action: 'get_catalog', data: { refresh } });
            this.catalog = this.fullCatalog;
            await this.loadDemoManifest();
            if (this.demoManifest?.records?.length) {
                await this.setDemoMode(true);
                return;
            }
            this.refreshCatalogStatus();
            this.renderSignatures();
            const first = this.catalog.input_signatures[0];
            if (first) {
                this.selectSignature(first.id);
            }
        } catch (error) {
            this.showError(error.message);
            this.$('catalog-status').textContent = 'Catalog unavailable';
        }
    }

    renderSignatures() {
        if (!this.catalog) return;
        this.$('signature-current').textContent =
            this.signature?.label || 'Select an input configuration';
        const query = this.$('signature-search').value.trim().toLowerCase();
        const options = this.catalog.input_signatures
            .filter(item => item.label.toLowerCase().includes(query))
            .sort((left, right) => left.label.localeCompare(right.label, undefined, { numeric: true }));
        const holder = this.$('signature-list');
        holder.innerHTML = options.length ? options.map(item => `
            <button class="wsm-signature ${this.signature?.id === item.id ? 'active' : ''}"
                    data-signature="${this.escape(item.id)}">
                <div class="wsm-signature-label">${this.escape(item.label)}</div>
                <div class="wsm-signature-count">
                    ${item.available_count} trained · ${item.configured_count} configured output(s)
                </div>
            </button>
        `).join('') : '<div class="wsm-empty">No input configuration matches this filter.</div>';
        holder.querySelectorAll('[data-signature]').forEach(button => {
            button.addEventListener('click', () => this.selectSignature(button.dataset.signature));
        });
    }

    selectSignature(id, autoSelectTask = true) {
        if (this.demoEnabled && autoSelectTask) {
            const demoTask = this.catalog.tasks.find(task => task.input_signature === id);
            const demoRecord = demoTask ? this.demoRecordsForTask(demoTask.id)[0] : null;
            if (demoRecord) {
                this.activateDemoRecord(demoRecord.id);
                return;
            }
        }
        const nextSignature = this.catalog.input_signatures.find(item => item.id === id) || null;
        const changed = this.signature?.id !== nextSignature?.id;
        this.signature = nextSignature;
        this.task = null;
        this.compositeRoute = null;
        if (changed) {
            this.uploads.clear();
            this.uploadFilenames.clear();
            this.inputValidation.clear();
            this.demoLatentInputs.clear();
            this.selfPreprocessorInputs.clear();
        }
        this.$('signature-search').value = '';
        this.$('signature-picker').classList.remove('open');
        this.renderSignatures();
        this.renderUploads();
        this.renderTasks(autoSelectTask);
        this.renderCheckpoint();
        this.setGraphSelectionMode('output');
    }

    selfPreprocessorForToken(token) {
        const tasks = this.fullCatalog?.tasks || this.catalog?.tasks || [];
        return tasks.filter(task => {
            const visualInputs = (task.inputs || []).filter(value => value !== 'caption');
            return task.supported
                && task.source === 'dense'
                && task.target === token
                && visualInputs.length === 1
                && visualInputs[0] === '0-2';
        }).sort((left, right) =>
            Number(right.official_recommended) - Number(left.official_recommended)
            || left.index - right.index
        )[0] || null;
    }


    renderUploads() {
        const holder = this.$('uploads');
        const guide = this.$('frame-guide');
        const prompt = this.$('prompt');
        const modalityPrompt = this.$('modality-prompt');
        const textInputPool = this.$('text-input-pool');
        textInputPool.append(modalityPrompt, prompt);
        holder.innerHTML = "";
        guide.hidden = true;
        guide.innerHTML = "";
        if (!this.signature) return;
        const inputs = this.compositeRoute ? ['caption'] : this.signature.inputs;
        const demoRecord = this.demoEnabled && !this.compositeRoute ? this.activeDemoRecord : null;
        const imageDetails = this.compositeRoute ? [] : (this.signature.input_details || [])
            .filter(detail => detail.token !== "caption");
        if (this.compositeRoute) {
            guide.innerHTML = '<strong>Two-stage route</strong><span><b>Stage 1:</b> Text → RGB with the base FLUX.2 model</span><span><b>Stage 2:</b> Generated RGB → ' + this.escape(this.task?.target_description || this.task?.target || 'target modality') + '</span>';
            guide.hidden = false;
        } else if (imageDetails.length) {
            guide.innerHTML = "<strong>Frame routing</strong>" + imageDetails.map(detail =>
                "<span><b>" + this.escape(detail.role_label || "Input") + ":</b> "
                + this.escape(detail.label) + "</span>"
            ).join("");
            guide.hidden = false;
        }
        prompt.readOnly = false;
        if (demoRecord) prompt.value = demoRecord.prompt || "";
        modalityPrompt.value = this.task?.modality_prompt || "";
        inputs.filter(token => token !== "caption").forEach(token => {
            const detail = this.signature.input_details?.find(item => item.token === token);
            const inputKind = detail?.kind || (RGB_INPUT_TOKENS.has(token) ? "image" : "wsg");
            const requiresWsg = inputKind === "wsg";
            const acceptsTensor = requiresWsg || RGB_INPUT_TOKENS.has(token);
            const selfPreprocessor = requiresWsg ? this.selfPreprocessorForToken(token) : null;
            let useSelfPreprocessor = Boolean(
                selfPreprocessor && this.selfPreprocessorInputs.has(token)
            );
            const inputLabel = detail?.label || token;
            const roleLabel = detail?.role_label || "Input image";
            const rawUploadPrompt = requiresWsg
                ? "Drop a WSG, NPZ, or NPY tensor here"
                : (detail?.upload_prompt || "Drop an image or raw WSG tensor here");
            const rawInputHint = requiresWsg
                ? "Upload raw World Snapshot values. The preview is for inspection only; the tensor is normalized and encoded at run time."
                : (detail?.hint || "Upload the RGB image or its raw WSG 0–2 channels.");
            const uploadPrompt = useSelfPreprocessor
                ? "Drop an RGB image for WSM to estimate this modality"
                : rawUploadPrompt;
            const inputHint = useSelfPreprocessor
                ? `WSM ${selfPreprocessor.head} first estimates ${token} from RGB; its generated raw WSG values are then encoded as this condition.`
                : rawInputHint;
            const demoInput = demoRecord?.inputs?.find(item => item.token === token);
            const locked = Boolean(demoInput);

            const row = document.createElement("div");
            row.className = "wsm-upload" + (locked ? " demo-locked" : "");
            const titleRow = document.createElement("div");
            titleRow.className = "wsm-upload-title-row";
            const title = document.createElement("div");
            title.className = "wsm-upload-title";
            title.textContent = inputLabel;
            title.title = inputLabel;
            const role = document.createElement("span");
            role.className = "wsm-role-badge";
            role.textContent = roleLabel;
            const titleActions = document.createElement("div");
            titleActions.className = "wsm-upload-title-actions";
            titleActions.append(role);
            let selfPreprocessorCheckbox = null;
            if (selfPreprocessor && !locked) {
                const toggle = document.createElement("label");
                toggle.className = "wsm-demo-input-toggle wsm-self-preprocessor-toggle";
                toggle.title = "Let WSM estimate this non-RGB modality from an uploaded RGB image, then use the generated WSG values as the condition.";
                selfPreprocessorCheckbox = document.createElement("input");
                selfPreprocessorCheckbox.type = "checkbox";
                selfPreprocessorCheckbox.checked = useSelfPreprocessor;
                selfPreprocessorCheckbox.setAttribute("aria-label", `Use WSM Self-Preprocessor for ${inputLabel}`);
                const track = document.createElement("span");
                track.className = "wsm-self-preprocessor-track";
                track.setAttribute("aria-hidden", "true");
                const label = document.createElement("span");
                label.textContent = "WSM Self-Preprocessor";
                toggle.append(selfPreprocessorCheckbox, track, label);
                titleActions.append(toggle);
            }
            titleRow.append(title, titleActions);

            const dropzone = document.createElement(locked ? "div" : "label");
            dropzone.className = "wsm-dropzone";
            const copy = document.createElement("div");
            copy.className = "wsm-dropzone-copy";
            const copyStrong = document.createElement("strong");
            copyStrong.textContent = uploadPrompt;
            const copySmall = document.createElement("span");
            copySmall.textContent = useSelfPreprocessor
                ? "or click to choose a PNG, JPEG, or WebP file"
                : inputKind === "audio"
                    ? "or click to choose a WAV file"
                    : acceptsTensor
                        ? "choose .wsg, .npz, .npy, PNG, JPEG, or WebP"
                        : "or click to choose a PNG, JPEG, or WebP file";
            copy.append(copyStrong, copySmall);

            const previewStage = document.createElement("div");
            previewStage.className = "wsm-preview-stage";
            previewStage.hidden = true;
            const preview = document.createElement(inputKind === "audio" ? "audio" : "img");
            preview.className = "wsm-upload-preview";
            preview.dataset.role = "preview";
            if (inputKind === "audio") preview.controls = true;
            else preview.alt = inputLabel + " preview";
            const previewNotice = document.createElement("span");
            previewNotice.className = "wsm-preview-notice";
            previewNotice.textContent = "!";
            previewNotice.hidden = true;
            previewStage.append(preview, previewNotice);
            const previewCaption = document.createElement("div");
            previewCaption.className = "wsm-preview-caption";
            previewCaption.hidden = true;

            const variableFile = document.createElement("div");
            variableFile.className = "wsm-variable-file";
            variableFile.hidden = true;
            const variableExtension = document.createElement("span");
            variableExtension.className = "wsm-variable-extension";
            const variableCopy = document.createElement("div");
            variableCopy.className = "wsm-variable-copy";
            const variableName = document.createElement("div");
            variableName.className = "wsm-variable-name";
            const variableRole = document.createElement("div");
            variableRole.className = "wsm-variable-role";
            variableCopy.append(variableName, variableRole);
            variableFile.append(variableExtension, variableCopy);

            const input = document.createElement("input");
            input.className = "wsm-file";
            input.type = "file";
            input.accept = useSelfPreprocessor
                ? "image/png,image/jpeg,image/webp"
                : inputKind === "audio"
                    ? "audio/wav,audio/x-wav,.wav"
                    : acceptsTensor
                        ? ".wsg,.npz,.npy,image/png,image/jpeg,image/webp,application/octet-stream"
                        : "image/png,image/jpeg,image/webp";
            dropzone.append(copy, previewStage, input);
            const hint = document.createElement("div");
            hint.className = "wsm-hint";
            hint.textContent = inputHint;
            hint.title = inputHint;
            const validation = document.createElement("div");
            validation.className = "wsm-input-validation";
            validation.hidden = true;
            row.append(titleRow, dropzone, previewCaption, variableFile, hint, validation);

            const setVariableFile = (filename, roleText) => {
                const cleanName = filename || "input.wsg.npz";
                const pieces = cleanName.split(".");
                variableExtension.textContent = (pieces[pieces.length - 1] || "WSG").toUpperCase();
                variableName.textContent = cleanName;
                variableName.title = cleanName;
                variableRole.textContent = roleText || "Actual model input · encoded at run time";
                variableFile.hidden = false;
            };
            const showPreview = (source, labelText, noticeText) => {
                preview.src = source;
                previewStage.hidden = false;
                copy.hidden = true;
                dropzone.classList.add("has-image");
                previewCaption.textContent = labelText || "";
                previewCaption.hidden = !labelText;
                previewNotice.hidden = !noticeText;
                previewNotice.title = noticeText || "";
                previewNotice.setAttribute("aria-label", noticeText || "Preview information");
            };
            const isTensorFilename = filename => {
                const lower = String(filename || "").toLowerCase();
                return lower.endsWith(".wsg") || lower.endsWith(".npz") || lower.endsWith(".npy");
            };

            if (locked) {
                const visualizationUrl = this.demoAssetUrl(demoInput.visualization_asset);
                const rawAsset = demoInput.raw_asset || "";
                const rawUrl = this.demoAssetUrl(rawAsset);
                const rawFilename = demoInput.raw_filename || rawAsset.split("/").pop() || "input.wsg.npz";
                const activeRecordId = demoRecord.id;
                const sourceToggle = document.createElement("label");
                sourceToggle.className = "wsm-demo-input-toggle wsm-latent-source-toggle";
                sourceToggle.title = "Choose whether inference encodes the raw NPZ at run time or uses the exact cached VAE latent.";
                const sourceCheckbox = document.createElement("input");
                sourceCheckbox.type = "checkbox";
                sourceCheckbox.setAttribute("aria-label", `Use latent cache for ${inputLabel}`);
                const rawSourceOption = document.createElement("span");
                rawSourceOption.className = "wsm-latent-source-option";
                rawSourceOption.textContent = "Raw NPZ";
                const latentSourceOption = document.createElement("span");
                latentSourceOption.className = "wsm-latent-source-option";
                latentSourceOption.textContent = "Latent Cache";
                sourceToggle.append(sourceCheckbox, rawSourceOption, latentSourceOption);
                titleActions.append(sourceToggle);
                input.disabled = true;
                input.hidden = true;
                showPreview(
                    visualizationUrl,
                    demoInput.visualization_label || "WSG Reader visualization · display only",
                    "This WSG Reader image is display-only and never enters inference. The switch selects raw NPZ encoding or the exact latent cache.",
                );
                setVariableFile(rawFilename, "Actual model input · validated, normalized, and VAE-encoded at run time");
                hint.innerHTML = "<span class=\"wsm-demo-lock\">Locked paper input · choose Raw NPZ or Latent Cache above</span><br>" + this.escape(inputHint);
                hint.title = "Locked paper input. " + inputHint;
                dropzone.addEventListener("click", event => {
                    event.preventDefault();
                    if (inputKind !== "audio") this.openImageLightbox(visualizationUrl, inputLabel);
                });
                const selectDemoInputSource = async useLatent => {
                    sourceCheckbox.disabled = true;
                    try {
                        const dataUrl = await this.demoDataUrl(rawUrl);
                        if (this.activeDemoRecord?.id !== activeRecordId || sourceCheckbox.checked !== useLatent) return;
                        this.uploads.set(token, dataUrl);
                        this.uploadFilenames.set(token, rawFilename);
                        if (useLatent) this.demoLatentInputs.add(token);
                        else this.demoLatentInputs.delete(token);
                        const latentAsset = useLatent ? String(demoInput.latent_asset || "") : "";
                        if (useLatent && !latentAsset) throw new Error(`No latent cache is published for ${inputLabel}`);
                        setVariableFile(
                            useLatent ? (latentAsset.split("/").pop() || "cached-latent.pt") : rawFilename,
                            useLatent
                                ? "Actual model input · exact cached VAE latent; raw NPZ encoding is bypassed"
                                : "Actual model input · validated, normalized, and VAE-encoded at run time",
                        );
                        this.updateRunState();
                        await this.validateInputUpload(
                            token, dataUrl, rawFilename, validation, false,
                            {paperDemo: true, latentAsset},
                        );
                    } catch (error) {
                        this.showError(error.message);
                    } finally {
                        if (this.activeDemoRecord?.id === activeRecordId) sourceCheckbox.disabled = false;
                    }
                };
                sourceCheckbox.addEventListener("change", () => selectDemoInputSource(sourceCheckbox.checked));
                selectDemoInputSource(false);
            } else {
                const resetUploadPresentation = () => {
                    input.value = "";
                    variableFile.hidden = true;
                    previewStage.hidden = true;
                    previewCaption.hidden = true;
                    previewNotice.hidden = true;
                    copy.hidden = false;
                    dropzone.classList.remove("has-image");
                    validation.hidden = true;
                    this.uploads.delete(token);
                    this.uploadFilenames.delete(token);
                    this.inputValidation.delete(token);
                };
                const refreshInputMode = () => {
                    copyStrong.textContent = useSelfPreprocessor
                        ? "Drop an RGB image for WSM to estimate this modality"
                        : rawUploadPrompt;
                    copySmall.textContent = useSelfPreprocessor
                        ? "or click to choose a PNG, JPEG, or WebP file"
                        : inputKind === "audio"
                            ? "or click to choose a WAV file"
                            : acceptsTensor
                                ? "choose .wsg, .npz, .npy, PNG, JPEG, or WebP"
                                : "or click to choose a PNG, JPEG, or WebP file";
                    hint.textContent = useSelfPreprocessor
                        ? `WSM ${selfPreprocessor.head} first estimates ${token} from RGB; its generated raw WSG values are then encoded as this condition.`
                        : rawInputHint;
                    hint.title = hint.textContent;
                    input.accept = useSelfPreprocessor
                        ? "image/png,image/jpeg,image/webp"
                        : inputKind === "audio"
                            ? "audio/wav,audio/x-wav,.wav"
                            : acceptsTensor
                                ? ".wsg,.npz,.npy,image/png,image/jpeg,image/webp,application/octet-stream"
                                : "image/png,image/jpeg,image/webp";
                };
                const loadFile = async file => {
                    if (!file) return;
                    const dataUrl = await this.readFile(file);
                    const filename = file.name || "";
                    const tensorInput = !useSelfPreprocessor && isTensorFilename(filename);
                    this.uploads.set(token, dataUrl);
                    this.uploadFilenames.set(token, filename);
                    variableFile.hidden = true;
                    previewStage.hidden = true;
                    previewCaption.hidden = true;
                    previewNotice.hidden = true;
                    copy.hidden = false;
                    dropzone.classList.remove("has-image");
                    if (tensorInput) {
                        setVariableFile(filename, "Actual model input · a quick preview is generated separately");
                        copyStrong.textContent = "Reading " + filename;
                        copySmall.textContent = "Validating raw World Snapshot values";
                    } else {
                        showPreview(dataUrl, "", "");
                    }
                    this.updateRunState();
                    const report = await this.validateInputUpload(
                        token, dataUrl, filename, validation, !useSelfPreprocessor,
                        {selfPreprocess: useSelfPreprocessor},
                    );
                    if (tensorInput && report?.preview_data_url) {
                        showPreview(
                            report.preview_data_url,
                            report.preview_label || "Quick pseudo-RGB preview · display only",
                            "This is a fast pseudo-RGB preview. The uploaded NPZ/NPY/WSG file below is the actual model input.",
                        );
                    }
                };
                if (selfPreprocessorCheckbox) {
                    selfPreprocessorCheckbox.addEventListener("change", () => {
                        useSelfPreprocessor = selfPreprocessorCheckbox.checked;
                        if (useSelfPreprocessor) this.selfPreprocessorInputs.add(token);
                        else this.selfPreprocessorInputs.delete(token);
                        resetUploadPresentation();
                        refreshInputMode();
                        this.updateRunState();
                    });
                }
                input.addEventListener("change", () => loadFile(input.files?.[0]));
                dropzone.addEventListener("dragover", event => {
                    event.preventDefault();
                    dropzone.classList.add("dragging");
                });
                dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
                dropzone.addEventListener("drop", event => {
                    event.preventDefault();
                    dropzone.classList.remove("dragging");
                    loadFile(event.dataTransfer?.files?.[0]);
                });
                if (inputKind !== "audio") preview.addEventListener("click", event => {
                    event.preventDefault();
                    event.stopPropagation();
                    this.openImageLightbox(preview.src, inputLabel);
                });
            }
            holder.appendChild(row);
        });

        const appendTextCard = (titleText, roleText, textarea, hintText, locked = false) => {
            const row = document.createElement("div");
            row.className = "wsm-upload wsm-text-input-card" + (locked ? " demo-locked" : "");
            const titleRow = document.createElement("div");
            titleRow.className = "wsm-upload-title-row";
            const title = document.createElement("div");
            title.className = "wsm-upload-title";
            title.textContent = titleText;
            title.title = titleText;
            const role = document.createElement("span");
            role.className = "wsm-role-badge";
            role.textContent = roleText;
            titleRow.append(title, role);
            const textBody = document.createElement("div");
            textBody.className = "wsm-text-input-body";
            textBody.appendChild(textarea);
            const hint = document.createElement("div");
            hint.className = "wsm-hint";
            hint.textContent = hintText;
            hint.title = hintText;
            row.append(titleRow, textBody, hint);
            holder.appendChild(row);
        };

        appendTextCard(
            "Modality semantic prompt",
            "Automatic · read-only",
            modalityPrompt,
            "System-generated route semantics. This is not an image caption.",
        );
        if (inputs.includes("caption")) {
            appendTextCard(
                "Caption input",
                "Text",
                prompt,
                demoRecord
                    ? "Paper caption loaded. You may edit it; edited text is re-encoded."
                    : "Natural-language content input; separate from the modality semantic prompt.",
            );
        }
        this.updateRunState();
    }

    beginGraphCanvasResize(event) {
        if (event.button !== 0) return;
        const handle = event.currentTarget;
        const canvas = this.$("graph-canvas");
        if (!canvas || !handle?.dataset.canvasResize) return;
        const rect = canvas.getBoundingClientRect();
        this.graphCanvasResize = {
            pointerId: event.pointerId,
            handle,
            direction: handle.dataset.canvasResize,
            startX: event.clientX,
            startY: event.clientY,
            width: rect.width,
            height: rect.height,
            topOffset: Number.parseFloat(canvas.style.top) || 0,
            marginBottom: Number.parseFloat(canvas.style.marginBottom) || 0,
        };
        handle.classList.add("resizing");
        handle.setPointerCapture?.(event.pointerId);
        event.preventDefault();
        event.stopPropagation();
    }

    resizeGraphCanvas(event) {
        const state = this.graphCanvasResize;
        if (!state || state.pointerId !== event.pointerId) return;
        const canvas = this.$("graph-canvas");
        if (!canvas) return;
        const dx = event.clientX - state.startX;
        const dy = event.clientY - state.startY;
        const direction = state.direction;
        const horizontalSign = direction.includes("right") ? 1 : (direction.includes("left") ? -1 : 0);
        const verticalSign = direction.includes("bottom") ? 1 : (direction.includes("top") ? -1 : 0);
        const isCorner = direction.includes("-");

        const minWidth = Math.min(560, Math.max(340, window.innerWidth - 40));
        const minHeight = 260;
        const maxWidth = Math.max(state.width, Math.min(2200, window.innerWidth * 1.35));
        const maxHeight = Math.max(state.height, Math.min(1200, window.innerHeight * 1.45));
        let width = state.width;
        let height = state.height;

        if (isCorner) {
            const widthScale = 1 + (2 * horizontalSign * dx) / Math.max(state.width, 1);
            const heightScale = 1 + (2 * verticalSign * dy) / Math.max(state.height, 1);
            let scale = Math.abs(widthScale - 1) >= Math.abs(heightScale - 1)
                ? widthScale
                : heightScale;
            const minimumScale = Math.max(minWidth / state.width, minHeight / state.height);
            const maximumScale = Math.min(maxWidth / state.width, maxHeight / state.height);
            scale = Math.max(minimumScale, Math.min(maximumScale, scale));
            width = state.width * scale;
            height = state.height * scale;
        } else if (horizontalSign) {
            width = Math.max(minWidth, Math.min(maxWidth, state.width + 2 * horizontalSign * dx));
        } else if (verticalSign) {
            height = Math.max(minHeight, Math.min(maxHeight, state.height + 2 * verticalSign * dy));
        }

        const verticalDelta = height - state.height;
        canvas.style.width = width.toFixed(1) + "px";
        canvas.style.height = height.toFixed(1) + "px";
        canvas.style.top = (state.topOffset - verticalDelta / 2).toFixed(1) + "px";
        canvas.style.marginBottom = (state.marginBottom - verticalDelta / 2).toFixed(1) + "px";
        event.preventDefault();
    }

    endGraphCanvasResize(event) {
        const state = this.graphCanvasResize;
        if (!state || state.pointerId !== event.pointerId) return;
        state.handle.releasePointerCapture?.(event.pointerId);
        state.handle.classList.remove("resizing");
        this.graphCanvasResize = null;
        event.preventDefault();
    }

    resetGraphView() {
        const canvas = this.$("graph-canvas");
        if (canvas) {
            canvas.style.removeProperty("width");
            canvas.style.removeProperty("height");
            canvas.style.removeProperty("top");
            canvas.style.removeProperty("margin-bottom");
        }
        this.taskGraph.resetLayout();
    }

    normalizeGraphToken(token) {
        return ['Reference_IMG', 'CurrentFrame_RGB_Condition', 'CurrentFrame_RGB_Reference']
            .includes(token) ? '0-2' : token;
    }

    setGraphSelectionMode(mode) {
        this.graphSelectionMode = mode === 'output' ? 'output' : 'input';
        const inputButton = this.$('graph-input-mode');
        const outputButton = this.$('graph-output-mode');
        if (!inputButton || !outputButton) return;
        const inputActive = this.graphSelectionMode === 'input';
        inputButton.classList.toggle('active', inputActive);
        outputButton.classList.toggle('active', !inputActive);
        inputButton.setAttribute('aria-pressed', String(inputActive));
        outputButton.setAttribute('aria-pressed', String(!inputActive));
    }

    handleGraphSelection(selection) {
        if (!this.catalog || !selection) return;
        this.showError('');
        if (selection.type === 'task') {
            this.selectTaskFromGraph(selection.taskId);
            return;
        }
        if (selection.type === 'composite') {
            this.selectCompositeFromGraph(selection.taskId);
            return;
        }
        if (selection.type !== 'node') return;
        if (this.graphSelectionMode === 'input') {
            this.selectGraphInput(selection.token);
        } else {
            this.selectGraphOutput(selection.token);
        }
    }

    selectCompositeFromGraph(taskId) {
        const task = this.catalog.tasks.find(item => item.id === taskId);
        if (!task || !task.supported) {
            this.showError('This Chain of Visual route has no runnable RGB estimation Head.');
            return;
        }
        const normalizedInputs = (task.inputs || []).map(token => this.normalizeGraphToken(token));
        if (normalizedInputs.length !== 1 || normalizedInputs[0] !== '0-2') {
            this.showError('This Head cannot use the RGB generated by the first stage as its only visual input.');
            return;
        }
        const signature = this.catalog.input_signatures.find(item => item.id === task.input_signature);
        if (!signature) {
            this.showError('The second-stage Head has no registered RGB input signature.');
            return;
        }
        this.activeDemoRecord = null;
        this.signature = signature;
        this.task = task;
        this.compositeRoute = {
            taskId: task.id,
            target: task.target,
            label: 'Text → RGB → ' + (task.target_description || task.target),
        };
        this.uploads.clear();
        this.uploadFilenames.clear();
        this.inputValidation.clear();
        this.demoLatentInputs.clear();
        this.selfPreprocessorInputs.clear();
        this.$('signature-search').value = '';
        this.$('signature-picker').classList.remove('open');
        this.$('task-search').value = '';
        this.$('prompt').readOnly = false;
        this.renderDemoSelector();
        this.renderSignatures();
        this.renderUploads();
        this.renderTasks(false);
        this.renderCheckpoint();
        this.setGraphSelectionMode('output');
        this.showQualityWarning(task);
    }

    selectGraphInput(token) {
        const normalized = this.normalizeGraphToken(token);
        const candidates = this.catalog.input_signatures
            .filter(item => (item.inputs || []).some(input => this.normalizeGraphToken(input) === normalized))
            .sort((left, right) => {
                const leftExact = left.inputs.length === 1
                    && this.normalizeGraphToken(left.inputs[0]) === normalized;
                const rightExact = right.inputs.length === 1
                    && this.normalizeGraphToken(right.inputs[0]) === normalized;
                return Number(rightExact) - Number(leftExact)
                    || right.available_count - left.available_count
                    || right.configured_count - left.configured_count
                    || left.inputs.length - right.inputs.length
                    || left.label.localeCompare(right.label, undefined, { numeric: true });
            });
        const signature = candidates[0];
        if (!signature) {
            this.showError('No configured input signature contains ' + token + '.');
            return;
        }
        this.selectSignature(signature.id, false);
        this.setGraphSelectionMode('output');
    }

    selectGraphOutput(token) {
        if (!this.signature) {
            this.setGraphSelectionMode('input');
            this.selectGraphInput(token);
            return;
        }
        const normalized = this.normalizeGraphToken(token);
        const candidates = this.catalog.tasks
            .filter(task =>
                task.pair_exists
                && task.input_signature === this.signature.id
                && this.normalizeGraphToken(task.target) === normalized
            )
            .sort((left, right) =>
                Number(right.supported) - Number(left.supported)
                || Number(right.trained) - Number(left.trained)
                || right.checkpoints.length - left.checkpoints.length
                || left.index - right.index
            );
        if (!candidates.length) {
            this.showError(
                'No Channel Head connects the current input to ' + token
                + '. Choose “Select input” to start a different path.'
            );
            return;
        }
        this.selectTaskFromGraph(candidates[0].id);
    }

    selectTaskFromGraph(taskId) {
        const task = this.catalog.tasks.find(item => item.id === taskId);
        if (!task) {
            this.showError('The selected graph edge does not map to a configured Channel Head.');
            return;
        }
        this.compositeRoute = null;
        if (this.demoEnabled) {
            const demoRecord = this.demoRecordsForTask(task.id)[0];
            if (demoRecord) this.activateDemoRecord(demoRecord.id);
            else this.showError('This Head has no Figure 14–39 demo example.');
            return;
        }
        const signature = this.catalog.input_signatures.find(item => item.id === task.input_signature);
        if (!signature) {
            this.showError('The selected Channel Head has no registered input signature.');
            return;
        }
        if (this.signature?.id !== signature.id) {
            this.signature = signature;
            this.uploads.clear();
            this.uploadFilenames.clear();
            this.inputValidation.clear();
            this.demoLatentInputs.clear();
            this.selfPreprocessorInputs.clear();
            this.renderSignatures();
            this.renderUploads();
        }
        this.$('signature-search').value = '';
        this.$('signature-picker').classList.remove('open');
        this.$('task-search').value = '';
        if (!task.trained) this.$('trained-only').checked = false;
        this.task = task;
        this.renderTasks(false);
        this.renderCheckpoint();
        this.updateRunState();
        this.setGraphSelectionMode('output');
        this.showQualityWarning(task);
    }

    matchingTasks() {
        if (!this.catalog || !this.signature) return [];
        const query = this.$('task-search').value.trim().toLowerCase();
        const trainedOnly = this.$("trained-only").checked;
        return this.catalog.tasks
            .filter(task =>
                (!trainedOnly || task.trained) &&
                task.input_signature === this.signature.id &&
                [task.source, task.head, task.target, task.target_description, task.task_type, task.adapter_task].join(' ').toLowerCase().includes(query)
            )
            .sort((left, right) => {
                const leftName = left.target_description || left.target;
                const rightName = right.target_description || right.target;
                return leftName.localeCompare(rightName, undefined, { numeric: true }) ||
                    left.head.localeCompare(right.head, undefined, { numeric: true });
            });
    }

    renderTasks(autoSelectTask = true) {
        const tasks = this.matchingTasks();
        const holder = this.$('task-list');
        holder.innerHTML = tasks.length ? tasks.map(task => `
            <button class="wsm-task ${this.task?.id === task.id ? 'active' : ''}"
                    data-task="${this.escape(task.id)}" ${task.trained ? '' : 'disabled'}>
                <div class="wsm-task-name-row">
                    <div class="wsm-task-name">${this.escape(task.source === 'sparse' ? 'Sparse' : (task.source === 'custom' ? 'Custom' : 'Dense'))} · ${this.escape(task.head)} · ${this.escape(task.target)}</div>
                    ${task.official_recommended ? '<span class="wsm-task-badge recommended">★ Recommended</span>' : ''}
                    ${task.quality_notice ? '<span class="wsm-task-badge warning">⚠ Needs improvement</span>' : ''}
                </div>
                <div class="wsm-task-target">${this.escape(task.target_description || 'No target description')}</div>
                <div class="wsm-task-type">
                    ${this.escape(task.task_type)} · ${task.checkpoints.length} checkpoint(s)
                    ${task.supported ? '' : ` · ${this.escape(task.unsupported_reason || 'Model unavailable')}`}
                </div>
            </button>
        `).join('') : '<div class="wsm-empty">No output matches this input.</div>';
        holder.querySelectorAll('[data-task]').forEach(button => {
            button.addEventListener('click', () => {
                const selectedTask = tasks.find(task => task.id === button.dataset.task);
                if (this.demoEnabled) {
                    const demoRecord = this.demoRecordsForTask(selectedTask?.id)[0];
                    if (demoRecord) this.activateDemoRecord(demoRecord.id);
                    return;
                }
                this.task = selectedTask;
                this.compositeRoute = null;
                this.renderUploads();
                this.renderTasks();
                this.renderCheckpoint();
                this.updateRunState();
                this.setGraphSelectionMode('output');
                this.showQualityWarning(this.task);
            });
        });
        const firstSelectable = tasks.find(task => task.supported) || tasks.find(task => task.trained);
        if (autoSelectTask && !this.task && firstSelectable) {
            this.task = firstSelectable;
            this.renderUploads();
            this.renderTasks();
            this.renderCheckpoint();
        }
        this.renderOutputEncoding();
        if (this.$('modality-prompt')) {
            this.$('modality-prompt').value = this.task?.modality_prompt || '';
        }
        this.updateRunState();
        this.renderTaskGraph();
    }

    renderTaskGraph() {
        this.taskGraph?.render(this.catalog, this.signature, this.task, this.compositeRoute);
    }

    renderOutputEncoding() {
        const panel = this.$("output-encoding-legend");
        if (!panel) return;
        if (!this.task) {
            panel.innerHTML = "<div class=\"wsm-encoding-kicker\">Output encoding</div><h4 class=\"wsm-encoding-title\">WSG Reader legend</h4><div class=\"wsm-encoding-rule\"></div><div class=\"wsm-encoding-placeholder\">Select an output to inspect its paper-style encoding, range, and color semantics.</div>";
            return;
        }
        const outputToken = this.task.target;
        let token = outputToken;
        let role = "Prediction / GT encoding";
        if (outputToken === "0-2") {
            const inputs = this.task.inputs || this.signature?.inputs || [];
            const referenceTokens = new Set([
                "caption", "0-2", "Reference_IMG", "FirstFrame_RGB",
                "CurrentFrame_RGB_Reference", "CurrentFrame_RGB_Condition",
            ]);
            token = inputs.find(input => !referenceTokens.has(input)) || outputToken;
            if (token !== outputToken) role = "Condition encoding";
        }
        const inputDetail = (this.task.input_details || []).find(item => item.token === token);
        const label = token === outputToken
            ? (this.task.target_description || token)
            : (inputDetail?.name || token);
        const spec = outputEncodingSpec(token, label);
        let body;
        if (spec.type === "swatches") {
            body = "<div class=\"wsm-encoding-swatches\">" + spec.items.map(([color, name]) => "<span class=\"wsm-encoding-swatch\"><i style=\"--swatch:" + this.escape(color) + "\"></i><span>" + this.escape(name) + "</span></span>").join("") + "</div>";
        } else if (spec.type === "gradient") {
            body = "<div class=\"wsm-encoding-gradient\" style=\"--encoding-gradient:" + this.escape(spec.gradient) + "\"></div><div class=\"wsm-encoding-scale\"><span>" + this.escape(spec.low) + "</span><span>" + this.escape(spec.high) + "</span></div>";
        } else {
            body = "";
        }
        panel.innerHTML = "<div class=\"wsm-encoding-kicker\">Output encoding · " + this.escape(token) + "</div><h4 class=\"wsm-encoding-title\">" + this.escape(label) + "</h4><div class=\"wsm-encoding-rule\"></div><div class=\"wsm-encoding-role\">" + this.escape(role) + "</div>" + body + "<div class=\"wsm-encoding-note\">" + this.escape(spec.note) + "</div>";
    }

    renderCheckpoint() {
        const select = this.$('checkpoint');
        if (!this.task) {
            select.innerHTML = '<option>Select an output first</option>';
            this.$('task-description').textContent = 'Select an output Channel Head.';
            return;
        }
        select.innerHTML = this.task.checkpoints.map(item =>
            `<option value="${this.escape(item.path)}">${this.escape(item.label)}</option>`
        ).join('');
        select.value = this.task.default_checkpoint;
        const relation = this.compositeRoute
            ? this.compositeRoute.label + ' · second-stage ' + this.task.head
            : (this.signature?.label || this.task.input_description || this.task.input_signature)
                + ' → ' + (this.task.target_description || this.task.target);
        this.$('task-description').textContent = relation
            + (this.task.quality_notice ? ' · ⚠ Further improvement recommended' : '');
    }

    updateRunState() {
        const ready = this.compositeRoute
            ? Boolean(this.task?.supported && this.$('prompt').value.trim())
            : Boolean(this.task?.supported) && (
                this.signature?.inputs.filter(token => token !== 'caption') || []
            ).every(token => {
                const report = this.inputValidation.get(token);
                return this.uploads.has(token) && report && report.runnable !== false;
            });
        const button = this.$('run');
        button.disabled = !ready;
        if (!this.inferenceJobId) button.textContent = this.compositeRoute
            ? 'Run two-stage inference' : 'Run inference';
    }

    renderInferenceLog(snapshot) {
        const panel = this.$('inference-log-panel');
        const output = this.$('inference-log');
        const status = String(snapshot?.status || 'running');
        panel.hidden = false;
        panel.dataset.status = status;
        this.$('inference-log-status').textContent = status;
        output.textContent = snapshot?.log || '[Browser] Waiting for worker output...';
        window.requestAnimationFrame(() => {
            output.scrollTop = output.scrollHeight;
        });
    }

    renderInferenceOutputs(response) {
        const holder = this.$('output-list');
        const result = this.$('result');
        const outputs = Array.isArray(response?.outputs) && response.outputs.length
            ? response.outputs
            : [{ label: 'Inference output', image: response.image, selected: true }];
        holder.innerHTML = '';
        holder.hidden = outputs.length <= 1;
        if (outputs.length <= 1) {
            result.src = outputs[0].image;
            return;
        }

        const title = document.createElement('div');
        title.className = 'wsm-output-list-title';
        title.textContent = `Inference outputs · ${outputs.length} view(s)`;
        const items = document.createElement('div');
        items.className = 'wsm-output-items';
        const buttons = [];
        const selectOutput = (record, button) => {
            result.src = record.image;
            buttons.forEach(item => item.classList.toggle('active', item === button));
        };
        outputs.forEach(record => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'wsm-output-item';
            button.title = record.label || record.filename || 'WSG reader output';
            const thumbnail = document.createElement('img');
            thumbnail.alt = button.title;
            thumbnail.src = record.image;
            const label = document.createElement('span');
            label.textContent = button.title;
            button.append(thumbnail, label);
            button.addEventListener('click', () => selectOutput(record, button));
            buttons.push(button);
            items.appendChild(button);
        });
        holder.append(title, items);
        const selectedIndex = Math.max(0, outputs.findIndex(record => record.selected));
        selectOutput(outputs[selectedIndex], buttons[selectedIndex]);
    }

    async waitForInference(jobId) {
        while (this.inferenceJobId === jobId) {
            const snapshot = await this.sendRequest({
                action: 'get_inference_status',
                data: { job_id: jobId },
            });
            this.renderInferenceLog(snapshot);
            if (snapshot.status === 'completed') return snapshot;
            if (snapshot.status === 'failed') {
                throw new Error(snapshot.error_message || 'WSM inference failed. See the inference log for details.');
            }
            await new Promise(resolve => window.setTimeout(resolve, 700));
        }
        throw new Error('Inference monitoring was interrupted.');
    }

    async runInference() {
        const button = this.$('run');
        const result = this.$('result');
        const empty = this.$('empty');
        this.showError('');
        button.disabled = true;
        button.textContent = this.compositeRoute ? 'Running two stages...' : 'Running...';
        empty.hidden = false;
        empty.textContent = this.compositeRoute
            ? 'Generating RGB, then estimating the selected visual modality...'
            : 'Loading the base model and running denoising...';
        result.hidden = true;
        this.$('output-list').hidden = true;
        this.$('output-list').innerHTML = '';
        this.renderInferenceLog({
            status: 'starting',
            log: '[Browser] Sending inference request to ControlWorld...',
        });
        try {
            const captionValue = this.$('prompt').value;
            const inputs = [...this.uploads.entries()].map(([token, dataUrl]) => {
                const demoInput = this.demoEnabled
                    ? this.activeDemoRecord?.inputs?.find(item => item.token === token)
                    : null;
                return {
                    token,
                    data_url: dataUrl,
                    filename: this.uploadFilenames.get(token) || "",
                    latent_asset: this.demoLatentInputs.has(token) ? (demoInput?.latent_asset || "") : "",
                    self_preprocess: this.selfPreprocessorInputs.has(token),
                };
            });
            const started = await this.sendRequest({
                action: 'start_inference',
                data: {
                    task_id: this.task.id,
                    composite: Boolean(this.compositeRoute),
                    checkpoint: this.$('checkpoint').value,
                    inputs,
                    prompt: captionValue,
                    exact_prompt_only: Boolean(
                        this.compositeRoute
                        || this.signature?.inputs?.includes('caption')
                        || (this.demoEnabled && this.activeDemoRecord?.exact_prompt_only)
                    ),
                    paper_demo: Boolean(this.demoEnabled),
                    paper_output_view: this.demoEnabled
                        ? (this.activeDemoRecord?.paper_output_view || "native")
                        : "",
                    steps: Number(this.$('steps').value),
                    size: Number(this.$('size').value),
                    seed: Number(this.$('seed').value),
                },
            });
            this.inferenceJobId = started.job_id;
            this.renderInferenceLog(started);
            const response = await this.waitForInference(started.job_id);
            this.renderInferenceOutputs(response);
            result.hidden = false;
            empty.hidden = true;
        } catch (error) {
            empty.textContent = 'Inference failed.';
            this.showError(error.message);
            const log = this.$('inference-log').textContent;
            if (!log.includes(error.message)) {
                this.renderInferenceLog({
                    status: 'failed',
                    log: `${log}\n[Browser] ERROR: ${error.message}`.trim(),
                });
            }
        } finally {
            this.inferenceJobId = null;
            this.updateRunState();
        }
    }

    showQualityWarning(task) {
        const notice = task?.quality_notice;
        if (!notice || notice.status !== 'needs_improvement') return;
        this.$('quality-warning-title').textContent =
            notice.title || 'Further improvement recommended';
        this.$('quality-warning-head').textContent =
            (task.head || 'Channel Head') + ' · '
            + (task.input_signature || 'Unknown input') + ' → '
            + (task.target_description || task.target || 'Unknown target');
        this.$('quality-warning-message').textContent = notice.message ||
            'Current evaluation indicates that this Head needs further improvement.';
        this.$('quality-warning-modal').hidden = false;
        window.requestAnimationFrame(() => {
            this.$('quality-warning-continue').focus();
        });
    }

    closeQualityWarning() {
        const modal = this.$('quality-warning-modal');
        if (modal) modal.hidden = true;
    }

    showError(message) {
        const element = this.$('error');
        element.textContent = message || '';
        element.style.display = message ? 'block' : 'none';
    }

    readFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(reader.error);
            reader.readAsDataURL(file);
        });
    }

    escape(value) {
        return String(value ?? '')
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#039;');
    }
}
