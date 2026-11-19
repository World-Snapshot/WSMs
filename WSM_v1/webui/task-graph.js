const MODALITY_NAMES = {
    caption: 'Text',
    Reference_IMG: 'Ref. RGB',
    CurrentFrame_RGB_Condition: 'Current RGB',
    CurrentFrame_RGB_Reference: 'Current RGB',
    '0-2': 'RGB',
    '3': 'Alpha',
    '32-33': 'Rel. Depth',
    '34-35': 'Metric Depth',
    '36-38': 'Normal',
    '39': 'Curvature',
    '40': 'Edge',
    '41-42': 'Optical Flow',
    '64-66': 'DWPose Sparse',
    '67-69': 'DWPose Map',
    '72-87,90-93': 'SMPL Data',
    '88-89': 'Boxes',
    '90-92': 'SMPL Map',
    '95-98': 'MediaPipe Sparse',
    '99-101': 'MediaPipe Map',
    '103': 'Human Seg.',
    '111': 'Object ID',
    '111,112,114-116': 'Object Points',
    '112-121': 'Object Skeleton',
    '122-124': 'XPose',
    '126': 'Solid Volume',
    '127': 'Surface Area',
    '128': 'Weight',
    '129': 'Support Area',
    '130': 'Friction Force',
    '131': 'Material ID',
    '132': 'Material Conf.',
    '133': 'Density',
    '134': 'Friction Coeff.',
    '135': 'Container Flag',
    '136': 'Category ID',
    '143': 'Material Region',
    '144-146': 'Ambient',
    '147-149': 'Diffuse',
    '150': 'Opacity',
    '151-153': 'Specular Color',
    '154': 'Shininess',
    '155-157': 'Irradiance',
    '161-163': 'Basecolor',
    '164-166': 'Diffuse Map',
    '167': 'Height',
    '168': 'Roughness',
    '169': 'Metallic',
    '170-172': 'Specular Map',
    '173': 'PBR Opacity',
    '175-184': 'Scene Audio',
    '195': 'Semantic Mask',
    '195_box': 'Sound Box',
    'Universal_Pose': 'Universal Pose',
};

const TASK_COLORS = {
    narrow_generation: '#2f6db3',
    general_generation: '#2f6db3',
    text_to_rgb: '#2f6db3',
    estimation_or_reconstruction: '#f28e2b',
    conditional_control: '#1f77b4',
    'conditional_control+narrow_generation': '#d84a45',
    inverse_rendering: '#7b3fc6',
    chain_of_visual: '#e7298a',
    unknown: '#6b7280',
};

const DENSE_PRIORITY = [
    '3', '32-33', '34-35', '36-38', '39', '40', '41-42', '67-69', '90-92', '99-101', '103', '195',
    '111', '136', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '143',
    '144-146', '147-149', '150', '151-153', '154', '161-163', '164-166', '167', '168', '169', '170-172', '173',
];

const SPARSE_PRIORITY = [
    '72-87,90-93', '88-89', '64-66', '95-98', '112-121', '195_box', '175-184', '122-124',
];

const RGB_ALIASES = new Set(['Reference_IMG', 'CurrentFrame_RGB_Condition', 'CurrentFrame_RGB_Reference']);
const CONTROL_TOKENS = new Set(['0-2', 'caption']);

function normalizeNode(token) {
    return RGB_ALIASES.has(token) ? '0-2' : token;
}

function taskKind(task) {
    const inputs = new Set(task.inputs || []);
    if (inputs.has('caption') && task.target === '0-2' && task.task_type !== 'conditional_control+narrow_generation') {
        return 'narrow_generation';
    }
    return TASK_COLORS[task.task_type] ? task.task_type : 'unknown';
}

function orderedNodes(tokens, priority) {
    return [...tokens].sort((left, right) => {
        const leftIndex = priority.indexOf(left);
        const rightIndex = priority.indexOf(right);
        if (leftIndex >= 0 || rightIndex >= 0) {
            return (leftIndex < 0 ? 999 : leftIndex) - (rightIndex < 0 ? 999 : rightIndex);
        }
        return left.localeCompare(right, undefined, { numeric: true });
    });
}

function wrapLabel(label, maxChars = 10) {
    const words = label.trim().split(/\s+/).filter(Boolean);
    if (!words.length) return [''];
    const lines = [];
    let current = '';
    words.forEach(word => {
        if (!current || current.length + word.length + 1 <= maxChars) {
            current = current ? current + ' ' + word : word;
        } else {
            lines.push(current);
            current = word;
        }
    });
    if (current) lines.push(current);
    if (lines.length === 1 && lines[0].length > maxChars + 3) {
        const text = lines[0];
        const middle = Math.ceil(text.length / 2);
        return [text.slice(0, middle), text.slice(middle)];
    }
    return lines;
}

function estimateRadius(lines, degree, sparse) {
    const fontSize = sparse ? 10.0 : 10.8;
    const maxChars = Math.max(...lines.map(line => line.length), 1);
    const width = maxChars * fontSize * 0.54 + 14;
    const height = lines.length * fontSize * 0.92 + 12;
    const textRadius = Math.hypot(width, height) / 2;
    const structural = (sparse ? 19 : 21) + Math.sqrt(degree || 0) * (sparse ? 0.75 : 0.9);
    const minimum = sparse ? 22 : 24;
    const maximum = sparse ? 38 : 42;
    return Math.min(Math.max(textRadius, structural, minimum), maximum);
}

function ellipsePositions(nodes, center, rx, ry, tiltDegrees, radii) {
    const tilt = tiltDegrees * Math.PI / 180;
    const positions = new Map();
    nodes.forEach((token, index) => {
        const theta = Math.PI / 2 - (Math.PI * 2 * index) / Math.max(nodes.length, 1);
        const x = rx * Math.cos(theta);
        const y = ry * Math.sin(theta);
        const rotatedX = x * Math.cos(tilt) - y * Math.sin(tilt);
        const rotatedY = x * Math.sin(tilt) + y * Math.cos(tilt);
        positions.set(token, {
            x: center.x + rotatedX,
            y: center.y - rotatedY,
            radius: radii.get(token),
        });
    });
    return positions;
}

function resolveOverlaps(positions, nodes, margin = 5, iterations = 700) {
    for (let iteration = 0; iteration < iterations; iteration += 1) {
        let moved = false;
        for (let leftIndex = 0; leftIndex < nodes.length; leftIndex += 1) {
            const left = positions.get(nodes[leftIndex]);
            if (!left) continue;
            for (let rightIndex = leftIndex + 1; rightIndex < nodes.length; rightIndex += 1) {
                const right = positions.get(nodes[rightIndex]);
                if (!right) continue;
                let dx = right.x - left.x;
                let dy = right.y - left.y;
                let distance = Math.hypot(dx, dy);
                const required = left.radius + right.radius + margin;
                if (distance >= required) continue;
                if (distance < 0.001) {
                    dx = right.x || 1;
                    dy = right.y || 1;
                    distance = Math.max(Math.hypot(dx, dy), 1);
                }
                const overlap = (required - distance) * 0.52;
                const ux = dx / distance;
                const uy = dy / distance;
                left.x -= ux * overlap;
                left.y -= uy * overlap;
                right.x += ux * overlap;
                right.y += uy * overlap;
                moved = true;
            }
        }
        if (!moved) break;
    }
}

function edgePath(source, target, curvature = 0) {
    const dx = target.x - source.x;
    const dy = target.y - source.y;
    const length = Math.max(Math.hypot(dx, dy), 1);
    const ux = dx / length;
    const uy = dy / length;
    const nx = -uy;
    const ny = ux;
    const sx = source.x + ux * (source.radius + 2);
    const sy = source.y + uy * (source.radius + 2);
    const tx = target.x - ux * (target.radius + 9);
    const ty = target.y - uy * (target.radius + 9);
    const bend = length * curvature;
    const cx = (sx + tx) / 2 + nx * bend;
    const cy = (sy + ty) / 2 + ny * bend;
    return [
        'M ', sx.toFixed(1), ' ', sy.toFixed(1),
        ' Q ', cx.toFixed(1), ' ', cy.toFixed(1),
        ' ', tx.toFixed(1), ' ', ty.toFixed(1),
    ].join('');
}

function countIncrement(map, key, amount = 1) {
    map.set(key, (map.get(key) || 0) + amount);
}

function edgeKey(source, target, kind) {
    return [source, target, kind].join('\u001f');
}

function pairKey(source, target) {
    return [source, target].sort().join('\u001f');
}

function edgeVisual(kind, duplicateCount, touchesRgb) {
    const styles = {
        conditional_control: { width: 0.70, opacity: 0.109 },
        inverse_rendering: { width: 0.756, opacity: 0.138 },
        estimation_or_reconstruction: { width: 1.476, opacity: 0.42 },
        'conditional_control+narrow_generation': { width: 0.744, opacity: 0.312 },
        unknown: { width: 0.675, opacity: 0.16 },
    };
    const visual = { ...(styles[kind] || styles.unknown) };
    if (duplicateCount > 8) {
        visual.width *= 0.72;
        if (kind !== 'conditional_control') {
            visual.opacity *= Math.min(1, 4 / Math.sqrt(duplicateCount));
        }
    }
    if (touchesRgb) visual.opacity = Math.min(visual.opacity + 0.025, 0.42);
    return visual;
}

function graphLegendMarkup() {
    const leftItems = [
        { color: TASK_COLORS.estimation_or_reconstruction, label: 'Estimation / reconstruction' },
        { color: TASK_COLORS.conditional_control, label: 'Conditional control' },
        { color: TASK_COLORS.chain_of_visual, label: "Chain of Visual", className: "chain" },
        { color: "#d9902f", label: "Sparse task route", className: "sparse-task" },
        { color: '#d9902f', fill: '#f8fafc', label: 'Sparse digitized target', node: true },
        { color: '#9ca3af', label: 'Rich holographic image / WSG region', plate: true },
    ];
    const rightItems = [
        { color: TASK_COLORS.text_to_rgb, label: 'Narrow generation: Text → RGB', className: 'primary' },
        { color: TASK_COLORS['conditional_control+narrow_generation'], label: 'Text-guided conditional generation' },
        { color: TASK_COLORS.inverse_rendering, label: 'Inverse rendering' },
        { color: '#2f6db3', fill: '#ffffff', label: 'Dense WSG / visual target', node: true },
    ];

    const renderColumn = (items, x, width, panelY) => {
        const rows = items.map((item, index) => {
            const y = panelY + 22 + index * 22;
            let sample;
            if (item.plate) {
                sample = [
                    '<ellipse cx="', x + 25, '" cy="', y,
                    '" rx="22" ry="7" transform="rotate(-18 ', x + 25, ' ', y,
                    ')" fill="none" stroke="', item.color,
                    '" stroke-width="1.65" stroke-dasharray="7 6" />',
                ].join('');
            } else if (item.node) {
                sample = [
                    '<circle cx="', x + 25, '" cy="', y,
                    '" r="6" fill="', item.fill,
                    '" stroke="', item.color, '" stroke-width="1.6" />',
                ].join('');
            } else {
                sample = [
                    '<line class="wsm-graph-legend-line ', item.className || '',
                    '" x1="', x + 4, '" y1="', y,
                    '" x2="', x + 48, '" y2="', y,
                    '" style="--legend-color:', item.color, '" />',
                ].join('');
            }
            return [
                sample,
                '<text class="wsm-graph-legend-text" x="', x + 58,
                '" y="', y + 3.5, '">', item.label, '</text>',
            ].join('');
        }).join('');
        return [
            '<g class="wsm-graph-legend-panel">',
            '<rect x="', x - 8, '" y="', panelY, '" width="', width,
            '" height="', 12 + items.length * 22, '" rx="10" />',
            rows,
            '</g>',
        ].join('');
    };

    return [
        '<g class="wsm-graph-inside-legend" aria-label="Task graph legend">',
        renderColumn(leftItems, 36, 405, 48),
        renderColumn(rightItems, 1030, 414, 562),
        '</g>',
    ].join('');
}

export class WSMTaskGraph {
    constructor(container, summary, onSelection = null) {
        this.container = container;
        this.summary = summary;
        this.onSelection = onSelection;
        this.userPositions = new Map();
        this.currentPositions = new Map();
        this.currentDimensions = { width: 1480, height: 680 };
        this.dragState = null;
        this.suppressNextActivation = false;
        this.lastRender = null;
        this.container.addEventListener('click', event => this.handleActivation(event));
        this.container.addEventListener('pointerdown', event => this.handlePointerDown(event));
        this.container.addEventListener('pointermove', event => this.handlePointerMove(event));
        this.container.addEventListener('pointerup', event => this.handlePointerUp(event));
        this.container.addEventListener('pointercancel', event => this.handlePointerUp(event));
        this.container.addEventListener('keydown', event => {
            if (event.key !== 'Enter' && event.key !== ' ') return;
            event.preventDefault();
            this.handleActivation(event);
        });
    }

    eventToGraphPoint(event) {
        const svg = this.container.querySelector(".wsm-graph-svg");
        if (!svg) return null;
        const matrix = svg.getScreenCTM();
        if (!matrix) return null;
        const point = svg.createSVGPoint();
        point.x = event.clientX;
        point.y = event.clientY;
        return point.matrixTransform(matrix.inverse());
    }

    handlePointerDown(event) {
        if (event.button !== 0) return;
        const node = event.target.closest?.(".wsm-graph-node[data-graph-node]");
        if (!node) return;
        const token = node.dataset.graphNode;
        const position = this.currentPositions.get(token);
        const point = this.eventToGraphPoint(event);
        if (!position || !point) return;
        this.dragState = {
            token,
            node,
            pointerId: event.pointerId,
            startClientX: event.clientX,
            startClientY: event.clientY,
            offsetX: point.x - position.x,
            offsetY: point.y - position.y,
            moved: false,
        };
        node.setPointerCapture?.(event.pointerId);
    }

    handlePointerMove(event) {
        const drag = this.dragState;
        if (!drag || drag.pointerId !== event.pointerId) return;
        const point = this.eventToGraphPoint(event);
        const position = this.currentPositions.get(drag.token);
        if (!point || !position) return;
        const distance = Math.hypot(
            event.clientX - drag.startClientX,
            event.clientY - drag.startClientY,
        );
        if (!drag.moved && distance < 4) return;
        drag.moved = true;
        drag.node.classList.add("dragging");
        this.container.classList.add("dragging-node");
        event.preventDefault();

        const radius = position.radius || 24;
        const { width, height } = this.currentDimensions;
        position.x = Math.max(radius + 5, Math.min(width - radius - 5, point.x - drag.offsetX));
        position.y = Math.max(radius + 5, Math.min(height - radius - 5, point.y - drag.offsetY));
        this.userPositions.set(drag.token, { x: position.x, y: position.y });
        this.updateDraggedNode(drag.token);
    }

    handlePointerUp(event) {
        const drag = this.dragState;
        if (!drag || drag.pointerId !== event.pointerId) return;
        drag.node.releasePointerCapture?.(event.pointerId);
        drag.node.classList.remove("dragging");
        this.container.classList.remove("dragging-node");
        this.suppressNextActivation = drag.moved;
        this.dragState = null;
    }

    updateDraggedNode(token) {
        const position = this.currentPositions.get(token);
        if (!position) return;
        const node = [...this.container.querySelectorAll(".wsm-graph-node[data-graph-node]")]
            .find(item => item.dataset.graphNode === token);
        if (node) node.setAttribute("transform", "translate(" + position.x.toFixed(1) + " " + position.y.toFixed(1) + ")");

        this.container.querySelectorAll("[data-edge-source][data-edge-target]").forEach(path => {
            if (path.dataset.edgeSource !== token && path.dataset.edgeTarget !== token) return;
            const source = this.currentPositions.get(path.dataset.edgeSource);
            const target = this.currentPositions.get(path.dataset.edgeTarget);
            if (!source || !target) return;
            path.setAttribute("d", edgePath(source, target, Number(path.dataset.edgeCurvature || 0)));
        });

        if (token === "175-184") {
            const label = this.container.querySelector("[data-audio-anchor]");
            if (label) {
                label.setAttribute("x", position.x.toFixed(1));
                label.setAttribute("y", (position.y - position.radius - 27).toFixed(1));
            }
        }
    }

    resetLayout() {
        this.userPositions.clear();
        if (this.lastRender) {
            const { catalog, signature, selectedTask, compositeRoute } = this.lastRender;
            this.render(catalog, signature, selectedTask, compositeRoute);
        }
    }

    handleActivation(event) {
        if (this.suppressNextActivation) {
            this.suppressNextActivation = false;
            event.preventDefault();
            event.stopPropagation();
            return;
        }
        if (!this.onSelection) return;
        const compositeEdge = event.target.closest?.("[data-graph-composite-task]");
        if (compositeEdge) {
            this.onSelection({
                type: "composite",
                taskId: compositeEdge.dataset.graphCompositeTask,
                target: compositeEdge.dataset.edgeTarget,
            });
            return;
        }
        const taskEdge = event.target.closest?.('[data-graph-task]');
        if (taskEdge) {
            this.onSelection({ type: 'task', taskId: taskEdge.dataset.graphTask });
            return;
        }
        const node = event.target.closest?.('[data-graph-node]');
        if (node) {
            this.onSelection({ type: 'node', token: node.dataset.graphNode });
        }
    }

    render(catalog, signature, selectedTask, compositeRoute = null) {
        this.lastRender = { catalog, signature, selectedTask, compositeRoute };
        if (!catalog) {
            this.container.innerHTML = '<div class="wsm-graph-empty">Task graph will appear when the catalog is available.</div>';
            this.summary.textContent = 'No task selected';
            return;
        }

        const tasks = catalog.tasks.filter(task => task.pair_exists);
        const denseTokens = new Set();
        const sparseTokens = new Set();

        tasks.forEach(task => {
            const bucket = task.source === 'sparse' ? sparseTokens : denseTokens;
            (task.inputs || []).map(normalizeNode).forEach(token => {
                if (token && !CONTROL_TOKENS.has(token)) bucket.add(token);
            });
            const target = normalizeNode(task.target);
            if (target && !CONTROL_TOKENS.has(target)) bucket.add(target);
        });

        catalog.tasks.filter(task => task.source === 'sparse').forEach(task => {
            (task.inputs || []).map(normalizeNode).forEach(token => {
                if (token && !CONTROL_TOKENS.has(token)) sparseTokens.add(token);
            });
            const target = normalizeNode(task.target);
            if (target && !CONTROL_TOKENS.has(target)) sparseTokens.add(target);
        });
        denseTokens.forEach(token => sparseTokens.delete(token));

        const denseNodes = orderedNodes(denseTokens, DENSE_PRIORITY);
        const sparseNodes = orderedNodes(sparseTokens, SPARSE_PRIORITY);
        const selectedInputs = new Set(
            compositeRoute ? ["caption"] : (signature?.inputs || []).map(normalizeNode)
        );
        const selectedTarget = selectedTask ? normalizeNode(selectedTask.target) : null;

        // A Chain of Visual route is executable only when the current catalog
        // contains a runnable, single-RGB-input Head for its second stage.
        // Paper Demo filtering therefore applies without a separate exception.
        const compositeByTarget = new Map();
        tasks.forEach(task => {
            const normalizedInputs = (task.inputs || []).map(normalizeNode);
            const target = normalizeNode(task.target);
            if (
                !task.supported
                || !target
                || target === "0-2"
                || normalizedInputs.length !== 1
                || normalizedInputs[0] !== "0-2"
            ) return;
            const current = compositeByTarget.get(target);
            if (!current || Number(task.official_recommended) > Number(current.official_recommended)) {
                compositeByTarget.set(target, task);
            }
        });

        const records = [];
        const degree = new Map();
        tasks.forEach(task => {
            const kind = taskKind(task);
            (task.inputs || []).forEach((rawSource, inputIndex) => {
                const source = normalizeNode(rawSource);
                const target = normalizeNode(task.target);
                if (!source || !target || source === target || source === 'caption') return;
                records.push({ task, source, target, kind, inputIndex });
                countIncrement(degree, source);
                countIncrement(degree, target);
            });
        });

        const radii = new Map();
        [...denseNodes, ...sparseNodes].forEach(token => {
            const sparse = sparseTokens.has(token);
            radii.set(token, estimateRadius(wrapLabel(MODALITY_NAMES[token] || token), degree.get(token), sparse));
        });

        const width = 1480;
        const height = 680;
        const center = { x: width / 2, y: 300 };
        const positions = new Map([
            ['0-2', { x: center.x, y: center.y, radius: 47 }],
            ['caption', { x: center.x, y: 642, radius: 8 }],
        ]);
        ellipsePositions(denseNodes, center, 552, 122, 18, radii).forEach((point, token) => positions.set(token, point));
        ellipsePositions(sparseNodes, center, 646, 165, 18, radii).forEach((point, token) => positions.set(token, point));
        resolveOverlaps(positions, [...denseNodes, ...sparseNodes]);
        this.userPositions.forEach((custom, token) => {
            const position = positions.get(token);
            if (!position) return;
            const radius = position.radius || 24;
            position.x = Math.max(radius + 5, Math.min(width - radius - 5, custom.x));
            position.y = Math.max(radius + 5, Math.min(height - radius - 5, custom.y));
        });
        this.currentPositions = positions;
        this.currentDimensions = { width, height };

        const pairCounts = new Map();
        const unorderedCounts = new Map();
        records.forEach(record => {
            countIncrement(pairCounts, edgeKey(record.source, record.target, record.kind));
            countIncrement(unorderedCounts, pairKey(record.source, record.target));
        });

        const kindOrder = {
            conditional_control: 0,
            unknown: 1,
            estimation_or_reconstruction: 2,
            inverse_rendering: 3,
            'conditional_control+narrow_generation': 4,
        };
        records.sort((left, right) => {
            const leftActive = selectedTask?.id === left.task.id ? 1 : 0;
            const rightActive = selectedTask?.id === right.task.id ? 1 : 0;
            if (leftActive !== rightActive) return leftActive - rightActive;
            return (kindOrder[left.kind] ?? 1) - (kindOrder[right.kind] ?? 1)
                || left.source.localeCompare(right.source)
                || left.target.localeCompare(right.target)
                || left.task.id.localeCompare(right.task.id);
        });

        const pairSeen = new Map();
        const edgeMarkup = records.map(record => {
            if (!positions.has(record.source) || !positions.has(record.target)) return '';
            const key = edgeKey(record.source, record.target, record.kind);
            const localTotal = pairCounts.get(key) || 1;
            const localIndex = pairSeen.get(key) || 0;
            pairSeen.set(key, localIndex + 1);
            const offset = localTotal === 1
                ? 0
                : (localIndex - (localTotal - 1) / 2) / Math.max(localTotal - 1, 1);
            const reverseCount = pairCounts.get(edgeKey(record.target, record.source, record.kind)) || 0;
            const hasTwoWay = reverseCount > 0 || (unorderedCounts.get(pairKey(record.source, record.target)) || 0) > localTotal;
            let baseCurvature;
            let spread;
            if (hasTwoWay) {
                const laneSign = record.source < record.target ? 1 : -1;
                baseCurvature = (record.source === '0-2' || record.target === '0-2' ? 0.10 : 0.14) * laneSign;
                spread = 0.075;
            } else {
                baseCurvature = record.source === '0-2' || record.target === '0-2' ? 0 : 0.055;
                spread = 0.16;
            }
            const curvature = baseCurvature + spread * offset;
            const isCandidate = signature?.id === record.task.input_signature;
            const isActive = selectedTask?.id === record.task.id;
            const classes = ["wsm-graph-edge"];
            if (record.task.source === "sparse") classes.push("sparse-task");
            if (isCandidate) classes.push('candidate');
            if (isActive) classes.push('active');
            const visual = edgeVisual(
                record.kind,
                localTotal,
                record.source === '0-2' || record.target === '0-2',
            );
            return [
                '<path class="', classes.join(' '),
                '" data-graph-task="', this.escape(record.task.id), '" role="button"',
                '" data-edge-source="', this.escape(record.source),
                '" data-edge-target="', this.escape(record.target),
                '" data-edge-curvature="', curvature.toFixed(6),
                '" d="', edgePath(positions.get(record.source), positions.get(record.target), curvature),
                '" style="--edge-color:', TASK_COLORS[record.kind] || TASK_COLORS.unknown,
                ';--edge-width:', visual.width.toFixed(3),
                ';--edge-opacity:', visual.opacity.toFixed(3), '" />',
            ].join('');
        }).join('');

        const chainTargets = denseNodes.filter(token => compositeByTarget.has(token));
        const chainMarkup = chainTargets.map((token, index) => {
            const task = compositeByTarget.get(token);
            const curvature = -0.18 + 0.36 * (index / Math.max(chainTargets.length - 1, 1));
            const active = compositeRoute?.taskId === task.id;
            const pathData = edgePath(positions.get("caption"), positions.get(token), curvature);
            return [
                "<path class=\"wsm-graph-edge-hit\" data-graph-composite-task=\"", this.escape(task.id),
                "\" data-edge-source=\"caption\" data-edge-target=\"", this.escape(token),
                "\" data-edge-curvature=\"", curvature.toFixed(6), "\" d=\"", pathData, "\" />",
                '<path class="wsm-graph-edge chain', active ? ' active' : '',
                '" data-graph-composite-task="', this.escape(task.id),
                '" role="button" tabindex="0" aria-label="Run Text to RGB to ',
                this.escape(MODALITY_NAMES[token] || token),
                '" data-edge-source="caption" data-edge-target="', this.escape(token),
                '" data-edge-curvature="', curvature.toFixed(6), '" d="',
                pathData,
                '" style="--edge-color:', TASK_COLORS.chain_of_visual, '" />',
            ].join('');
        }).join('');

        const primaryClasses = ['wsm-graph-edge', 'primary'];
        const selectedTaskInputs = new Set((selectedTask?.inputs || []).map(normalizeNode));
        if (signature?.inputs?.length === 1 && selectedInputs.has('caption')) primaryClasses.push('candidate');
        if ((selectedTaskInputs.has('caption') && selectedTarget === '0-2') || compositeRoute) primaryClasses.push('active');
        const primaryMarkup = [
            '<path class="', primaryClasses.join(' '),
            '" data-edge-source="caption" data-edge-target="0-2" data-edge-curvature="0" d="',
            edgePath(positions.get('caption'), positions.get('0-2'), 0),
            '" style="--edge-color:', TASK_COLORS.text_to_rgb, '" />',
        ].join('');

        const nodeMarkup = [...positions.entries()].map(([token, point]) => {
            if (token === 'caption') {
                const classes = ['wsm-graph-text-node'];
                if (selectedInputs.has(token)) classes.push('input-active');
                if (selectedTask && selectedInputs.has(token)) classes.push('task-active');
                return [
                    '<g class="', classes.join(' '), '" data-graph-node="caption" role="button" tabindex="0" aria-label="Select Text" transform="translate(', point.x, ' ', point.y, ')">',
                    '<rect class="text-hitbox" x="-76" y="-25" width="152" height="50" rx="4" />',
                    '<text y="2.5" text-anchor="middle" dominant-baseline="central">Text Space</text>',
                    '</g>',
                ].join('');
            }

            const classes = ['wsm-graph-node'];
            classes.push(token === '0-2' ? 'rgb' : (sparseTokens.has(token) ? 'sparse' : 'dense'));
            if (selectedInputs.has(token)) classes.push('input-active');
            if (selectedTarget === token) classes.push('target-active');
            if (selectedTask && (selectedInputs.has(token) || selectedTarget === token)) classes.push('task-active');
            const label = MODALITY_NAMES[token] || token;
            const lines = wrapLabel(label);
            return [
                '<g class="', classes.join(' '),
                '" data-graph-node="', this.escape(token),
                '" role="button" tabindex="0" aria-label="Select ', this.escape(label),
                '" transform="translate(', point.x.toFixed(1), ' ', point.y.toFixed(1), ')">',
                '<title>', this.escape(label), ' · ', this.escape(token), ' · Drag to rearrange; click to select</title>',
                '<circle class="node-shadow" cx="3.5" cy="4" r="', point.radius.toFixed(1), '" />',
                '<circle class="node-disc" r="', point.radius.toFixed(1), '" />',
                '<text text-anchor="middle" dominant-baseline="central">',
                lines.map((line, index) => [
                    '<tspan x="0" dy="',
                    index === 0 ? (lines.length > 1 ? '-0.43em' : '0') : '1.02em',
                    '">', this.escape(line), '</tspan>',
                ].join('')).join(''),
                '</text>',
                '</g>',
            ].join('');
        }).join('');

        const audio = positions.get('175-184');
        const audioLabel = audio
            ? [
                '<text class="wsm-graph-audio-label" data-audio-anchor="175-184" x="', audio.x.toFixed(1),
                '" y="', (audio.y - audio.radius - 27).toFixed(1),
                '" text-anchor="middle">Audio Space</text>',
            ].join('')
            : '';

        this.container.innerHTML = [
            '<svg class="wsm-graph-svg" viewBox="0 0 ', width, ' ', height,
            '" role="img" aria-label="WSM directed task graph">',
            '<defs>',
            '<marker id="wsm-graph-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth">',
            '<path d="M0,0 L9,4.5 L0,9 z" fill="context-stroke" />',
            '</marker>',
            '<filter id="wsm-graph-soft-shadow" x="-30%" y="-30%" width="170%" height="180%">',
            '<feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#344054" flood-opacity="0.12" />',
            '</filter>',
            '</defs>',
            '<ellipse class="wsm-graph-plate shadow" cx="748" cy="310" rx="646" ry="165" transform="rotate(-18 748 310)" />',
            '<ellipse class="wsm-graph-plate inner" cx="740" cy="300" rx="552" ry="122" transform="rotate(-18 740 300)" />',
            '<ellipse class="wsm-graph-plate outer" cx="740" cy="300" rx="646" ry="165" transform="rotate(-18 740 300)" />',
            '<text class="wsm-graph-space-label" x="740" y="31" text-anchor="middle">Visual Space</text>',
            audioLabel,
            '<g class="wsm-graph-chain-edges">', chainMarkup, '</g>',
            '<g class="wsm-graph-flow-edges">', edgeMarkup, '</g>',
            '<g class="wsm-graph-primary-edge">', primaryMarkup, '</g>',
            '<g class="wsm-graph-nodes">', nodeMarkup, '</g>',
            graphLegendMarkup(),
            '</svg>',
        ].join('');

        if (compositeRoute && selectedTask) {
            const target = MODALITY_NAMES[selectedTask.target] || selectedTask.target;
            this.summary.textContent = 'Chain of Visual · Text → RGB → ' + target
                + ' · second stage ' + selectedTask.head;
        } else if (selectedTask) {
            const inputs = (selectedTask.inputs || []).map(token => MODALITY_NAMES[token] || token).join(' + ');
            const target = MODALITY_NAMES[selectedTask.target] || selectedTask.target;
            const sourceLabel = selectedTask.source === 'sparse' ? 'Sparse' : (selectedTask.source === 'custom' ? 'Custom' : 'Dense');
            this.summary.textContent = sourceLabel + ' · ' + selectedTask.head + ': ' + inputs + ' → ' + target;
        } else if (signature) {
            this.summary.textContent = signature.label + ': select an output to highlight its Channel Head';
        } else {
            this.summary.textContent = tasks.length + ' trainable Channel Heads · '
                + denseNodes.length + ' dense · ' + sparseNodes.length + ' sparse modalities';
        }
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
