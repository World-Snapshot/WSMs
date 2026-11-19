<h2 style="display: none;">Teaser</h2>

<details>
<summary id="rotating-summary">TLDR (1/15): Requirements for implementing an ideal world model: 1. Encode all information to the boundary. 2. Conform to conformal symmetry. Follow this rule, flattens the world into a two-dimensional space, and visual generation is almost lossless world generation.</summary>
<!--
<center><a href="https://world-snapshot.github.io/doc/index.html?page=S6_materials/02_guideline.md#tldr-of-project-page">All TLDRs</a></center>
<div style="display: flex; justify-content: space-between; gap: 20px; margin-top: 15px;">
    <img src="./static/images/WaveGen_cover.jpg" alt="WaveGen Cover Image" style="max-width: calc(50% - 10px); height: auto;">
    <img src="./static/images/ControlWave_cover.jpg" alt="ControlWave Cover Image" style="max-width: calc(50% - 10px); height: auto;">
</div>
-->
</details>

<script>
// Array of rotating summary texts
const summaries = [
  "TLDR (1/15): Encode all information to the boundary. By flattening the world into a two-dimensional space, visual generation becomes almost lossless world generation.",
  "TLDR (2/15): A World Snapshot Model turns visual generation into rich-information generation across RGB, geometry, motion, pose, physics, materials, semantics, audio, and other visualizable information.",
  "TLDR (3/15): An ideal visual model requires spatial and temporal structure, dense representations, semantic instances, physical constraints, bidirectional inference, and representations that remain Reasonable, Interpretable, Visualizable, and Editable.",
  "TLDR (4/15): Generation, estimation, conditional control, and reconstruction are four placements of evidence and target inside one generative system.",
  "TLDR (5/15): TunnelingNet lets thousands of compact task routes learn independently and in parallel while sharing one visual generator.",
  "TLDR (6/15): Conformal symmetry and relative position give visual representations greater geometric freedom, robustness, and compatibility across resolution and scale.",
  "TLDR (7/15): Increasing the information carried by WSG while constraining the learned distribution improves semantic consistency across visual tasks.",
  "TLDR (8/15): Like normal and red-green color-blind observers describing the same object, task routes need not share identical internal perception to use the same generator.",
  "TLDR (9/15): A moving camera provides essential evidence about the world; changing viewpoint queries a higher-dimensional visual manifold rather than one fixed image plane.",
  "TLDR (10/15): Explicit geometry, motion, material, support, and physical-property targets create a path toward visual models that reason about how the world can evolve.",
  "TLDR (11/15): A visual task is defined jointly by its input evidence and requested output target; changing either side defines a different task.",
  "TLDR (12/15): Task tunnels are separate for efficient training and distribution, but they can be registered together as one unified model at deployment.",
  "TLDR (13/15): Connecting trained routes creates Chains of Visual Thought, where one generated World Snapshot becomes evidence for the next inference step.",
  "TLDR (14/15): Estimation derives World Snapshots from images, conditional generation fixes part of a World Snapshot, and reconstruction recovers hidden snapshots from observations.",
  "TLDR (15/15): WSM is compatible with 1,477 established visual tasks and creates 655–1,056 new ones, unifying and expanding Computer Vision in one generative framework."
];

// Get the summary element
const summaryElement = document.getElementById('rotating-summary');

// Current index
let currentIndex = 0;

// Simple rotation function
function rotateSummary() {
  currentIndex = (currentIndex + 1) % summaries.length;
  summaryElement.textContent = summaries[currentIndex];
}

// Set rotation interval (every 5.5 seconds)
setInterval(rotateSummary, 5500);
</script>

<!--
<video autoplay muted loop playsinline>
    <source src="./static/videos/teaser.mp4" type="video/mp4">
</video>
<img src="./static/videos/srvae_generations_y12x5.png" alt="WSM Cover Image" style="width: calc(100% - 10px); height: auto;">

<center>
<b>This picture is a visualization of the training data for the part of information that our model can generate, estimate, control, and Reconstruct:</b>
Our model can control hundreds of information conditions at the pixel level and utilize them for more tasks (Note that a modal can have multiple information conditions).
Due to space limitations, we only present demonstrations for 60 information conditions.
We also provide <a href="#demos">more demonstrations</a> and the corresponding <a href="https://github.com/World-Snapshot/The-results-of-Augustus">task lists</a> supported by these information conditions.
</center>
-->

<img src="./assets/wsg_task_graph_v3.png" alt="WSG task graph and the four generative interfaces" style="width: calc(100% - 10px); height: auto;">

<center>
<b>WSG task graph and the four generative interfaces.</b>
RGB is the center of the visual task space. WSG modalities surround RGB as target or condition nodes, and each directed arrow is a trainable task route.
</center>
