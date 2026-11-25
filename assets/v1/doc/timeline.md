## Timeline

This timeline summarizes preserved, timestamped development records for the Unified Law for Visual Tasks (The Four Laws of Computer Vision), WSMs, ControlWorld, and related work.

Some historical commit evidence is retained in private, read-only repositories after the public repositories were reorganized into clean releases. Repository-record links may therefore be inaccessible to public visitors; links labeled **Archived record** specifically point to renamed private-history repositories. The original repositories, commit objects, timestamps, branches, and verified offline archives are preserved by the authors and Rutgers and can be provided for formal verification when needed. Links labeled **Current link** continue to point to public materials whenever available.

**What this timeline establishes.** The preserved records show a continuous progression from the first representation ideas to the present system. The dates below mark the earliest records we have verified for each idea; later dates mark implementation or formalization, not the origin of the idea.

- **Replace display pixels with World Pixels** ([December 20--24, 2024](https://www.overleaf.com/read/zqypdthzctwx#094edf)). We distinguished pixels as units of display from the richer units needed to represent the world, then defined World Pixels, WSG, and WSV to carry spatial, physical, semantic, pose, audio, and other information.
- **Treat text-to-WSG as text-to-world, with one generator producing many kinds of World Information** ([December 23, 2024](https://github.com/World-Snapshot/doc/commit/1e84a6a1dd2c6fb169eb6f5855db795eb4fdbe29)). The original record explicitly described one model producing dozens or hundreds of output properties, replacing many separate encoders and modality-fusion routes.
- **Separate RGB appearance from other World Information so that visual attributes can be edited and recombined** ([December 23, 2024](https://github.com/World-Snapshot/doc/commit/1e84a6a1dd2c6fb169eb6f5855db795eb4fdbe29)). The same record proposed exchanging RGB appearance while preserving pose information to perform pixel-level motion and style transfer.
- **Unify computer vision through shared representations and generation across 2D, 3D, and 4D** ([January 17, 2025](https://github.com/World-Snapshot/doc/commit/14232e666b7171a8b2dd33d50d7adce1b9283e08)). We explicitly argued that computer vision should be completely unified through standardized basic formats and unified generation across images, video, scenes, and dynamic scenes.
- **Extend existing pretrained generators instead of rebuilding every capability from scratch** ([February 7, 2025](https://github.com/World-Snapshot/doc/commit/ea5e63879cdbac161faafbb6c8cb92f33a723d9a)). The record proposed fine-tuning existing generators on WSG to acquire additional physical and world attributes while retaining their original generative capabilities.
- **Understand RGB itself as a native conditional-control interface** (conceived [April 9, 2025](https://github.com/World-Snapshot/doc/commit/007af418b71eb5ac0f590be15526e87432eaccdd), recorded April 12). We identified RGB's three channels as three forms of conditional control and connected parameter-based control with channel-based control through a common interpretation.
- **Express all visual tasks as generative forms** ([October 25, 2025](https://github.com/World-Snapshot/WSMs-private-history/commit/58b35510decdf7b2cc7cd19f63a82258ff8ddcb7)). The WSM record stated this principle directly and presented encoding information into images as the methodology for unifying visual tasks.
- **Compose these visual transformations into a visual chain of thought** ([October 30, 2025](https://github.com/World-Snapshot/WSMs-private-history/commit/5cdf20e8a70cfd60a38c6e5ccdf1c0fb285af187)). The project record explicitly added Chain of Thought to WSM months before the present system and demonstrations were completed.
- **Redirect a frozen generator toward new capabilities through sparsely activated TunnelingNets** ([May--June 2026](https://github.com/World-Snapshot/WSMs-private-history/commit/22bfc4a0044f85cca25a15efd35aad4b7be11692)). This later engineering stage turned the earlier representation and generation principles into a scalable training method, culminating in [TunnelingNet](https://github.com/World-Snapshot/WSMs-private-history/commit/f6ade3b108019f4a3f0c9f08ff2bd2a170c9bf34) and the four concrete interfaces.

<button id="timeline-toggle" class="timeline-toggle" type="button" aria-expanded="false">Show details</button>

<script>
(() => {
  const button = document.getElementById('timeline-toggle');
  const viewer = button && button.closest('.markdown-section');
  if (!button || !viewer) return;

  function setCollapsed(collapsed) {
    viewer.classList.toggle('timeline-details-collapsed', collapsed);
    button.textContent = collapsed ? 'Show details' : 'Hide details';
    button.setAttribute('aria-expanded', String(!collapsed));
  }

  button.addEventListener('click', () => {
    setCollapsed(!viewer.classList.contains('timeline-details-collapsed'));
  });
  setCollapsed(true);
})();
</script>

- **2024 Dec 20:** [About 2:38 a.m., BOOM! The ideas for the project began to emerge continuously.](https://world-snapshot.github.io/doc/index.html?page=S1_preface/01_song.md)
  - Description: This marks the origin moment of the World Snapshot idea. The preface records the thought process from processing sign language data to asking whether models could be trained on the world itself, leading to the "BOOM!" moment where the project direction began to form. The first public record is later than the stated Dec 20 event time; private chat records can be provided to lock the more precise earlier timestamp.
  - First public record: 2024-12-23T13:46:21-05:00, https://github.com/World-Snapshot/doc/commit/46c71f70343100373b668a499599e9aecfc29956
  - Current link: https://world-snapshot.github.io/doc/index.html?page=S1_preface/01_song.md

- **2024 Dec 21:** [About 4:27-5:05 a.m., "The Song of Computer Vision" completed.](https://world-snapshot.github.io/doc/index.html?page=S1_preface/01_song.md)
  - Description: This entry records the early methodology through "The Song of Computer Vision": store multimodal and world information in basic visual units, move beyond RGB-only pixels, and guide models toward richer world representations. The first public record is later than the stated Dec 21 completion time; private chat records can be provided to lock the more precise earlier timestamp.
  - First overleaf/public record: 2024-12-21T04:35-04:54, https://www.overleaf.com/read/zqypdthzctwx#094edf (Overleaf's time cannot be changed, as it is recorded directly by a third-party system. In 2024, I clearly stated that we should generate more information about world pixels.)
  - First git record: 2024-12-23T13:46:21-05:00, https://github.com/World-Snapshot/doc/commit/46c71f70343100373b668a499599e9aecfc29956
  - Current link: https://world-snapshot.github.io/doc/index.html?page=S1_preface/01_song.md

- **2024 Dec 23:** Blog ["Discuss Real World Models"](https://world-snapshot.github.io/doc/index.html?page=S5_blogs/00_world_model.md) completed.
  - Description: This blog explains why existing large models and image/video generation models are not enough to become real world models. It argues that the path should start from data structure itself, proposing World Snapshot Graphics as a way for models to learn richer world information directly.
  - First record: https://github.com/World-Snapshot/doc/commit/46c71f70343100373b668a499599e9aecfc29956
  - Current link: https://world-snapshot.github.io/doc/index.html?page=S5_blogs/00_world_model.md

- **2024 Dec 24:** Blog ["Rethink Best Papers"](https://world-snapshot.github.io/doc/index.html?page=S5_blogs/01_best_paper.md) and ["Computer Vision's Rebirth"](https://world-snapshot.github.io/doc/index.html?page=S5_blogs/02_cv_rebirth.md) completed.
  - Description: "Rethink Best Papers" frames WSG as a solution to training breadth by changing the data unit instead of only changing model depth or architecture. "Computer Vision's Rebirth" states that text-to-WSG is text-to-world, proposes one generator that can output dozens or hundreds of world properties, and separates RGB appearance from pose information for pixel-level transfer. It also expands the idea into graphics, physics, world generation, simulation, games, audio, autonomous driving, medicine, and other WS-based fields.
  - First record: https://github.com/World-Snapshot/doc/commit/fe0a24af71829aa469643980f1ff5e5ca3179663
  - Current link: https://world-snapshot.github.io/doc/index.html?page=S5_blogs/01_best_paper.md
  - First record: https://github.com/World-Snapshot/doc/commit/1e84a6a1dd2c6fb169eb6f5855db795eb4fdbe29
  - Current link: https://world-snapshot.github.io/doc/index.html?page=S5_blogs/02_cv_rebirth.md

- **2024 Dec 24:** [WSG/WSV core definitions publicly added.](https://world-snapshot.github.io/doc/index.html?page=S2_standard/01_overview.md)
  - Description: This entry records the formal definitions of World Pixels, World Snapshot Graphics, and World Snapshot Video. It defines WSG as an image-like format whose basic unit carries richer world attributes than RGB pixels, making it a direct record of the method for encoding world information into graphics.
  - First record: https://github.com/World-Snapshot/doc/commit/77b21f038558cdd1b1bb8b7b31613b26eba0e052
  - Current link: https://world-snapshot.github.io/doc/index.html?page=S2_standard/01_overview.md

- **2024 Dec 25:** Blog ["Create Laplace's Demon"](https://world-snapshot.github.io/doc/index.html?page=S5_blogs/04_laplace_demon.md) completed.
  - Description: This blog develops WSV as a computable world segment and explores time reversal, past prediction, and temporal reasoning in virtual worlds. It connects world pixels, physical information, and the possibility of predicting or reconstructing world states over time.
  - First record: https://github.com/World-Snapshot/doc/commit/bdec0dd943904c97c8d9d43c5ad7e2cf169aa362
  - Current link: https://world-snapshot.github.io/doc/index.html?page=S5_blogs/04_laplace_demon.md

- **2025 Jan 1:** Blog ["Plato's Cave Allegory"](https://world-snapshot.github.io/doc/index.html?page=S5_blogs/08_plato_cave_allegory.md) added.
  - Description: This essay uses Plato's cave as an early conceptual analogy for learning a world through observable projections, connecting image-space observations with the broader World Snapshot view of encoded world information.
  - First record: https://github.com/World-Snapshot/doc/commit/9b8f8f7e30d40398d611cda0cbd14837333547a5
  - Current link: https://world-snapshot.github.io/doc/index.html?page=S5_blogs/08_plato_cave_allegory.md

- **2025 Jan 3:** Blog ["Time Flowing in Graphics"](https://world-snapshot.github.io/doc/index.html?page=S5_blogs/03_time_in_graphics.md) and ["Relative be in All Things"](https://world-snapshot.github.io/doc/index.html?page=S5_blogs/05_relative_learning.md) completed.
  - Description: "Time Flowing in Graphics" adds temporal properties to graphics and treats time as part of the visual/world representation. "Relative be in All Things" frames relative learning as a foundation for embodied intelligence and world models, emphasizing that world understanding depends on relations learned from the world itself.
  - First record: https://github.com/World-Snapshot/doc/commit/4c4e860d0c2c523e99856dd0eda5fd25ec1e8f40
  - Current link: https://world-snapshot.github.io/doc/index.html?page=S5_blogs/03_time_in_graphics.md
  - First record: https://github.com/World-Snapshot/doc/commit/de05b03d7e808a15645eb100323648fe03c81420
  - Current link: https://world-snapshot.github.io/doc/index.html?page=S5_blogs/05_relative_learning.md

- **2025 Jan 17:** [Complete unification of computer vision was stated as an explicit objective.](https://github.com/World-Snapshot/doc/commit/14232e666b7171a8b2dd33d50d7adce1b9283e08)
  - Description: The updated project record says that computer vision should be completely unified through standardized basic formats and unified generation across 2D images and video, 3D scenes, and 4D dynamic scenes. This predates the later WSM implementation and records the field-level objective behind it.
  - First record: https://github.com/World-Snapshot/doc/commit/14232e666b7171a8b2dd33d50d7adce1b9283e08
  - Current link: https://world-snapshot.github.io/doc/index.html?page=S5_blogs/01_best_paper.md

- **2025 Jan 18:** Blog ["Pixel Grandfather Paradox"](https://world-snapshot.github.io/doc/index.html?page=S5_blogs/06_grandfather_paradox.md) completed.
  - Description: This blog introduces the World Pixel Grandfather Paradox and uses world-pixel/time-travel reasoning to discuss causal continuity in image-space and world-space generation. It connects pixel-level representation with wave-function-style generation and temporal consistency.
  - First record: https://github.com/World-Snapshot/doc/commit/9fa39b685818bb2fb60c9c820c6df3cff2a96b31
  - Current link: https://world-snapshot.github.io/doc/index.html?page=S5_blogs/06_grandfather_paradox.md

- **2025 Jan 23:** Blog ["Tensors are Basis of Light"](https://world-snapshot.github.io/doc/index.html?page=S5_blogs/07_light.md) completed.
  - Description: This blog argues that images and tensors are already a kind of world representation, because cameras sample and compress real-world visual information. It discusses light, material control, and why richer WSG channels can carry more useful world information than RGB alone.
  - First record: https://github.com/World-Snapshot/doc/commit/4a2fe6090a0fea5e920dc3dca7a074591af92d99
  - Current link: https://world-snapshot.github.io/doc/index.html?page=S5_blogs/07_light.md

- **2025 Jan 25:** [WaveGen implementation repository initialized.](https://github.com/World-Snapshot/WaveGen/commit/f879c94d76e6f555927bdcd0ad8400d982c4b0f9)
  - Description: This marks the beginning of the WaveGen implementation history. Before the later WSMs naming became dominant, this line explored the wave/core-space route for generating world-pixel outputs.
  - First record: https://github.com/World-Snapshot/WaveGen/commit/f879c94d76e6f555927bdcd0ad8400d982c4b0f9
  - Current link: https://github.com/World-Snapshot/WaveGen

- **2025 Feb 7:** Guideline ["Model Design Guideline Memo"](https://world-snapshot.github.io/doc/index.html?page=S6_materials/01_guideline.md) completed.
  - Description: This memo records the model design path after the early WSG/WSV ideas: train the world itself, make data closer to world reality through WSG, and seek one model spanning 1D, 2D, 3D, 4D, and other dimensions. The same update also proposes fine-tuning existing pretrained generators on WSG so they acquire additional physical and world attributes without discarding their original generative capabilities.
  - First record: https://github.com/World-Snapshot/doc/commit/b0dfc14cc8f22ef63f1a932ef2812953dbc7115e
  - Current link: https://world-snapshot.github.io/doc/index.html?page=S6_materials/01_guideline.md

- **2025 Mar 8:** WaveGen prototype uploaded.
  - Description: This records an early public implementation prototype for the wave-based generative route, before the later Wave2Pixel/WSM training line was separated more clearly.
  - First record: https://github.com/World-Snapshot/WaveGen/commit/8714c7aed0406598e905a2b56af2f6b4880eaf32
  - Current link: https://github.com/World-Snapshot/WaveGen

- **2025 Apr 9--12:** [RGB and WSG channels were interpreted as native conditional-control interfaces.](https://github.com/World-Snapshot/doc/commit/007af418b71eb5ac0f590be15526e87432eaccdd)
  - Description: A note conceived on April 9 and committed on April 12 identifies RGB's three color channels as three conditional controls and relates parameter-based conditioning to data-structure- and channel-based conditioning. This is the earliest verified bridge from RGB generation to the later use of RGB as the shared interface of the WSM task graph.
  - First record: https://github.com/World-Snapshot/doc/commit/007af418b71eb5ac0f590be15526e87432eaccdd
  - Current link: https://world-snapshot.github.io/doc/index.html?page=S5_blogs/01_best_paper.md

- **2025 Apr 14:** WaveGen prototype and flow-model base added.
  - Description: This records the move toward a flow-model implementation basis for WaveGen and the wave-to-pixel decoder direction. The same period's notes describe treating wave/core space as meaningful initialization rather than random pixels.
  - First record: https://github.com/World-Snapshot/WaveGen/commit/80469dc92d6b25764db4baf9d02fd8604e06ed2c
  - Current link: https://github.com/World-Snapshot/WaveGen

- **2025 May 7:** WaveGen two-way information system added.
  - Description: This records a bidirectional information-system design in WaveGen, restructuring the model code and adding encoder-side experiments. It is part of the pre-WSMs model-development period where the wave/core-space route was still being actively tried.
  - First record: https://github.com/World-Snapshot/WaveGen/commit/274349e5e1ca3a2817abbc2b423bf7a96b411c85
  - Current link: https://github.com/World-Snapshot/WaveGen

- **2025 May 21:** WaveGen 19.5 added.
  - Description: This records a later WaveGen version before the independent Wave2Pixel/WSM implementation line. The surrounding commits show continued organization and checkpoint cleanup around the wave-generation prototype.
  - First record: https://github.com/World-Snapshot/WaveGen/commit/0206e2c5dea55166562078baca7c7bb3ff44b64f
  - Current link: https://github.com/World-Snapshot/WaveGen

- **2025 May 31:** Wave2Pixel / WSMs implementation repository initialized.
  - Description: This marks the beginning of the implementation repository that was also recorded as Wave2Pixel in the project history and later appears as WSMs. The initial work records training infrastructure for WSG-related models, including auto-test support, LCoPE, and resume-training foundations.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/ea9fdab1122139be9628dd864dab1aec50dc12ca
  - Current link: https://github.com/World-Snapshot/WSMs

- **2025 Jun 4:** VAE and CLIP encoder added to the Wave2Pixel / WSMs line.
  - Description: This records early model-side exploration before the later WSG VAE and bridge attempts. It shows that the WSM/Wave2Pixel implementation was already testing encoder and VAE components shortly after repository initialization.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/147bb6ff6caef71093912448d4261c6bae26ae75
  - Current link: https://github.com/World-Snapshot/WSMs

- **2025 Jun 8:** WSG processing pipeline expanded with PrimeDepth, additional channels, and package fixes.
  - Description: This records WSG data-processing work rather than model training: extraction and formatting pipelines were expanded, more channels were added, and DWPose/PrimeDepth-related processing was improved.
  - Archived record: https://github.com/World-Snapshot/wsg-private-history/commit/a6c3bb791cfd41713668bf22c8b79080d0921c2c
  - Current link: https://github.com/World-Snapshot/wsg

- **2025 Jun 18:** DWPose injection and reading completed in WSG.
  - Description: This records a concrete WSG format workflow: DWPose information could be injected into WSG and read back, supporting the broader plan of storing task-specific visual/world information in image-like data.
  - Archived record: https://github.com/World-Snapshot/wsg-private-history/commit/efe73e8975492370f803641bc449b3809a1b3023
  - Current link: https://github.com/World-Snapshot/wsg

- **2025 Jun 22:** WaveGen decoder content cleared and Transformer version added.
  - Description: This records a WaveGen turning point before the later abandonment/renaming path. The decoder content was cleared to prepare for visualization, then a Transformer version was added, showing continued attempts to make the wave-to-pixel route trainable.
  - First record: https://github.com/World-Snapshot/WaveGen/commit/4a9c9ee08f3d6785edf8425e36e4a6f55f1b51a0
  - Current link: https://github.com/World-Snapshot/WaveGen

- **2025 Jun 27:** Relative and metric depth pipeline added to WSG.
  - Description: This records WSG data-format expansion for depth information, including relative and metric depth processing. It is part of the 2025 summer WSG pipeline work that prepared data for later model training.
  - Archived record: https://github.com/World-Snapshot/wsg-private-history/commit/5a4c1d8d2703dbc1fcaa5f3f1fcd76c2910d6cf1
  - Current link: https://github.com/World-Snapshot/wsg

- **2025 Jul 4:** SMPLerX body boxes and vertices added to WSG workflows.
  - Description: This records human-body 3D information being converted into WSG-side visual/data representations, including body bounding boxes and vertices. Nearby commits also record optimizing vertex injection and abandoning per-person camera assignment.
  - Archived record: https://github.com/World-Snapshot/wsg-private-history/commit/b0a54b3b50da1fba20c2c52f35a87d76e86f20af
  - Current link: https://github.com/World-Snapshot/wsg

- **2025 Jul 20:** Geometry features added to the WSG pipeline.
  - Description: This records additional geometry-processing support, including normals, curvature, edges, and optical flow. It expanded WSG from a narrow graphics container toward a richer processing format for many visual tasks.
  - Archived record: https://github.com/World-Snapshot/wsg-private-history/commit/35f92184588b02b3f2c2bd57834bc81e03872130
  - Current link: https://github.com/World-Snapshot/wsg

- **2025 Aug 5:** WaveGen v29 began depth-map-based training exploration.
  - Description: This records a later WaveGen development stage focused on training from depth maps. The following August commits include RealCam depth-to-pointcloud training, structured output, multi-GPU fixes, complete-frame retraining, complex-valued calculation, and several unsuccessful or unstable versions.
  - First record: https://github.com/World-Snapshot/WaveGen/commit/f7984ab149506d9784faba2cfe858df1363a751c
  - Current link: https://github.com/World-Snapshot/WaveGen

- **2025 Aug 14:** Physical-property pipeline added to WSG.
  - Description: This records the WSG pipeline17 work for physical/psychophysical property injection and reading. It further expanded WSG as a data-processing and world-attribute container before later WSM training.
  - Archived record: https://github.com/World-Snapshot/wsg-private-history/commit/29fef75c4534500b123fe3dce513956400faa01a
  - Current link: https://github.com/World-Snapshot/wsg

- **2025 Sep 12:** WaveGen optimization became incomplete, and the old wave-system route was separated as historical material.
  - Description: This records a turning point in the WaveGen route. The documentation added `old_wave_sys.md` and marked the optimization of WaveGen as not yet complete. Later notes also record that the earlier spherical-harmonic / mathematical-wave content was unsatisfactory and was abandoned, while WSM implementation increasingly moved toward practical WSG training routes.
  - First record: https://github.com/World-Snapshot/doc/commit/3d4b16802f2aa629dee225bca90313217c751523
  - Current link: https://world-snapshot.github.io/doc/index.html?page=S3_get_start/01_wave_gen.md

- **2025 Sep 12:** MediaPipe segmentation and SMPLer-X processing optimizations added to WSG.
  - Description: This records WSG processing improvements around segmentation and SMPLer-X throughput, including skip detection, vectorization, GPU acceleration, asynchronous saving, batching, and progress tracking.
  - Archived record: https://github.com/World-Snapshot/wsg-private-history/commit/d0c833bdb70f18b4f60f0bf1067aea4e33283360
  - Current link: https://github.com/World-Snapshot/wsg

- **2025 Sep 20:** Audio pipeline added to WSG.
  - Description: This records the expansion of WSG/WSV processing into audio-related data, including audio pipeline work, extraction-speed optimization, injection-logic fixes, and subtitle modifications.
  - Archived record: https://github.com/World-Snapshot/wsg-private-history/commit/f7227a4f4d88aa06806912caefe302f4e631f7e8
  - Current link: https://github.com/World-Snapshot/wsg

- **2025 Sep 24:** External paper <a class="timeline-external-paper" href="https://video-zero-shot.github.io/">"Video models are zero-shot learners and reasoners"</a> appeared.
  - Description: This paper argues that Veo 3 can solve many vision tasks without task-specific training, including segmentation, edge detection, physical-property understanding, manipulation, and visual reasoning. It is listed here to mark when a similar external public work appeared in relation to this timeline.
  - First record: https://arxiv.org/abs/2509.20328
  - Current link: https://video-zero-shot.github.io/

- **2025 Sep 24:** Optimized WSG injection/reading file formats and preview generation added.
  - Description: This records a WSG format upgrade: optimized injection and reading file formats were added, and preview images could be generated automatically during injection. This belongs to the data-format and tooling line that supported later model training.
  - Archived record: https://github.com/World-Snapshot/wsg-private-history/commit/c9ab84f4efcdbd61a5322461d3887243cef77f10
  - Current link: https://github.com/World-Snapshot/wsg

- **2025 Sep 28:** Functioning PBR material pipeline completed in WSG.
  - Description: This records material/PBR information being injected and extracted in a functioning WSG pipeline version. It shows WSG handling richer object/material properties before the later WSM training consolidation.
  - Archived record: https://github.com/World-Snapshot/wsg-private-history/commit/591c39a6724197b85d0710433e40c53494d95f04
  - Current link: https://github.com/World-Snapshot/wsg

- **2025 Sep 30:** Preliminary WSM training script completed.
  - Description: This entry records the transition from concept and documentation into executable training code. The preliminary training script became an early implementation basis for training models on WSG-style data.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/20520cdc35ef2ddf662b8de661409e42506e14bb
  - Current link: https://github.com/World-Snapshot/WSMs

- **2025 Oct 3:** WSM training updated from 253 channels to 256 channels with general mask loss.
  - Description: This records an important WSG training format correction: moving to the full 256-channel setting and adding a general mask loss. This aligns the model training code more closely with the WSG channel design.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/b52d81dadd80d43aae007efd36a788afe3702782
  - Current link: https://github.com/World-Snapshot/WSMs

- **2025 Oct 9:** LayerDiffuse-inspired VAE/decoder route was added to the WaveGen notes.
  - Description: This records the attempt to use VAE and LayerDiffuse-like ideas so generation could carry additional information beyond RGB. The notes describe freezing or expanding VAE-related components, adding shadow/offset channels, and trying to transmit extra WSG information through decoder-side mechanisms. This route was important because it exposed the difficulty of making ordinary VAE/decoder paths carry precise WSG information.
  - First record: https://github.com/World-Snapshot/doc/commit/5ba6289b75f46049d42785b8d210450c175f1757
  - Current link: https://world-snapshot.github.io/doc/index.html?page=S3_get_start/01_wave_gen.md

- **2025 Oct 15:** Hierarchical normalization method added for WSG model training.
  - Description: This entry records the addition of hierarchical normalization for WSG training. It supports more stable handling of heterogeneous WSG channels and became part of the training-side treatment of high-dimensional world-pixel information.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/07475c659e5152625e080865f08d2b35abdcdc4f
  - Current link: https://github.com/World-Snapshot/WSMs

- **2025 Oct 25:** WaveGen submodule reference renamed from wave2pixel to WSMs.
  - Description: This records more than a naming transition. The WSMs record directly states that all visual tasks are generative forms, describes encoding information into images as the unifying methodology, and presents WSM as a controllable generator of visual outputs. The same model-development line that had been recorded as Wave2Pixel was renamed toward WSMs in the WaveGen integration record.
  - First record: https://github.com/World-Snapshot/WaveGen/commit/9b9315fd94da613fef999f076d5b786b07356144
  - Archived WSM record: https://github.com/World-Snapshot/WSMs-private-history/commit/58b35510decdf7b2cc7cd19f63a82258ff8ddcb7
  - Current link: https://github.com/World-Snapshot/WSMs

- **2025 Oct 30:** [Chain of Thought was explicitly added to the WSM formulation.](https://github.com/World-Snapshot/WSMs-private-history/commit/5cdf20e8a70cfd60a38c6e5ccdf1c0fb285af187)
  - Description: The WSM comparison was extended with a Chain-of-Thought capability, recording the idea that visual transformations could be composed as an internal chain months before the current visual-chain demonstrations and interfaces were completed.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/5cdf20e8a70cfd60a38c6e5ccdf1c0fb285af187
  - Current link: https://github.com/World-Snapshot/WSMs

- **2025 Nov 14:** VAE encoder/decoder route showed clear limits.
  - Description: This records an experimental failure around using encoder weights and additional channel initialization to make the model carry extra WSG information. The result did not exceed the expected limit, and nearby records show that even relatively consistent outputs were not easy to encode reliably.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/e5532aa51e2bb52ce6cd7372a71c43d8c576cdf4
  - Current link: https://github.com/World-Snapshot/WSMs

- **2025 Nov 16:** Complete VAE restoration attempt failed.
  - Description: This records a later VAE/decoder attempt, where the Scheme E model tried a more complete restoration but failed. It belongs to the same exploration line: using VAE-style reconstruction and decoder offsets to preserve or generate additional WSG information.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/7de1bee39ff75d29ed302906170531a1e5596556
  - Current link: https://github.com/World-Snapshot/WSMs

- **2026 Jan 20:** RF bridge repair training attempt failed.
  - Description: This records the bridge route: separately encode WSG/modal information into latent spaces, then use bridge/repair models to connect or translate between those spaces. After multiple steps of RF bridge repair training, the process required a large amount of memory and later produced black outputs, showing that this route was not a practical solution at that stage.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/8288645aae601774d211f947a7e91109c3a1d87a
  - Current link: https://github.com/World-Snapshot/WSMs

- **2026 Jan 25:** Bridge training refined and latent debug tools added.
  - Description: This records follow-up bridge work after the failed RF repair route, including latent debug tools and refinements around codebook/latent-pair handling. It supports the record that the bridge idea was not a single one-off attempt but a branch-line exploration.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/2dd5ff5db128a7fcec43d1ca720091ad6b59a297
  - Current link: https://github.com/World-Snapshot/WSMs

- **2026 Feb 5:** VQ-Bridge optimization and codebook-utilization experiments added.
  - Description: This records another bridge branch, trying VQ/codebook-based bridge training and utilization tracking. The next record shows that the VQ bridge MLP and index-MSE attempt failed.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/84e88ab0de6a70e060f090d9168e0cb172dad100
  - Current link: https://github.com/World-Snapshot/WSMs

- **2026 Feb 6:** Transformer-style WSG VAE route also failed.
  - Description: This records that a transformer-based attempt also failed, suggesting that the issue was not only a single architecture choice but also related to the distribution and representation problem of high-dimensional WSG information.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/18c6bcf85535e107cff54b4f9164eb95482c137c
  - Current link: https://github.com/World-Snapshot/WSMs

- **2026 Mar 9:** Prior-posterior / encoder-KL bridge strategy was recorded as not very useful.
  - Description: This records the `ce-mode-bridge` branch line, including prior-posterior, encoder KL distribution, and best-of-k strategies. The commit message records that these strategies were not very useful.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/42bb14e3f51d692bc775a723721f021337d07e91
  - Current link: https://github.com/World-Snapshot/WSMs

- **2026 Mar 13:** Wan2.1 video-model route was imported for WSG experiments.
  - Description: This records the attempt to use a video-generation-style route based on Wan2.1. It was part of exploring whether video model infrastructure could carry WSG/WSV generation and extra information, but later development moved away from this older path.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/1366d02ac10866f12b5e0c78c97d869e4d5070cf
  - Current link: https://github.com/World-Snapshot/WSMs

- **2026 Mar 21:** LTX / deep visualization and training attempt failed.
  - Description: This records another video-model/VAE bridge exploration. The deep visualization and training attempt failed, although the record notes this did not necessarily mean the LTX direction itself was ineffective. It is included to show that several practical routes were tested before the later WSM_v1/TunnelingNet direction.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/ab621afdb69c4721a860a8e454c3ce47d7c4f1b8
  - Current link: https://github.com/World-Snapshot/WSMs

- **2026 Apr 22:** External paper <a class="timeline-external-paper" href="https://arxiv.org/abs/2604.20329">"Image Generators are Generalist Vision Learners"</a> appeared.
  - Description: This paper introduces Vision Banana and argues that image generation pretraining can become a generalist vision learner. It is listed here to mark when this similar external public work appeared in relation to this timeline.
  - First record: https://arxiv.org/abs/2604.20329
  - Current link: https://arxiv.org/abs/2604.20329

- **2026 May 4:** WSM_v1 files added to the Wan LoRA branch.
  - Description: This records the start of the WSM_v1 implementation path in the model repository, moving WSG/WSV training into a larger generative-model training framework and preparing the later FLUX/Wan/TunnelingNet experiments.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/22bfc4a0044f85cca25a15efd35aad4b7be11692
  - Current link: https://github.com/World-Snapshot/WSMs/tree/main/WSM_v1

- **2026 May 6:** External paper <a class="timeline-external-paper" href="https://arxiv.org/abs/2605.04566">"Open-Source Image Editing Models Are Zero-Shot Vision Learners"</a> first appeared.
  - Description: This paper showed that open-source image-editing models can perform depth, surface-normal, and semantic-segmentation tasks without task-specific fine-tuning.
  - First record: https://arxiv.org/abs/2605.04566v1
  - Current link: https://arxiv.org/abs/2605.04566

- **2026 May 10:** UniVidX route was imported and aligned for WSM experiments.
  - Description: This records the attempt to use a UniVidX-style video-generation/intrinsic-video route for WSM training. It followed the Wan2.1/LTX explorations and was another attempt to adapt video-generation infrastructure for WSG/WSV information generation, before later code cleanup and the standalone TunnelingNet direction.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/ce5bba48c2452ce83d31e700080259909429835c
  - Current link: https://github.com/World-Snapshot/WSMs/tree/main/WSM_v1

- **2026 Jun 14:** Dense large-scale WSG training baseline marked successful.
  - Description: This entry records a successful dense large-scale training baseline. It is an implementation milestone showing that WSG dense training could move beyond isolated tests into a larger training path.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/412b8379439b8f8752820665cd8fb9e35a77ec38
  - Current link: https://github.com/World-Snapshot/WSMs/tree/main/WSM_v1

- **2026 Jun 22:** FLUX2 WSG training added with train/validation split and validation loss tracking.
  - Description: This records the FLUX2 training path for WSG, including train/validation split support and validation loss tracking. It also led into metadata support, WSG reader path fixes, and true inference visualization.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/f22690923bc30d839b05186dfc95f54f0183257d
  - Current link: https://github.com/World-Snapshot/WSMs/tree/main/WSM_v1

- **2026 Jun 27:** [TunnelingNet made standalone and Channel Head naming adopted.](https://github.com/World-Snapshot/WSMs/blob/main/WSM_v1/scripts/train_TunnelingNet_Flux2_4B_wsg.py)
  - Description: This entry records the consolidation of the FLUX-4B WSG training route. Dense adapters were renamed to Channel Head, frozen FLUX modules were offloaded, and TunnelingNet was separated into a standalone training path.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/f6ade3b108019f4a3f0c9f08ff2bd2a170c9bf34
  - Current link: https://github.com/World-Snapshot/WSMs/blob/main/WSM_v1/scripts/train_TunnelingNet_Flux2_4B_wsg.py

- **2026 Jul 3:** Channel-aware WSG reader visualizations added.
  - Description: This entry records the addition of channel-aware visualization support for WSG reader workflows. It helps inspect model behavior across WSG channels rather than only through RGB-style previews.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/568a0e7a33a24cd0b2f11daff5500961090e9c5b
  - Current link: https://github.com/World-Snapshot/WSMs/tree/main/WSM_v1

- **2026 Jul 9:** Sparse audio Channel Head adapters and sparse structure training pipeline added.
  - Description: This entry records the sparse-side expansion of WSM training. Sparse audio Channel Head adapters were added, and the sparse structure training pipeline was updated for WSV/WSG structure tasks.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/b1abf447b98cb36c9f28771b8c97d28de8b4d81e
  - Current link: https://github.com/World-Snapshot/WSMs/blob/main/WSM_v1/third_party/ScoreHMR/wsm_adapters/train_wsv_audio.py

- **2026 Jul 11:** Sparse structure training stabilized.
  - Description: This records stabilization of the sparse structure training path after the audio and structure updates. It consolidated changes across sparse configuration, training launch scripts, audio dataset handling, MusicGen LoRA training, and WSV structure adapters.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/006d25f81818ae4ea978453cac1acc9c23187638
  - Current link: https://github.com/World-Snapshot/WSMs/tree/main/WSM_v1

- **2026 Jul 13:** WSMs README updated with the current public formulation.
  - Description: This README version states the WSM methodology as "Encode Information into Images" and explains the relation between WSMs, WSG data, WaveGen, training stages, dense/sparse domains, and the broader goal of unifying visual tasks through WSG-style training.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/8d81e0350ff5c7c3c1ec75a4de89464279ae3ce1
  - Current link: https://github.com/World-Snapshot/WSMs

- **2026 Jul 14:** Sparse training defaults and AVIS reindex tools updated.
  - Description: This entry records the latest synchronized WSMs update. It adds AVIS WSV/cache reindex repair tools, semantic alignment spot-check visualization, sparse training default updates, and cache repair utilities for channel 195 and related AVIS records.
  - Archived record: https://github.com/World-Snapshot/WSMs-private-history/commit/a02d9d312f7563467afaed65e42e9df36197d24e
  - Current link: https://github.com/World-Snapshot/WSMs/tree/main/WSM_v1/manual_processing/flux2_preprocessing/cache

- **2026 Jul 14:** External paper <a class="timeline-external-paper" href="https://arxiv.org/abs/2607.12450">"Let RGB Be the Language of Vision"</a> first appeared.
  - Description: This paper formulated diverse visual inputs and outputs as RGB and converted general visual tasks into a shared RGB-to-RGB image-editing problem.
  - First record: https://arxiv.org/abs/2607.12450v1
  - Current link: https://arxiv.org/abs/2607.12450

- **2026 Nov 18:** <a class="timeline-release-milestone" href="https://github.com/World-Snapshot/WSMs/commit/989b890d">WSMs was formally released to the public.</a>
  - Description: The public release made the WSM paper, code, data, documentation, WebUI, training states, evaluation tools, and training records available as one open research project.
  - First public release record: https://github.com/World-Snapshot/WSMs/commit/989b890d
  - Current paper: https://arxiv.org/abs/2611.24917
  - Current project: https://world-snapshot.github.io/

- **2026 Nov 20:** [The World Snapshot documentation and its nine historical essays were formally released as a public documentation collection.](https://world-snapshot.github.io/doc/index.html)
  - Description: The public documentation preserves the early conceptual record behind World Snapshot, including the nine essays written from December 2024 through January 2025, alongside standards, implementation guidance, and project materials.
  - Release record: https://github.com/World-Snapshot/doc/commit/07c7d91b4c51541f99b86f05d429ee7123e43c61
  - Current link: https://world-snapshot.github.io/doc/index.html
