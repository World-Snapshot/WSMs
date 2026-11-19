## Acknowledgments

We thank all our colleagues and collaborators for their valuable feedback and discussions. This work was supported in part by NSF grants, and we gratefully acknowledge the [open-source code and various resources](https://github.com/World-Snapshot/wsg) used for this project. Special thanks to the research community for its constructive feedback during the development of this work.

**Additional Declaration & To the Media:** Our work explores an ideal World Snapshot Model through the principles of conformal symmetry, encoding information onto the boundary, and the Unified Law for Visual Tasks. WSM is not another name for a video generator, VLM, robot policy, or next-state predictor. It is a generative visual framework intended to unify, expand, and ultimately transcend Computer Vision.

- Work that follows, approximates, or extends this definition in pursuit of ideal World Snapshot Models may belong to the World Snapshot Model series or family. Like deep learning emerging from machine learning, WSM establishes an independent research direction while remaining connected to the broader visual field.
- World Snapshot Graphics refers to image-like representations that encode RGB and additional world information into aligned visual channels. Implementations may use different data sources, generators, or training paths while remaining compatible with this principle.
- TunnelingNet refers to the task-acquisition method used here: task routes learn compactly and independently around a shared generator, then become one registered visual-task system at deployment.
- The process of initializing generation from meaningful multidimensional information rather than structure-free noise is called [Sen Fang Initialization](https://github.com/World-Snapshot/Sen-Fang-Initialization). It can be applied beyond the present WSM implementation.
- Previous work may contain related methods or variations, and different implementation paths may reach similar results. We hope the Unified Law, WSG, WSM, and TunnelingNet provide useful inspiration and an open foundation for further research.

**Limitations:** Because this is a non-industrial academic project with limited computing resources, WSM v1 remains an initial large-scale proof of concept rather than a production-ready system.

- Individual task Heads were usually adapted on only hundreds of examples for several hours. Uncommon samples, unseen distributions, and weakly related modality pairs can therefore remain difficult.
- The released task bank is broad rather than uniformly mature. Some categorical, semantic, physical-property, and low-evidence mappings still require stronger data, objectives, or refiners.
- The current work validates the law, methodology, training system, and a large task bank. Future versions can improve individual routes, add aligned modalities, compose deeper visual programs, and merge related routes more tightly.

This project—including its repositories, concepts, code, models, data, and evaluation—will continue to be updated and developed. Subsequent results will remain open-source or be released in article form.

**Q&As:** More information will continue to be updated in the [project documentation](https://world-snapshot.github.io/doc/index.html). If an answer does not yet exist, please submit an issue.

**Announcement:** Selected news will be published on the [World Snapshot organization page](https://github.com/World-Snapshot), with longer updates in the project documentation.
