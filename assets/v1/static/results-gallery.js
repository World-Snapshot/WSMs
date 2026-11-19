/* WSM v1 paper-results explorer. */
(function () {
    'use strict';

    const root = './assets/v1/static/images/wsm_v1/results/';
    const assetVersion = '?v=20260905-paper-current';
    const preferredColumns = {
        qualitative: 5,
        matrices: 2,
        analysis: 4
    };
    const groups = {
        qualitative: [
            ['evaluation_four_interfaces_overview.jpg', 'Part I · Four WSM Interfaces'],
            ['evaluation_qualitative_part_ii_geometry_motion.jpg', 'Part II · Geometry and Motion'],
            ['evaluation_qualitative_part_iii_human_structure.jpg', 'Part III · Human Structure'],
            ['evaluation_qualitative_part_iv_physical_properties.jpg', 'Part IV · Physical Properties'],
            ['evaluation_qualitative_part_v_material_pbr_sheet1.jpg', 'Part V · Material and PBR (1/2)'],
            ['evaluation_qualitative_part_v_material_pbr_sheet2.jpg', 'Part V · Material and PBR (2/2)'],
            ['evaluation_qualitative_part_vi_segmentation_semantics.jpg', 'Part VI · Segmentation and Semantics'],
            ['evaluation_qualitative_part_vii_narrow_geometry_human_sheet1.jpg', 'Part VII · Geometry-conditioned RGB (1/2)'],
            ['evaluation_qualitative_part_vii_narrow_geometry_human_sheet2.jpg', 'Part VII · Motion- and Pose-conditioned RGB (2/2)'],
            ['evaluation_qualitative_part_viii_narrow_physics_semantics_material_sheet1.jpg', 'Part VIII · Physics, Semantics, and Materials (1/4)'],
            ['evaluation_qualitative_part_viii_narrow_physics_semantics_material_sheet2.jpg', 'Part VIII · Physics, Semantics, and Materials (2/4)'],
            ['evaluation_qualitative_part_viii_narrow_physics_semantics_material_sheet3.jpg', 'Part VIII · Physics, Semantics, and Materials (3/4)'],
            ['evaluation_qualitative_part_viii_narrow_physics_semantics_material_sheet4.jpg', 'Part VIII · Physics, Semantics, and Materials (4/4)'],
            ['evaluation_qualitative_part_ix_conditional_control_sheet1.jpg', 'Part IX · Conditional Control (1/6)'],
            ['evaluation_qualitative_part_ix_conditional_control_sheet2.jpg', 'Part IX · Conditional Control (2/6)'],
            ['evaluation_qualitative_part_ix_conditional_control_sheet3.jpg', 'Part IX · Conditional Control (3/6)'],
            ['evaluation_qualitative_part_ix_conditional_control_sheet4.jpg', 'Part IX · Conditional Control (4/6)'],
            ['evaluation_qualitative_part_ix_conditional_control_sheet5.jpg', 'Part IX · Conditional Control (5/6)'],
            ['evaluation_qualitative_part_ix_conditional_control_sheet6.jpg', 'Part IX · Conditional Control (6/6)'],
            ['evaluation_qualitative_part_x_inverse_rendering_sheet1.jpg', 'Part X · RGB-assisted Inverse Rendering (1/6)'],
            ['evaluation_qualitative_part_x_inverse_rendering_sheet2.jpg', 'Part X · RGB-assisted Inverse Rendering (2/6)'],
            ['evaluation_qualitative_part_x_inverse_rendering_sheet3.jpg', 'Part X · RGB-assisted Inverse Rendering (3/6)'],
            ['evaluation_qualitative_part_x_inverse_rendering_sheet4.jpg', 'Part X · RGB-assisted Inverse Rendering (4/6)'],
            ['evaluation_qualitative_part_x_inverse_rendering_sheet5.jpg', 'Part X · RGB-assisted Inverse Rendering (5/6)'],
            ['evaluation_qualitative_part_x_inverse_rendering_sheet6.jpg', 'Part X · RGB-assisted Inverse Rendering (6/6)'],
            ['evaluation_categorical_output_stages.jpg', 'Part XI · Categorical Outputs'],
            ['postrelease_qualitative_ab.jpg', 'External Comparison · Post-release Interfaces A/B'],
            ['postrelease_qualitative_c.jpg', 'External Comparison · Post-release Interface C'],
        ],
        matrices: [
            ['cvpr_complete_task_tables.jpg', 'Complete CVPR Task Tables · Parts I–V'],
            ['evaluation_categorical_output_metrics.jpg', 'Categorical Output Metrics'],
            ['evaluation_categorical_refiner_gain.jpg', 'Categorical Refiner Results']
        ],
        analysis: [
            ['ablation_validation_curves.jpg', 'Task-dependent Attention Topology'],
            ['routing_270_native_failure.jpg', 'Native-reference Routing Failure'],
            ['ablation_design_choices.jpg', 'Task-dependent Structural Choices'],
            ['ablation_objective_qualitative.jpg', 'Rectified-flow Objective Shaping'],
            ['ablation_minute_scale.jpg', 'Rapid Acquisition in the First 200 Updates'],
            ['unified_law_breadth_depth_efficiency.jpg', 'Unified-Law Breadth, Depth, and Efficiency'],
            ['pwc_historical_task_scope.jpg', 'Unifying and Expanding Computer Vision']
        ]
    };

    function initializeResultsWalls() {
        document.querySelectorAll('[data-results-wall]').forEach((wall) => {
            const category = wall.dataset.resultsWall;
            const items = groups[category];
            if (!items) return;
            wall.style.setProperty('--wall-columns', preferredColumns[category] || Math.ceil(Math.sqrt(items.length)));

            items.forEach(item => {
                const figure = document.createElement('figure');
                const image = document.createElement('img');
                const caption = document.createElement('figcaption');
                const title = document.createElement('strong');

                figure.className = 'results-wall-item';
                image.src = root + item[0] + assetVersion;
                image.alt = item[1];
                image.decoding = 'async';
                title.textContent = 'Fig. ' + item[1];
                caption.title = item[1];

                caption.appendChild(title);
                figure.appendChild(image);
                figure.appendChild(caption);
                wall.appendChild(figure);
            });
        });
    }

    function initializeResultsGalleries() {
        document.querySelectorAll('[data-results-gallery]').forEach((gallery) => {
            const category = gallery.dataset.resultsGallery;
            const items = groups[category];
            if (!items) return;

            const itemButtons = gallery.querySelector('.results-item-tabs');
            const image = gallery.querySelector('.results-stage-image');
            const title = gallery.querySelector('.results-stage-title');
            const counter = gallery.querySelector('.results-stage-counter');
            const previous = gallery.querySelector('[data-results-prev]');
            const next = gallery.querySelector('[data-results-next]');
            let index = 0;

            function renderStage() {
                const item = items[index];
                image.src = root + item[0] + assetVersion;
                image.alt = item[1];
                title.textContent = 'Fig. ' + item[1];
                counter.textContent = (index + 1) + ' / ' + items.length;
                Array.from(itemButtons.children).forEach((button, itemIndex) => {
                    button.disabled = itemIndex === index;
                });
            }

            items.forEach((item, itemIndex) => {
                const button = document.createElement('button');
                button.type = 'button';
                button.className = 'button results-item-tab';
                button.textContent = item[1];
                button.addEventListener('click', () => {
                    index = itemIndex;
                    renderStage();
                });
                itemButtons.appendChild(button);
            });

            previous.addEventListener('click', () => {
                index = (index - 1 + items.length) % items.length;
                renderStage();
            });
            next.addEventListener('click', () => {
                index = (index + 1) % items.length;
                renderStage();
            });
            renderStage();
        });
    }

    function initializeResults() {
        initializeResultsWalls();
        initializeResultsGalleries();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeResults);
    } else {
        initializeResults();
    }
}());
