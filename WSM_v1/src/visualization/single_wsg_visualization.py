"""
Single WSG Visualization - Run all available pipelines on a single WSG file
Supports any standard 256-channel WSG file
Based on wsg_visualization_all_modes.py but simplified for single file processing
"""

import os
import re
import sys
import textwrap
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import importlib.util


_PIPELINE_MODULE_CACHE = {}
_SIGNAL_EPS = 1e-6
_CHANNEL_OUTPUT_PATTERN = re.compile(
    r"^channel(?P<start>\d+)(?:_(?P<end>\d+))?_(?P<detail>.+)$"
)
_OUTPUT_VARIANTS = {
    "overlay_labeled": "Overlay labeled",
    "region_ids_labeled": "Region IDs labeled",
    "value_ids": "Value IDs",
    "region_ids": "Region IDs",
    "overlay": "Overlay",
    "heatmap": "Heatmap",
    "rgb": "RGB",
    "map": "Map",
}


# Add wsg_reader path (use absolute path so later chdir calls don't break imports).
# WSM_v1 now keeps the WSG readers local so training/debug visualizers are self-contained inside this repo.
wsg_reader_path = str(Path(__file__).resolve().parents[2] / 'wsg_reader')
if wsg_reader_path not in sys.path:
    sys.path.insert(0, wsg_reader_path)


def load_wsg_array_for_filter(wsg_file):
    """Load only the WSG array needed for cheap pipeline gating."""
    try:
        data = np.load(wsg_file, allow_pickle=True)
        try:
            wsg_data = data.get('data', data.get('wsg_data'))
            if wsg_data is None:
                return None
            return np.asarray(wsg_data)
        finally:
            data.close()
    except Exception as e:
        print(f"  ⚠️ Failed to load WSG for channel filtering: {e}")
        return None


def channel_has_signal(wsg_array, channel):
    """Return True when a channel contains real content instead of empty filler."""
    if wsg_array is None or wsg_array.ndim < 3:
        return False
    if channel < 0 or channel >= int(wsg_array.shape[-1]):
        return False

    values = np.asarray(wsg_array[..., channel])
    finite = np.isfinite(values)
    if not finite.any():
        return False

    valid = values[finite].astype(np.float32, copy=False)
    if valid.size == 0:
        return False

    # Generated/training pseudo WSG files usually leave absent lanes as all zero.
    # Several sparse readers also use all -1 as "invalid/no observation"; that
    # should not trigger an expensive modality pipeline by itself.
    if np.nanmax(np.abs(valid)) <= _SIGNAL_EPS:
        return False
    if np.nanmax(np.abs(valid + 1.0)) <= _SIGNAL_EPS:
        return False
    return True


def any_channel_has_signal(wsg_array, channels):
    return any(channel_has_signal(wsg_array, channel) for channel in channels)


def pipeline_should_run(wsg_array, required_channels, force_all=False):
    if force_all or required_channels is None:
        return True
    return any_channel_has_signal(wsg_array, required_channels)


def semantic_output_label(pipeline_name, output_path, fallback=None):
    """Create a summary title from a channel-specific reader filename."""
    stem = Path(output_path).stem
    channel_offset = stem.rfind("_channel")
    channel_segment = stem[channel_offset + 1:] if channel_offset >= 0 else stem
    match = _CHANNEL_OUTPUT_PATTERN.match(channel_segment)
    if not match:
        return fallback or pipeline_name

    start = match.group("start")
    end = match.group("end")
    detail = match.group("detail")
    variant = None
    for suffix, label in sorted(_OUTPUT_VARIANTS.items(), key=lambda item: -len(item[0])):
        if detail == suffix:
            detail = ""
            variant = label
            break
        if detail.endswith(f"_{suffix}"):
            detail = detail[:-(len(suffix) + 1)]
            variant = label
            break

    channel_label = f"{start}-{end}" if end else start
    parts = [channel_label]
    if detail:
        detail_label = detail.replace("_", " ").title()
        detail_label = (
            detail_label.replace(" Id", " ID")
            .replace(" Rgb", " RGB")
            .replace(" Wsg", " WSG")
        )
        parts.append(detail_label)
    if variant:
        parts.append(variant)
    return " · ".join(parts)


def pipeline_specs():
    """Pipeline registry plus the WSG lanes that should trigger each reader."""
    return [
        # Extraction helpers.
        ('RGB_Extraction', None, None, list(range(0, 3))),
        ('RGBA_Extraction', None, None, [3]),

        # Human-centric pipelines.
        ('pipeline01_read_dwpose_from_wsg.py', 'DWPose', ['vis', 'vis_overlay', 'save'], list(range(64, 72))),
        ('pipeline02_read_smplerx_from_wsg.py', 'SMPLer-X', ['save', 'save_light', 'vis_silhouette', 'vis_stored', 'vis_overlay', 'vis_3d_silhouette', 'vis_3d_overlay', 'original'], list(range(72, 95))),
        ('pipeline03_read_mediapipe_from_wsg.py', 'MediaPipe', ['vis', 'vis_overlay', 'original', 'save', 'json'], list(range(95, 103))),
        ('pipeline04c_read_mediapipe_segmentation_from_wsg.py', 'MediaPipe_Seg', ['vis', 'vis_overlay', 'save'], list(range(103, 111))),

        # Depth and geometry pipelines.
        ('pipeline05_read_primedepth_from_wsg.py', 'PrimeDepth', ['vis', 'vis_overlay', 'save'], [32, 33]),
        ('pipeline06_read_metricdepth_from_wsg.py', 'MetricDepth', ['vis', 'vis_overlay', 'save'], [34, 35]),
        ('pipeline07_read_normals_from_wsg.py', 'Normals', ['vis', 'vis_overlay', 'save'], [36, 37, 38]),
        ('pipeline08_read_curvature_from_wsg.py', 'Curvature', ['vis', 'vis_overlay', 'save'], [39]),
        ('pipeline09_read_edge_from_wsg.py', 'Edge', ['vis', 'vis_overlay', 'save'], [40]),
        ('pipeline10_read_opticalflow_from_wsg.py', 'OpticalFlow', ['vis', 'vis_overlay', 'save'], [41, 42]),
        ('pipeline14_read_xpose_pose_visualization_from_wsg.py', 'XPose_Pose', ['vis', 'vis_overlay', 'save'], [122, 123, 124]),

        # Object and physics pipelines.
        ('pipeline15_read_object_from_wsg.py', 'Objects', ['all'], list(range(111, 122))),
        ('pipeline16_read_psy_from_wsg.py', 'Physics', ['all'], list(range(126, 143))),

        # Material and lighting pipelines. Do not include 36-38 here; those are
        # also surface normals, but normal-only pseudo WSG should not trigger PBR.
        ('pipeline17_read_PBR_from_wsg.py', 'PBR', ['vis', 'vis_overlay', 'save', 'npz'], list(range(143, 175))),

        # Audio, semantic and wireless sensing pipelines.
        ('pipeline18_read_audio_from_wsg.py', 'Audio', ['wav', 'vis', 'sem', 'npz'], list(range(175, 185)) + [195]),
        ('pipeline19_read_caption_audio_from_wsg.py', 'Caption_Audio', ['wav', 'vis', 'npz'], list(range(185, 195))),
        ('pipeline20_read_MMFi_from_wsg.py', 'MMFi', ['vis', 'save', 'npz'], list(range(207, 236))),
    ]


def extract_rgb_from_wsg(wsg_file, output_dir):
    """Extract RGB from WSG file for visualization"""
    try:
        data = np.load(wsg_file)
        wsg_data = data.get('data', data.get('wsg_data'))
        
        if wsg_data is None:
            return None
            
        # Extract RGB (first 3 channels)
        rgb = wsg_data[:, :, :3]
        
        # Ensure RGB is in proper range [0, 255]
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            
        # Save as image
        output_name = Path(wsg_file).stem + "_rgb.png"
        output_path = output_dir / output_name
        Image.fromarray(rgb).save(output_path)
        
        data.close()
        return str(output_path)
    except Exception as e:
        print(f"    Failed to extract RGB: {e}")
        return None


def extract_rgba_from_wsg(wsg_file, output_dir):
    """Extract RGBA from WSG file for visualization (RGB + alpha channel)"""
    try:
        data = np.load(wsg_file)
        wsg_data = data.get('data', data.get('wsg_data'))
        
        if wsg_data is None:
            return None
            
        # Extract RGBA (first 4 channels)
        if wsg_data.shape[2] < 4:
            return None  # Not enough channels for RGBA
            
        rgba = wsg_data[:, :, :4]
        
        # Ensure RGBA is in proper range [0, 255]
        if rgba.max() <= 1.0:
            rgba = (rgba * 255).astype(np.uint8)
        else:
            rgba = np.clip(rgba, 0, 255).astype(np.uint8)
            
        # Save as image with alpha channel
        output_name = Path(wsg_file).stem + "_rgba.png"
        output_path = output_dir / output_name
        Image.fromarray(rgba, mode='RGBA').save(output_path)
        
        data.close()
        return str(output_path)
    except Exception as e:
        print(f"    Failed to extract RGBA: {e}")
        return None


def process_single_pipeline(wsg_file, pipeline_script, output_dir, pipeline_name, modes):
    """Process a single pipeline with specified modes using module import method"""
    outputs = []
    pipeline_path = Path(wsg_reader_path) / pipeline_script
    
    if not pipeline_path.exists():
        print(f"    ⚠️ Pipeline script not found: {pipeline_path}")
        return outputs
    
    # Create a temporary directory for pipeline output
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output = Path(temp_dir)
        
        try:
            cache_key = str(pipeline_path.resolve())
            module = _PIPELINE_MODULE_CACHE.get(cache_key)
            if module is None:
                spec = importlib.util.spec_from_file_location(
                    pipeline_path.stem, pipeline_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _PIPELINE_MODULE_CACHE[cache_key] = module
            
            # Get absolute path of WSG file
            abs_wsg_file = Path(wsg_file).resolve()
            success = False
            
            # Special handling for specific pipelines
            if pipeline_script == 'pipeline15_read_object_from_wsg.py':
                if hasattr(module, 'WSGToPNGPipeline'):
                    original_dir = os.getcwd()
                    os.chdir(wsg_reader_path)
                    try:
                        pipeline = module.WSGToPNGPipeline()
                        with np.load(abs_wsg_file, allow_pickle=False) as payload:
                            if 'wsg_data' in payload:
                                object_wsg = np.asarray(payload['wsg_data'])
                            elif 'data' in payload:
                                object_wsg = np.asarray(payload['data'])
                            else:
                                object_wsg = np.asarray(payload[payload.files[0]])
                        pipeline.export_category_id_map(
                            object_wsg, temp_output, Path(wsg_file).stem
                        )
                        success = True

                        # Full object reconstruction additionally requires the
                        # structural record-type lane. Target-only channel-111
                        # predictions intentionally contain no channels 112-125.
                        if channel_has_signal(object_wsg, 112):
                            export_dir = temp_output / "export_obj_from_wsg"
                            gltf_dir = temp_output / "gltf"
                            export_dir.mkdir(exist_ok=True)
                            gltf_dir.mkdir(exist_ok=True)
                            pipeline.process_wsg_file(
                                str(abs_wsg_file),
                                str(export_dir),
                                str(gltf_dir),
                                num_renders=1,
                                resolution=512,
                                keep_gltf=False,
                            )
                    finally:
                        os.chdir(original_dir)
                        
            elif pipeline_script == 'pipeline16_read_psy_from_wsg.py':
                # Physics scalar and category maps do not require the expensive
                # Blender and glTF reconstruction path. Pseudo WSG files used by
                # Channel Head inference are NPZ containers, so export their
                # 2D property maps directly with the lightweight reader API.
                if hasattr(module, 'WSGToPNGPipeline'):
                    pipeline = module.WSGToPNGPipeline()
                    with np.load(abs_wsg_file, allow_pickle=False) as payload:
                        if 'wsg_data' in payload:
                            wsg_array = np.asarray(payload['wsg_data'])
                        elif 'data' in payload:
                            wsg_array = np.asarray(payload['data'])
                        else:
                            wsg_array = np.asarray(payload[payload.files[0]])
                    pipeline.export_expanded_property_maps(
                        wsg_array, str(abs_wsg_file), temp_output
                    )
                    pipeline.extract_physics_from_wsg(
                        wsg_array, str(temp_output / 'physics_properties.txt')
                    )
                    success = True

            elif pipeline_script == 'pipeline18_read_audio_from_wsg.py':
                # Audio/semantic category pipeline. Use the reader's standard
                # single-file entrypoint so channel 195 semantic masks are
                # handled consistently with direct pipeline18 usage.
                if hasattr(module, 'process_single_wsg_file'):
                    success = bool(module.process_single_wsg_file((str(abs_wsg_file), str(temp_output), modes)))
                            
            elif pipeline_script == 'pipeline19_read_caption_audio_from_wsg.py':
                # Caption audio pipeline
                if hasattr(module, 'extract_caption_from_folder'):
                    # Create a temporary wsg directory
                    temp_wsg_dir = temp_output / "wsg"
                    temp_wsg_dir.mkdir(exist_ok=True)
                    
                    # Copy the wsg file
                    temp_wsg_file = temp_wsg_dir / Path(wsg_file).name
                    shutil.copy(wsg_file, temp_wsg_file)
                    
                    # Process
                    module.extract_caption_from_folder(
                        str(temp_wsg_dir), 
                        str(temp_output), 
                        modes=modes, 
                        max_files=1,
                        max_workers=1
                    )
                    success = True
                    
            elif pipeline_script == 'pipeline20_read_MMFi_from_wsg.py':
                # MMFi pipeline
                if hasattr(module, 'extract_mmfi_from_wsg'):
                    result = module.extract_mmfi_from_wsg(
                        str(abs_wsg_file),
                        str(temp_output),
                        modes=modes
                    )
                    success = bool(result)
                    
            # The current dense PBR reader exposes export_single_wsg() and its
            # CLI main() parses sys.argv, unlike the legacy main(args) readers.
            elif (
                pipeline_script == 'pipeline17_read_PBR_from_wsg.py'
                and hasattr(module, 'export_single_wsg')
            ):
                module.export_single_wsg(abs_wsg_file, temp_output, overwrite=True)
                success = True

            # Standard processing functions
            elif hasattr(module, 'process_single_wsg_file'):
                # Generic single file processor
                args = (str(abs_wsg_file), str(temp_output), modes)
                success = module.process_single_wsg_file(args)
                
            elif hasattr(module, 'process_wsg_input'):
                # Alternative processor
                module.process_wsg_input(
                    str(abs_wsg_file), 
                    str(temp_output), 
                    modes=modes, 
                    max_workers=1
                )
                success = True
                
            elif hasattr(module, 'WSGReader'):
                # WSGReader based pipelines
                reader = module.WSGReader()
                if hasattr(reader, 'extract_and_save_info'):
                    success = reader.extract_and_save_info(
                        str(abs_wsg_file), 
                        str(temp_output)
                    )
                elif hasattr(reader, 'process'):
                    success = reader.process(
                        str(abs_wsg_file), 
                        str(temp_output),
                        modes=modes
                    )
                    
            elif hasattr(module, 'main'):
                # Some pipelines have a main function
                # Create mock command line args
                class Args:
                    def __init__(self):
                        self.input = str(abs_wsg_file)
                        self.output = str(temp_output)
                        self.mode = ','.join(modes) if isinstance(modes, list) else modes
                        self.max_workers = 1
                
                module.main(Args())
                success = True
            
            # Find and copy generated files
            base_name = Path(wsg_file).stem
            
            # Look for files in various possible locations
            search_patterns = [
                "*",  # Direct outputs
                "vis_*/*",  # Visualization directories
                "export_*/*",  # Export directories
                "*/*",  # One level deep
                "visualization_analysis/*",  # Physics visualization analysis
            ]
            
            # Debug: List all files in temp directory for problem pipelines
            debug_pipelines = ["SMPLer-X", "Objects", "Physics", "PBR"]
            if pipeline_name in debug_pipelines:
                print(f"    [DEBUG] Files in temp directory for {pipeline_name}:")
                for f in temp_output.rglob("*"):
                    if f.is_file():
                        print(f"      - {f.relative_to(temp_output)}")
            
            for pattern in search_patterns:
                for file in temp_output.glob(pattern):
                    if file.is_file():
                        # Determine output type
                        if file.suffix == '.png':
                            if pipeline_name == "Audio" and 'semantic_overlay' in file.name:
                                output_type = 'semantic_overlay'
                            elif pipeline_name == "Audio" and 'semantic_mask' in file.name:
                                output_type = 'semantic_mask'
                            elif pipeline_name == "Audio" and 'semantic_rgb' in file.name:
                                output_type = 'semantic_rgb'
                            elif '3d_overlay' in file.name:
                                output_type = '3d_overlay'
                            elif '3d_silhouette' in file.name:
                                output_type = '3d_silhouette'
                            elif 'stored' in file.name:
                                output_type = 'stored'
                            elif 'original' in file.name:
                                output_type = 'original'
                            elif 'overlay' in file.name:
                                output_type = 'overlay'
                            elif 'silhouette' in file.name:
                                output_type = 'silhouette'
                            elif '3d' in file.name:
                                output_type = '3d'
                            elif 'vis_soft' in file.name:
                                output_type = 'vis_soft'
                            elif 'vis_normalcrafter' in file.name:
                                output_type = 'vis_normalcrafter'  # NormalCrafter风格
                            elif 'vis_blender' in file.name:
                                output_type = 'vis_blender'  # Blender风格光照渲染
                            elif 'vis_color' in file.name:
                                output_type = 'vis_blender'  # 旧名兼容
                            elif pipeline_name == "Normals" and 'rgb' in file.name:
                                output_type = 'vis_normalcrafter'  # 旧名兼容
                            elif 'vis_blue' in file.name or 'blue' in file.name:
                                output_type = 'vis_blue'  # 标准法线贴图（P5-P95拉伸）
                            elif 'property_progress_bars_with_references' in file.name:
                                output_type = 'vis_with_references'
                            elif 'property_progress_bars' in file.name:
                                output_type = 'vis_progress_bars'
                            else:
                                output_type = 'vis'
                        elif file.suffix in ['.obj', '.ply', '.gltf', '.glb']:
                            output_type = '3d_model'
                        elif file.suffix == '.wav':
                            output_type = 'audio'
                        elif file.suffix == '.json':
                            output_type = 'json'
                        elif file.suffix == '.npz':
                            output_type = 'npz'
                        elif file.suffix == '.txt':
                            # Special handling for Physics pipeline txt files
                            if pipeline_name == 'Physics':
                                if 'physics_properties' in file.name:
                                    output_type = 'physics_properties'
                                elif 'material_properties' in file.name:
                                    output_type = 'material_properties'
                                else:
                                    output_type = 'txt'
                            else:
                                output_type = 'txt'
                        else:
                            output_type = 'other'
                        
                        # Copy to pipeline output directory
                        pipeline_output_dir = output_dir / pipeline_name
                        pipeline_output_dir.mkdir(exist_ok=True)
                        
                        # Create unique filename - preserve original filename structure for certain files
                        if pipeline_name == "SMPLer-X" and "_light.npz" in file.name:
                            output_filename = f"{base_name}_{pipeline_name}_light.npz"
                        elif pipeline_name in {"PBR", "Physics", "Objects"}:
                            # Dense PBR and Physics exports emit many
                            # channel-specific maps. Preserve their semantic names
                            # instead of collapsing every PNG into one generic
                            # *_vis file and dropping the remaining maps as
                            # duplicates.
                            output_filename = file.name
                        else:
                            output_filename = f"{base_name}_{pipeline_name}_{output_type}{file.suffix}"
                        
                        output_path = pipeline_output_dir / output_filename
                        
                        # Reinference may intentionally reuse a label. Replace stale
                        # artifacts and always register the refreshed output.
                        shutil.copy2(file, output_path)
                        outputs.append((output_type, str(output_path)))
                        
        except Exception as e:
            print(f"    ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    return outputs


def visualize_single_wsg(wsg_path, output_dir=None, force_all=False, declared_channels=None):
    """
    Visualize a single WSG file through all available pipelines.
    
    Args:
        wsg_path: Path to the WSG file
        output_dir: Output directory (if None, creates timestamped dir)
    
    Returns:
        Path to visualization output directory
    """
    # Validate input
    wsg_path = Path(wsg_path)
    if not wsg_path.exists():
        print(f"❌ WSG file not found: {wsg_path}")
        return None
        
    # Create output directory
    if output_dir is None:
        # Use WSG filename as base for output directory
        wsg_name = wsg_path.stem  # Get filename without extension
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vis_dir = Path("vis") / f"one_stop_vis_{wsg_name}_{timestamp}"
    else:
        vis_dir = Path(output_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Single WSG Visualization ===")
    print(f"Input WSG: {wsg_path}")
    print(f"Output directory: {vis_dir.absolute()}")
    
    all_pipelines = pipeline_specs()
    wsg_array = load_wsg_array_for_filter(wsg_path)
    declared = {int(channel) for channel in (declared_channels or [])}

    def explicitly_requested(required_channels):
        return bool(
            declared and required_channels is not None
            and declared.intersection(required_channels)
        )

    selected_pipelines = [
        spec for spec in all_pipelines
        if explicitly_requested(spec[3])
        or pipeline_should_run(wsg_array, spec[3], force_all=force_all)
    ]

    # Process only pipelines whose required lanes are present. This is important
    # for training-time pseudo WSG files, which often contain only one target
    # modality and should not launch every reader.
    all_visualizations = []
    skipped = len(all_pipelines) - len(selected_pipelines)

    if force_all:
        print(f"\nProcessing all {len(all_pipelines)} pipelines (--all enabled)...")
    else:
        print(f"\nProcessing {len(selected_pipelines)} active pipelines; skipped {skipped} empty pipelines...")

    for i, (pipeline_script, display_name, modes, required_channels) in enumerate(selected_pipelines, 1):

        # Special handling for RGB and RGBA extraction
        if display_name is None:
            if pipeline_script == 'RGB_Extraction':
                print(f"\n[{i}/{len(selected_pipelines)}] Extracting RGB...")
                rgb_file = extract_rgb_from_wsg(wsg_path, vis_dir)
                if rgb_file:
                    all_visualizations.append(("RGB", rgb_file))
                    print(f"  ✓ Extracted RGB")
            elif pipeline_script == 'RGBA_Extraction':
                print(f"\n[{i}/{len(selected_pipelines)}] Extracting RGBA...")
                rgba_file = extract_rgba_from_wsg(wsg_path, vis_dir)
                if rgba_file:
                    all_visualizations.append(("RGBA", rgba_file))
                    print(f"  ✓ Extracted RGBA")
                else:
                    print(f"  ⚠️ RGBA extraction failed (not enough channels)")
            continue

        print(f"\n[{i}/{len(selected_pipelines)}] {display_name}...")
        outputs = process_single_pipeline(
            wsg_path, pipeline_script, vis_dir, display_name, modes
        )

        if outputs:
            for output_type, output_file in outputs:
                fallback = f"{display_name} · {output_type.replace('_', ' ').title()}"
                all_visualizations.append(
                    (semantic_output_label(display_name, output_file, fallback), output_file)
                )
            print(f"  ✓ Generated {len(outputs)} outputs")
        else:
            print(f"  ⚠️ No outputs generated")

    # Handle visualization_analysis directory from Physics pipeline first
    # Check if it needs to be moved to main output directory
    physics_outputs = [o for o in all_visualizations if 'Physics' in o[0]]
    for name, path in physics_outputs:
        if '_viz_analysis_dir' in name:
            # This is the visualization_analysis directory path
            viz_analysis_src = Path(path)
            if viz_analysis_src.exists() and viz_analysis_src.is_dir():
                viz_analysis_dest = vis_dir / 'visualization_analysis'
                if viz_analysis_src.resolve() == viz_analysis_dest.resolve():
                    print(f"  ✓ visualization_analysis already in output directory")
                else:
                    try:
                        if viz_analysis_dest.exists():
                            shutil.rmtree(viz_analysis_dest)
                        shutil.move(str(viz_analysis_src), str(viz_analysis_dest))
                        print(f"  ✓ Moved visualization_analysis to output directory")
                    except FileNotFoundError:
                        print(f"  ⚠️ visualization_analysis disappeared before move: {viz_analysis_src}")
    
    # Create summary visualization if we have images
    image_outputs = [(name, path) for name, path in all_visualizations 
                     if Path(path).is_file() and Path(path).suffix.lower() in ['.png', '.jpg', '.jpeg']]
    
    # Also include images from visualization_analysis directory
    viz_analysis_dir = vis_dir / 'visualization_analysis'
    if viz_analysis_dir.exists():
        for img_file in viz_analysis_dir.glob('*.png'):
            display_name = semantic_output_label("Physics", img_file, img_file.stem)
            image_outputs.append((display_name, str(img_file)))
    
    if image_outputs:
        print(f"\n📊 Creating summary visualization...")
        create_summary_grid(image_outputs, vis_dir / "summary.png")
    
    # Create output summary file
    summary_path = vis_dir / "outputs_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"WSG Visualization Summary\n")
        f.write(f"========================\n\n")
        f.write(f"Input WSG: {wsg_path}\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total outputs: {len(all_visualizations)}\n\n")
        
        f.write("Generated Files:\n")
        f.write("-" * 80 + "\n")
        
        # Group by type
        by_type = {}
        for name, path in all_visualizations:
            ext = Path(path).suffix
            if ext not in by_type:
                by_type[ext] = []
            by_type[ext].append((name, path))
        
        for ext in sorted(by_type.keys()):
            f.write(f"\n{ext.upper()} files ({len(by_type[ext])}):\n")
            for name, path in sorted(by_type[ext]):
                rel_path = Path(path).relative_to(vis_dir) if Path(path).is_relative_to(vis_dir) else Path(path).name
                f.write(f"  - {name}: {rel_path}\n")
    
    print(f"\n✅ Visualization complete!")
    print(f"   Total outputs: {len(all_visualizations)}")
    print(f"   Summary saved to: {summary_path}")
    print(f"   Output directory: {vis_dir.absolute()}")
    
    return str(vis_dir)


def create_summary_grid(image_outputs, output_path, max_cols=6):
    """Create a grid visualization of all generated images"""
    try:
        n_images = len(image_outputs)
        if n_images == 0:
            return
        
        # Calculate grid dimensions
        n_cols = min(n_images, max_cols)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        # Create figure with better spacing
        fig_width = n_cols * 3
        fig_height = n_rows * 3.5  # Extra height for titles
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        
        # Handle different array shapes
        if n_images == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes] if n_cols > 1 else [[axes]]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]
        
        # Plot images
        for idx, (name, path) in enumerate(image_outputs):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row][col] if isinstance(axes, list) else axes[row, col]
            
            try:
                img = Image.open(path)
                # Handle grayscale images properly
                if img.mode == 'L':
                    ax.imshow(img, cmap='gray')
                else:
                    ax.imshow(img)
                title = textwrap.fill(name, width=28)
                ax.set_title(title, fontsize=9, pad=3)
                ax.axis('off')
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading\n{name}", 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=8, wrap=True)
                ax.axis('off')
        
        # Hide unused axes
        for idx in range(n_images, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row][col] if isinstance(axes, list) else axes[row, col]
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ Summary grid saved to: {output_path}")
        
    except Exception as e:
        print(f"  ⚠️ Failed to create summary: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize a single WSG file through all pipelines")
    parser.add_argument("wsg_path", help="Path to the WSG file")
    parser.add_argument("--output", "-o", help="Output directory (default: vis/one_stop_vis_<filename>_<timestamp>)")
    parser.add_argument("--all", action="store_true", help="Run every reader pipeline instead of filtering by active channels")
    
    args = parser.parse_args()
    
    visualize_single_wsg(args.wsg_path, args.output, force_all=args.all)
