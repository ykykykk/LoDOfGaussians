import os, sys
import subprocess
import argparse
import time
import platform
import torch
from pathlib import Path
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
import train_hierarchy
from utils import view_graph_utils
import networkx as nx
import train_scaffold
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument('--project_dir', required=True, help="Only the project dir has to be specified, other directories will be set according to the ones created using generate_colmap and generate_chunks scripts. They still can be explicitly specified.")
    parser.add_argument('--env_name', default="A_LoD_of_Gaussians")
    parser.add_argument('--extra_training_args', default="", help="Additional arguments that can be passed to training scripts. Not passed to slurm yet")
    parser.add_argument('--colmap_dir', default="")
    parser.add_argument('--images_dir', default="")
    parser.add_argument('--masks_dir', default="")
    parser.add_argument('--depths_dir', default="")
    parser.add_argument('--config', default="")
    parser.add_argument('--output_dir', default="")
    parser.add_argument('--skip_if_exists', action="store_true", default=False, help="Skip coarse training if a scaffold already exists. This is determined by checking if there are any iterations in the scaffold point cloud directory.")
    parser.add_argument('--export_ply', default="", help="Write only the finest leaf Gaussians to this PLY path (no ancestor LoDs).")
    parser.add_argument('--training_backend', choices=('legacy', 'resident'), default=None,
                        help="Fine-training cache backend; overrides the JSON setting. Existing configs default to legacy.")
    parser.add_argument('--resident_version', type=int, choices=(1, 2), default=None,
                        help="Resident runtime version; overrides JSON resident_version (default: 2).")
    parser.add_argument('--seed', type=int, default=None, help="Optional RNG seed for controlled A/B runs.")
    args = parser.parse_args()

    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    model_params = model_params.extract(args)
    pipeline_params = pipeline_params.extract(args)
    print(args.extra_training_args)
    os_name = platform.system()
    f_path = Path(__file__)
    images_dir = args.images_dir if args.images_dir else os.path.join(args.project_dir, "images")
    depths_dir = args.depths_dir if args.depths_dir else (os.path.join(args.project_dir, "depths") if os.path.exists(os.path.join(args.project_dir, "depths")) else None)
    masks_dir = args.masks_dir if args.masks_dir else (os.path.join(args.project_dir, "masks") if os.path.exists(os.path.join(args.project_dir, "masks")) else None)
    colmap_dir = args.colmap_dir if args.colmap_dir else os.path.join(args.project_dir, "sparse")
    output_dir = args.output_dir if args.output_dir else os.path.join(args.project_dir, "output")
    model_params.source_path = args.project_dir
    model_params.images = images_dir
    start_time = time.time()
    model_params.model_path = os.path.join(output_dir, "scaffold")
    with open(f"configs/{args.config}", "r") as f:
        data = json.load(f)

    optimization_params = OptimizationParams(parser)
    config = argparse.Namespace(**data)
    optimization_params = optimization_params.extract(config)
    training_backend = args.training_backend or data.get("training_backend", "legacy")
    if training_backend not in ("legacy", "resident"):
        raise ValueError(f"Unknown training_backend: {training_backend}")
    training_function = train_hierarchy.training
    training_kwargs = {}
    if training_backend == "resident":
        version = args.resident_version or data.get("resident_version", 2)
        runtime = dict(data.get("resident") or {})
        if version == 2:
            from train_resident_v2 import training as training_function, validate_options
        elif version == 1:
            from dataclasses import fields
            from train_resident import training as training_function, validate_options, ResidentOptions
            allowed = {field.name for field in fields(ResidentOptions)}
            ignored = sorted(set(runtime) - allowed)
            if ignored:
                print("Resident v1 reference: v2-only settings ignored: " + ", ".join(ignored))
            runtime = {key: value for key, value in runtime.items() if key in allowed}
        else:
            raise ValueError("resident_version must be 1 or 2")
        validate_options(optimization_params, runtime)
        training_kwargs["runtime"] = runtime
        print(f"Resident runtime version: {version}")
    print(f"Fine training backend: {training_backend}")

    # Choose the scaffold that has been trained the longest.
    if args.skip_if_exists and os.path.exists(os.path.join(output_dir, "scaffold/point_cloud/")) and len(os.listdir(os.path.join(output_dir, "scaffold/point_cloud/"))) > 0:
        possible_scaffolds = os.listdir(os.path.join(output_dir, "scaffold/point_cloud/"))
        iterations = [int(s.split("_")[1]) for s in possible_scaffolds if "iteration_" in s]
        chosen_iteration = max(iterations)
        print(f"Skipping coarse training, scaffold has been trained for {chosen_iteration} iterations.")
    else:
        try:
            train_scaffold.training(
                model_params, optimization_params, pipeline_params,
                saving_iterations=[optimization_params.coarse_iterations],
                checkpoint_iterations=[], checkpoint=False, debug_from=-1)
        except subprocess.CalledProcessError as e:
            print(f"Error executing train_coarse: {e}")
            sys.exit(1)
        chosen_iteration = optimization_params.coarse_iterations

    if optimization_params.graph_view_select:
        graph_path = os.path.join(colmap_dir, "0/consistency_graph.edge_list")
        if os.path.isfile(graph_path) and False:
            view_graph_utils = nx.read_edgelist(graph_path)
            print("Read Camera Graph")
        else:
            view_graph_utils = view_graph_utils.construct_distance_graph(colmap_dir + "/0/images.txt", optimization_params.view_graph_k, optimization_params.llff_hold)
            nx.write_edgelist(view_graph_utils, graph_path)
    else:
        view_graph_utils = None

    if args.skip_if_exists and os.path.exists(os.path.join(output_dir, f"scaffold/point_cloud/iteration_{chosen_iteration}/hierarchy.dhier")):
        print(f"Skipping coarse training, scaffold has been trained for {chosen_iteration} iterations.")
    else:
        hierarchy_creator = f_path.parent / "submodules" / "gaussianhierarchy" / "build"
        hierarchy_creator /= "Release/GaussianHierarchyCreator.exe" if os_name == "Windows" else "GaussianHierarchyCreator"
        try:
            subprocess.run(
                [str(hierarchy_creator),
                 os.path.join(output_dir, f"scaffold/point_cloud/iteration_{chosen_iteration}/point_cloud.ply"),
                 os.path.join(output_dir, "../"),
                 os.path.join(output_dir, f"scaffold/point_cloud/iteration_{chosen_iteration}/"),
                 os.path.join(output_dir, f"scaffold/point_cloud/iteration_{chosen_iteration}/")],
                check=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing hierarchy_creator: {e}")
            raise

    model_params.hierarchy = os.path.join(output_dir, f"scaffold/point_cloud/iteration_{chosen_iteration}/", "hierarchy.dhier")
    model_params.scaffold_file = os.path.join(output_dir, f"scaffold/point_cloud/iteration_{chosen_iteration}/")
    model_params.output_path = output_dir
    training_function(
        model_params, optimization_params, pipeline_params,
        saving_iterations=[200000, 250000, 300000], view_graph=view_graph_utils,
        **training_kwargs)

    if args.export_ply:
        from tools.export_ply import export_hierarchy_ply
        export_hierarchy_ply(Path(output_dir) / optimization_params.output_file_name, Path(args.export_ply))
    print(f"Training finished in {time.time() - start_time:.2f} seconds.")
