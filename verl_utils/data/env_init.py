from pathlib import Path
from tree_sitter import Node
from tree_sitter_languages import get_parser
from collections import defaultdict
import os
import sys
import json
from tqdm import tqdm
import argparse
import math
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from verl_utils.data.envs.DB import DataBase, FunctionEntry, ClassEntry
from verl_utils.data.envs.WS import WorkSpace


def recursive_visit_python(
    root_node: Node,
    db: DataBase,
    file_path: str,
    current_entities: list,
    parent_class: ClassEntry | None = None,
    parent_function: FunctionEntry | None = None,
):
    """Recursively visit the Python AST and insert the entries into the database."""
    if root_node.type == "function_definition":
        function_name_node = root_node.child_by_field_name("name")
        if function_name_node:
            function_name = function_name_node.text.decode()
            function_entry = FunctionEntry(
                name=function_name,
                file_path=file_path,
                body=root_node.text.decode(),
                start_line=root_node.start_point[0] + 1,
                end_line=root_node.end_point[0] + 1,
            )
                
            if parent_function and parent_class:
                # determine if the function is a method of the class or a function within a function
                if (
                    parent_function.start_line >= parent_class.start_line
                    and parent_function.end_line <= parent_class.end_line
                ):
                    function_entry.parent_function = parent_function
                else:
                    function_entry.parent_class = parent_class
            elif parent_function:
                function_entry.parent_function = parent_function
            elif parent_class:
                function_entry.parent_class = parent_class
            db.insert_entry(function_entry)
            parent_function = function_entry
    elif root_node.type == "class_definition":
        class_name_node = root_node.child_by_field_name("name")
        if class_name_node:
            class_name = class_name_node.text.decode()
            class_body_node = root_node.child_by_field_name("body")
            class_methods: list[str] = []
            class_entry = ClassEntry(
                name=class_name,
                file_path=file_path,
                body=root_node.text.decode(),
                fields=[],
                methods=[],
                start_line=root_node.start_point[0] + 1,
                end_line=root_node.end_point[0] + 1,
            )
            
            if class_body_node:
                for child in class_body_node.children:
                    if child.type == "function_definition":
                        method_name_node = child.child_by_field_name("name")
                        if method_name_node:
                            parameters_node = child.child_by_field_name("parameters")
                            return_type_node = child.child_by_field_name("return_type")
                            class_method_info = method_name_node.text.decode()
                            if parameters_node:
                                class_method_info += f"{parameters_node.text.decode()}"
                            if return_type_node:
                                class_method_info += f" -> {return_type_node.text.decode()}"
                            class_methods.append(class_method_info)
            class_entry.methods = class_methods
            parent_class = class_entry
            db.insert_entry(class_entry)

    if len(root_node.children) != 0:
        for child in root_node.children:
            recursive_visit_python(
                child, 
                db, 
                file_path, 
                current_entities,
                parent_class, 
                parent_function
            )


def construct_ckg(db: DataBase, ws_path: str) -> None:
    ws_path = Path(ws_path)
    
    parser = get_parser("python")
    for file in ws_path.glob("**/*"):
        # skip hidden files and files in a hidden directory
        if (
            file.is_file()
            and not file.name.startswith(".")
            and "/." not in file.absolute().as_posix()
        ):
            extension = file.suffix
            # only consider python files
            if extension != '.py':
                continue
                
            rel_path = file.relative_to(ws_path).as_posix()
            
            # Parse file and extract entities
            current_entities = []
            tree = parser.parse(file.read_bytes())
            root_node = tree.root_node
            recursive_visit_python(
                root_node, 
                db, 
                rel_path, 
                current_entities
            )


def init(root, data, total_parts, part_idx):

    if 'r2e-gym' in data:
        filepath = f'{root}/r2e.parquet' # train set for minise
    elif 'swe-gym' in data:
        filepath = f'{root}/swe-gym.parquet' # train set for r4p
    elif 'swe-bench' in data:
        filepath = f'{root}/swe-bench-verified.parquet' # test set

    df = pd.read_parquet(filepath)
    total_lines = len(df)
    chunk_size = math.ceil(total_lines / total_parts)
    start_idx = part_idx * chunk_size
    end_idx = min((part_idx + 1) * chunk_size, total_lines)
    df = df.iloc[start_idx:end_idx]
    for idx, meta in tqdm(df.iterrows()):
        instance_id = meta['instance_id']
        base_commit = meta['base_commit']

        if os.path.exists(f'{root}/codegraph/{instance_id}.db'):
            continue
        
        ws = WorkSpace(root, f'{root}/workspace',instance_id)
        ws.init_src()
        try:
            ws.create_ws(base_commit, git=False)
        except Exception as e:
            try:
                ws.update_src()
                ws.create_ws(base_commit, git=False)
            except Exception as e:
                ws.del_ws()
                with open(f'{root}/debug.log', 'a') as logf:
                    logf.write(f"SKPI: Error when creating workspace on instance_id: {instance_id}.\nError message: {e}\n{'-'*80}\n")
                continue

        db = DataBase(root, instance_id)
        db.init_db()
        try:
            construct_ckg(db, ws.path)
        except Exception as e:
            os.system(f'rm -rf {db.path}')
            os.system(f'rm -rf {ws.path}')
            with open(f'{root}/debug.log', 'a') as logf:
                logf.write(f"SKIP: Error when constructing ckg on instance_id: {instance_id}.\nError message: {e}\n{'-'*80}\n")
        db.disconnect()
        ws.del_ws()
                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data", 
                       help="Root directory path")
    parser.add_argument("--data", type=str, default="r2e-gym", 
                       help="Dataset")
    parser.add_argument("--total_parts", type=int, default=1, 
                       help="Total number of parts to split the work into")
    parser.add_argument("--part_idx", type=int, default=0, 
                       help="Index of the part to process (0-based)")
    
    args = parser.parse_args()
    
    init(args.root, args.data, args.total_parts, args.part_idx)