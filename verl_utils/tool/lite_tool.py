import asyncio
import sqlite3
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl_utils.data.envs.WS import WorkSpace

MAX_RESPONSE_LEN: int = 16000
SNIPPET_LINES: int = 4

def remove_last_function(text):
    last_def = text.rfind('def ')
    if last_def != -1:
        last_newline = text.rfind('\n', 0, last_def)
        return text[:last_newline].rstrip()
    return text

class SearchTool(BaseTool):

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.root_dir = self.config["root_dir"]
        
        self.db_paths: Dict[str, str] = {}

        self.workspace_manager = None

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: str, id: str, sha: str, **kwargs) -> tuple[str, ToolResponse]:
        db_path = f'{self.root_dir}/codegraph/{id}.db'
        self.db_paths[instance_id] = db_path
        return await super().create(instance_id, **kwargs)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[ToolResponse, float, dict]:
        db_path = self.db_paths.get(instance_id)
        if not db_path:
            error_msg = f"No database path found for instance_id '{instance_id}'. `create` must be called first."
            print(error_msg)
            return ToolResponse(text=error_msg), 0.0, {}

        required_params = ["construct", "entity"]
        
        missing_params = [
            param for param in required_params if parameters.get(param) is None
        ]

        if missing_params:
            error_msg = f"No search was performed. The following required arguments were not provided: {', '.join(missing_params)}. Please provide all required arguments and try again."
            return ToolResponse(text=error_msg), 0.0, {}

        construct = parameters.get("construct")
        entity = parameters.get("entity")

        response = await asyncio.to_thread(
            self._blocking_execute,
            db_path,
            construct,
            entity
        )
        
        return ToolResponse(text=response), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self.db_paths:
            del self.db_paths[instance_id]
        await super().release(instance_id, **kwargs)

    def _blocking_execute(self, db_path: str, construct: str, entity: str) -> str:
        # Check availability
        if not construct:
            return "`construct` argument is empty. Please ensure the function calling format is valid."
        elif not entity:
            return "`entity` argument is empty. Please ensure the function calling format is valid."
        
        try:
            with sqlite3.connect(db_path) as db_connection:
                match construct:
                    case "function":
                        return self._search_function(db_connection, entity)
                    case "class":
                        return self._search_class(db_connection, entity)
                    case "class_method":
                        return self._search_class_method(db_connection, entity)
                    case _:
                        return f"Invalid program construct: {construct}"
        except sqlite3.OperationalError as e:
            return f"Database error: {e}. The database file at '{db_path}' might be missing or corrupted."
        except Exception as e:
            return f"An unexpected error occurred during search: {e}"

    def _search_function(self, db_connection: sqlite3.Connection, entity: str) -> str:
        entries = db_connection.execute(
            "SELECT file_path, start_line, end_line, body FROM functions WHERE name = ?", (entity,)
        ).fetchall()
        if not entries:
            msg = f"No function named `{entity}` found."
            if '.' in entity:
                return f"{msg} Please use **simple name** rather than full qualified name."
            return f"{msg} Maybe you are looking for a class method or a class?"
        output = ""
        for entry in entries:
            file_path, start_line, end_line, body = entry
            body = f"\n{body}\n"
            output += f"{file_path}:{start_line}-{end_line}\n{body}\n\n"
            if len(output) > MAX_RESPONSE_LEN:
                output = remove_last_function(output[:MAX_RESPONSE_LEN]) + "\n[response clipped due to overlong]"
                break
        return output

    def _search_class(self, db_connection: sqlite3.Connection, entity: str) -> str:
        entries = db_connection.execute(
            "SELECT file_path, start_line, end_line, fields, methods, body FROM classes WHERE name = ?", (entity,)
        ).fetchall()
        if not entries:
            msg = f"No class named `{entity}` found."
            if '.' in entity:
                return f"{msg} Please use **simple name** rather than full qualified name."
            return f"{msg} Maybe you are looking for a function or a class method?"
        output = ""
        for entry in entries:
            file_path, start_line, end_line, fields, methods, body = entry
            body = f"\n{body}\n"
            output += f"{file_path}:{start_line}-{end_line}\n"
            # output += f"Fields:\n{fields}\n" if fields else "Fields: None\n"
            output += f"Methods:\n{methods}\n" if methods else "Methods: None\n"
            output += f"Class body:\n{body}\n\n"
            if len(output) > MAX_RESPONSE_LEN:
                output = remove_last_function(output[:MAX_RESPONSE_LEN]) + "\n[response clipped due to overlong]"
                break
        return output

    def _search_class_method(self, db_connection: sqlite3.Connection, entity: str) -> str:
        entries = db_connection.execute(
            "SELECT file_path, start_line, end_line, body, class_name FROM class_methods WHERE name = ?", (entity,)
        ).fetchall()
        if not entries:
            msg = f"No class method named `{entity}` found."
            if '.' in entity:
                return f"{msg} Please use **simple name** rather than full qualified name."
            return f"{msg} Maybe you are looking for a function or a class?"
        output = ""
        for entry in entries:
            file_path, start_line, end_line, body, class_name = entry
            body = f"\n{body}\n"
            output += f"{file_path}:{start_line}-{end_line} Within class {class_name}\n{body}\n\n"
            if len(output) > MAX_RESPONSE_LEN:
                output = remove_last_function(output[:MAX_RESPONSE_LEN]) + "\n[response clipped due to overlong]"
                break
        return output

class EditTool(BaseTool):

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.root_dir = self.config["root_dir"]
        self.temp_dir = self.config["temp_dir"]

        self.workspace_manager = None
        
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir, exist_ok=True)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: str, id: str, sha: str, **kwargs) -> tuple[str, ToolResponse]:
        if await self.workspace_manager.get(instance_id):
            await self.release(instance_id)

        workspace = WorkSpace(self.root_dir, self.temp_dir, id)

        await self.workspace_manager.register(instance_id, workspace)

        await asyncio.to_thread(workspace.create_ws, sha)

        return await super().create(instance_id, **kwargs)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        workspace = await self.workspace_manager.get(instance_id)
        if workspace is None:
            error_msg = f"No workspace found for instance_id '{instance_id}'. `create` must be called first."
            print(error_msg)
            return ToolResponse(text=error_msg), 0.0, {}

        required_params = ["path", "old_str", "new_str"]
        
        missing_params = [
            param for param in required_params if parameters.get(param) is None
        ]

        if missing_params:
            error_msg = f"No edit was performed. The following required arguments were not provided: {', '.join(missing_params)}. Please provide all required arguments and try again."
            return ToolResponse(text=error_msg), 0.0, {}

        path = parameters.get("path")
        old_str = parameters.get("old_str")
        new_str = parameters.get("new_str")

        response = await asyncio.to_thread(
            self._blocking_execute,
            workspace.path,
            path,
            old_str,
            new_str
        )

        return ToolResponse(text=response), 0.0, {}
    
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        workspace = await self.workspace_manager.unregister(instance_id)
        if workspace:
            await asyncio.to_thread(workspace.del_ws)
        await super().release(instance_id, **kwargs)

    def _blocking_execute(self, workspace_path: str, path: str, old_str: str, new_str: Optional[str]) -> str:
        # get absolute path from relative path
        short_path = path
        if not path:
            return "No edit was performed. The required argument `path` is empty."
        full_path_obj = Path(workspace_path) / path
        path = str(full_path_obj)
        if not full_path_obj.exists():
            if '/' not in short_path:
                return f"No edit was performed. The file '{short_path}' does not exist. Please use a relative path (e.g., a/b/c.py) rather than a simple file name (e.g., c.py)."
            return f"No edit was performed. The file '{short_path}' does not exist."
        if not full_path_obj.is_file():
            return f"No edit was performed. The path '{short_path}' points to a directory, not a file."

        file_content = self._read_file(path).expandtabs()
        old_str = old_str.expandtabs()
        new_str = (new_str or "").expandtabs()

        # Check for occurrences, similar to the reference tool
        occurrences = file_content.count(old_str)
        if occurrences == 0:
            return f"No replacement was performed. The `old_str` was not found verbatim in {short_path}. Be mindful of whitespaces and special characters."
        elif occurrences > 1:
            file_content_lines = file_content.split("\n")
            lines = [idx + 1 for idx, line in enumerate(file_content_lines) if old_str in line]
            return f"No replacement was performed. Multiple occurrences ({occurrences}) of `old_str` found in lines {lines}. Please provide a more specific `old_str` to ensure a unique match."

        # Perform the replacement
        new_file_content = file_content.replace(old_str, new_str)

        # Find the starting and ending line numbers of the replacement for snippet context and reporting
        replacement_start_line = file_content.split(old_str)[0].count("\n") + 1
        num_new_lines = new_str.count("\n") + 1
        replacement_end_line = replacement_start_line + num_new_lines - 1
        
        modified_lines = new_file_content.split("\n")
        
        snippet_start_line_idx = max(0, replacement_start_line - 1 - SNIPPET_LINES)
        snippet_end_line_idx = min(len(modified_lines), replacement_start_line - 1 + num_new_lines + SNIPPET_LINES)
        
        snippet_lines = modified_lines[snippet_start_line_idx:snippet_end_line_idx]
        snippet = "\n".join(snippet_lines)
        snippet = f"\n{snippet}\n"

        # Check for linter errors before writing
        if path.endswith('.py'):
            original_file_name = os.path.basename(path)
            with tempfile.NamedTemporaryFile(mode='w', delete=True, prefix=original_file_name, suffix='.py', dir=workspace_path) as temp:
                temp.write(new_file_content)
                temp.flush()
                
                command_options = f"pylint {temp.name} --disable=all --enable=E0001"
                result = subprocess.run(command_options, shell=True, capture_output=True, text=True)
                
                if 'E0001' in result.stdout or 'E0001' in result.stderr:
                    error_msg = f"No edit was performed. Be careful! Your edit would introduce syntax errors!\nSpecifically, this is your intended edit:\n"
                    error_msg += f"{short_path}:{replacement_start_line}-{replacement_end_line}{snippet}\n\n"
                    error_msg += f"However, Pylint finds the following errors:\n{result.stdout + result.stderr}\n"
                    error_msg += "Thus, your edit has been aborted. Please fix these errors and retry."
                    return error_msg

        self._write_file(path, new_file_content)
        
        if len(snippet) > MAX_RESPONSE_LEN:
            snippet = remove_last_function(snippet[:MAX_RESPONSE_LEN]) + "\n[response clipped due to overlong]"

        success_msg = "The file has been successfully edited:\n"
        success_msg += f"{short_path}:{replacement_start_line}-{replacement_end_line}\n{snippet}\n\n"
        success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

        return success_msg

    def _read_file(self, path: str) -> str:
        """Reads the content of a file."""
        try:
            return Path(path).read_text()
        except Exception as e:
            print(f"Error reading file {path}: {str(e)}")
            return ""

    def _write_file(self, path: str, content: str) -> None:
        """Writes content to a file."""
        try:
            Path(path).write_text(content)
        except Exception as e:
            print(f"Error writing to file {path}: {str(e)}")