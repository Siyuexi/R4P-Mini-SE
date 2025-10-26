import asyncio
from typing import Any, Dict, Optional, Tuple

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl_utils.data.envs.WS import WorkSpace


class WorkspaceManager:
    def __init__(self):
        self._workspaces: Dict[str, WorkSpace] = {}
        self._lock = asyncio.Lock()

    async def register(self, instance_id: str, workspace: WorkSpace):
        async with self._lock:
            if instance_id in self._workspaces:
                print(f"Warning: Overwriting existing workspace for instance_id '{instance_id}'")
            self._workspaces[instance_id] = workspace

    async def get(self, instance_id: str) -> Optional[WorkSpace]:
        async with self._lock:
            return self._workspaces.get(instance_id)

    async def unregister(self, instance_id: str) -> Optional[WorkSpace]:
        async with self._lock:
            return self._workspaces.pop(instance_id, None)

class PatchSubmission(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.workspace_manager = None

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: str, **kwargs) -> tuple[str, ToolResponse]:
        return await super().create(instance_id, **kwargs)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        workspace = await self.workspace_manager.get(instance_id)
        if not workspace:
            error_msg = f"Error: Could not find workspace for instance '{instance_id}'. This may indicate an internal error."
            print(error_msg)
            return ToolResponse(text=error_msg), 0.0, {}

        diff_output = await asyncio.to_thread(workspace.get_diff)
        
        return ToolResponse(text=diff_output), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        await super().release(instance_id, **kwargs)