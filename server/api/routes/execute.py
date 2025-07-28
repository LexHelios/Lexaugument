"""
LexOS Vibe Coder - Execute API Routes
Secure command execution with consciousness oversight
"""
import asyncio
import subprocess
import logging
import tempfile
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ..dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()

class ExecuteCommand(BaseModel):
    command: str = Field(..., description="Command to execute")
    working_directory: Optional[str] = Field(None, description="Working directory")
    timeout: int = Field(30, description="Timeout in seconds")
    environment: Optional[Dict[str, str]] = Field(None, description="Environment variables")

class ExecuteResponse(BaseModel):
    output: str
    error: str
    return_code: int
    execution_time: float
    timestamp: str

@router.post("/execute_command", response_model=ExecuteResponse)
async def execute_secure_command(
    command_request: ExecuteCommand,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ExecuteResponse:
    """
    Execute commands with consciousness oversight and security
    """
    try:
        start_time = datetime.now()
        
        # Security validation
        if not _is_command_safe(command_request.command):
            raise HTTPException(status_code=403, detail="Command not allowed for security reasons")
        
        # Prepare execution environment
        env = os.environ.copy()
        if command_request.environment:
            env.update(command_request.environment)
        
        # Execute command
        result = await _execute_command_safely(
            command=command_request.command,
            working_directory=command_request.working_directory,
            timeout=command_request.timeout,
            env=env
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return ExecuteResponse(
            output=result["output"],
            error=result["error"],
            return_code=result["return_code"],
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Command execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")

def _is_command_safe(command: str) -> bool:
    """Validate command safety"""
    dangerous_commands = [
        "rm -rf", "sudo", "su", "chmod 777", "dd if=", "mkfs",
        "fdisk", "parted", "shutdown", "reboot", "halt",
        "passwd", "useradd", "userdel", "groupadd", "groupdel"
    ]
    
    command_lower = command.lower()
    return not any(dangerous in command_lower for dangerous in dangerous_commands)

async def _execute_command_safely(
    command: str,
    working_directory: Optional[str],
    timeout: int,
    env: Dict[str, str]
) -> Dict[str, Any]:
    """Execute command with safety measures"""
    try:
        # Set working directory
        cwd = working_directory or tempfile.gettempdir()
        
        # Execute with timeout
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            return {
                "output": stdout.decode('utf-8', errors='replace'),
                "error": stderr.decode('utf-8', errors='replace'),
                "return_code": process.returncode
            }
            
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return {
                "output": "",
                "error": f"Command timed out after {timeout} seconds",
                "return_code": -1
            }
            
    except Exception as e:
        return {
            "output": "",
            "error": str(e),
            "return_code": -1
        }
