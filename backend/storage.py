"""JSON-based storage for conversations."""

import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from .config import DATA_DIR


# Conversation ids are server-generated UUID4 strings (see main.create_conversation).
# Allow-list UUIDv4 specifically: 8-4-4-4-12 hex with version=4 and RFC 4122 variant.
# This guarantees the id contains only hex digits and hyphens (no separators, "..", or NUL).
_CONVERSATION_ID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}"
)


def ensure_data_dir():
    """Ensure the data directory exists."""
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def get_conversation_path(conversation_id: str) -> str:
    """Validate the id and return a real path proven to sit inside DATA_DIR.

    Hardening follows CodeQL's recognized py/path-injection "normalize then check" form:
      1. Allow-list — the id must fully match _CONVERSATION_ID_RE (canonical UUIDv4), so
         it holds only hex digits and hyphens: no path separator, "..", or NUL byte.
      2. Normalize — os.path.realpath() resolves "../" segments and symlinks to a real path.
      3. Containment — that real path must start with the resolved DATA_DIR + os.sep.
    Raises ValueError on a malformed or out-of-tree id.
    """
    if not isinstance(conversation_id, str) or not _CONVERSATION_ID_RE.fullmatch(conversation_id):
        raise ValueError("Invalid conversation_id")
    base = os.path.realpath(DATA_DIR)
    path = os.path.realpath(os.path.join(base, f"{conversation_id}.json"))
    if not path.startswith(base + os.sep):
        raise ValueError("Invalid conversation_id")
    return path


def _open_conversation_file(conversation_id: str, mode: str):
    """Open a conversation's JSON file with the path-traversal guard CO-LOCATED with
    open(). The allow-list, realpath() normalization and startswith() containment check
    all run in THIS scope, immediately before open(). CodeQL's path-injection barrier is
    domination-based and does not carry across a function return, so the check must sit in
    the same scope as the sink to be recognized as sanitizing it (validating it once in a
    helper and returning the path does not). Raises ValueError for a malformed/out-of-tree
    id; FileNotFoundError propagates when a valid file is simply absent.
    """
    if not isinstance(conversation_id, str) or not _CONVERSATION_ID_RE.fullmatch(conversation_id):
        raise ValueError("Invalid conversation_id")
    base = os.path.realpath(DATA_DIR)
    path = os.path.realpath(os.path.join(base, f"{conversation_id}.json"))
    if not path.startswith(base + os.sep):
        raise ValueError("Invalid conversation_id")
    return open(path, mode)


def create_conversation(conversation_id: str) -> Dict[str, Any]:
    """
    Create a new conversation.

    Args:
        conversation_id: Unique identifier for the conversation

    Returns:
        New conversation dict
    """
    ensure_data_dir()

    conversation = {
        "id": conversation_id,
        "created_at": datetime.utcnow().isoformat(),
        "title": "New Conversation",
        "messages": []
    }

    # Save to file (validated open; the path guard is co-located with open()).
    with _open_conversation_file(conversation_id, 'w') as f:
        json.dump(conversation, f, indent=2)

    return conversation


def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a conversation from storage.

    Args:
        conversation_id: Unique identifier for the conversation

    Returns:
        Conversation dict or None if not found
    """
    try:
        with _open_conversation_file(conversation_id, 'r') as f:
            return json.load(f)
    except ValueError:
        # Malformed / malicious id (e.g. path traversal) — treat as "not found".
        return None
    except FileNotFoundError:
        # Valid id, but no such conversation yet.
        return None


def save_conversation(conversation: Dict[str, Any]):
    """
    Save a conversation to storage.

    Args:
        conversation: Conversation dict to save
    """
    ensure_data_dir()

    with _open_conversation_file(conversation['id'], 'w') as f:
        json.dump(conversation, f, indent=2)


def list_conversations() -> List[Dict[str, Any]]:
    """
    List all conversations (metadata only).

    Returns:
        List of conversation metadata dicts
    """
    ensure_data_dir()

    conversations = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json'):
            path = os.path.join(DATA_DIR, filename)
            with open(path, 'r') as f:
                data = json.load(f)
                # Return metadata only
                conversations.append({
                    "id": data["id"],
                    "created_at": data["created_at"],
                    "title": data.get("title", "New Conversation"),
                    "message_count": len(data["messages"])
                })

    # Sort by creation time, newest first
    conversations.sort(key=lambda x: x["created_at"], reverse=True)

    return conversations


def add_user_message(conversation_id: str, content: str):
    """
    Add a user message to a conversation.

    Args:
        conversation_id: Conversation identifier
        content: User message content
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    conversation["messages"].append({
        "role": "user",
        "content": content
    })

    save_conversation(conversation)


def add_assistant_message(
    conversation_id: str,
    stage1: List[Dict[str, Any]],
    stage2: List[Dict[str, Any]],
    stage3: Dict[str, Any]
):
    """
    Add an assistant message with all 3 stages to a conversation.

    Args:
        conversation_id: Conversation identifier
        stage1: List of individual model responses
        stage2: List of model rankings
        stage3: Final synthesized response
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    conversation["messages"].append({
        "role": "assistant",
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3
    })

    save_conversation(conversation)


def update_conversation_title(conversation_id: str, title: str):
    """
    Update the title of a conversation.

    Args:
        conversation_id: Conversation identifier
        title: New title for the conversation
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    conversation["title"] = title
    save_conversation(conversation)
