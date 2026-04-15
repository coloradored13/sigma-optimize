#!/usr/bin/env python3
"""Test budget pause/resume mechanism end-to-end (A3).

Validates that:
1. Workers pause when a budget error is detected
2. The RESUME file mechanism re-opens the gate
3. Workers continue after resume
4. Only the first worker to detect triggers the pause message

Run: python3 test_budget_pause.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# We need to test the actual mechanism from experiment_multi.py
sys.path.insert(0, str(Path(__file__).parent))

from experiment_multi import (
    _async_single_call,
    _budget_resume_watcher,
    _is_budget_error,
    RESUME_FILE,
    RESULTS_DIR,
    MAX_RETRIES,
)


class BudgetError(Exception):
    pass


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"  PASS: {name}")

    def fail(self, name, msg=""):
        self.failed += 1
        self.errors.append(f"{name}: {msg}")
        print(f"  FAIL: {name} — {msg}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"Results: {self.passed}/{total} passed")
        if self.errors:
            print("Failures:")
            for e in self.errors:
                print(f"  - {e}")
        return self.failed == 0


def test_is_budget_error():
    """Test budget error detection."""
    r = TestResults()

    if _is_budget_error(Exception("You have reached your specified usage limit")):
        r.ok("detects 'usage limit' error")
    else:
        r.fail("detects 'usage limit' error")

    if _is_budget_error(Exception("Budget exceeded for this billing period")):
        r.ok("detects 'budget' error")
    else:
        r.fail("detects 'budget' error")

    if not _is_budget_error(Exception("Rate limit exceeded")):
        r.ok("ignores non-budget errors")
    else:
        r.fail("ignores non-budget errors")

    if not _is_budget_error(Exception("Connection timeout")):
        r.ok("ignores connection errors")
    else:
        r.fail("ignores connection errors")

    return r


async def test_budget_gate_pauses_workers():
    """Test that workers actually pause when budget gate closes."""
    r = TestResults()
    gate = asyncio.Event()
    gate.set()

    # Track timing
    call_count = 0
    paused = False

    async def mock_create(**kwargs):
        nonlocal call_count, paused
        call_count += 1
        if call_count == 1:
            # First call triggers budget error
            raise BudgetError("You have reached your specified usage limit")
        # Subsequent calls succeed (after resume)
        paused = True
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="test response")]
        mock_resp.usage = MagicMock(output_tokens=50)
        return mock_resp

    mock_client = AsyncMock()
    mock_client.messages.create = mock_create

    semaphore = asyncio.Semaphore(10)

    # Start the call — it should pause on budget error
    call_task = asyncio.create_task(
        _async_single_call(mock_client, semaphore, gate, "test prompt", "simpsons_paradox")
    )

    # Give it time to hit the error and pause
    await asyncio.sleep(0.3)

    if not gate.is_set():
        r.ok("gate closed after budget error")
    else:
        r.fail("gate closed after budget error", "gate still open")

    # Now resume
    gate.set()
    try:
        result = await asyncio.wait_for(call_task, timeout=5.0)
        if result.task_id == "simpsons_paradox":
            r.ok("worker continued after resume")
        else:
            r.fail("worker continued after resume", f"wrong task_id: {result.task_id}")
    except asyncio.TimeoutError:
        r.fail("worker continued after resume", "timed out waiting for completion")

    return r


async def test_resume_file_watcher():
    """Test that the RESUME file mechanism works."""
    r = TestResults()

    gate = asyncio.Event()
    gate.clear()  # start paused

    # Start the watcher
    watcher_task = asyncio.create_task(_budget_resume_watcher(gate))

    # Create the resume file
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESUME_FILE.touch()

    # Wait for watcher to detect it
    try:
        await asyncio.wait_for(watcher_task, timeout=10.0)
        if gate.is_set():
            r.ok("RESUME file re-opened gate")
        else:
            r.fail("RESUME file re-opened gate", "gate still closed")
    except asyncio.TimeoutError:
        r.fail("RESUME file re-opened gate", "watcher didn't detect file in 10s")

    # Verify file was cleaned up
    if not RESUME_FILE.exists():
        r.ok("RESUME file removed after detection")
    else:
        r.fail("RESUME file removed after detection", "file still exists")
        RESUME_FILE.unlink(missing_ok=True)

    return r


async def test_first_worker_triggers_message():
    """Test that only the first worker to detect budget error prints the message.
    Note: in asyncio (single-threaded), only one coroutine runs at a time,
    so the gate.clear() is atomic. This test verifies that the is_set() guard
    prevents duplicate messages from concurrent workers on the SAME detection event.
    """
    r = TestResults()
    gate = asyncio.Event()
    gate.set()

    pause_messages = []
    original_print = print
    counting_enabled = True

    def counting_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        if "BUDGET EXHAUSTED" in msg and counting_enabled:
            pause_messages.append(msg)
        original_print(*args, **kwargs)

    attempt_count = 0

    async def mock_create(**kwargs):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count <= 3:
            # First 3 attempts (one per worker) trigger budget error
            raise BudgetError("You have reached your specified usage limit")
        # After resume, succeed
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="test response")]
        mock_resp.usage = MagicMock(output_tokens=50)
        return mock_resp

    mock_client = AsyncMock()
    mock_client.messages.create = mock_create

    semaphore = asyncio.Semaphore(10)

    with patch("builtins.print", counting_print):
        # Launch multiple workers that all hit budget error
        tasks = [
            asyncio.create_task(
                _async_single_call(mock_client, semaphore, gate, "test", "simpsons_paradox")
            )
            for _ in range(3)
        ]

        # Let them all hit the error
        await asyncio.sleep(0.5)

        # Stop counting before resume (post-resume retries are separate events)
        counting_enabled = False

        # Resume to let them finish
        gate.set()

        for t in tasks:
            try:
                await asyncio.wait_for(t, timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                t.cancel()

    # In asyncio, only 1 message per detection event since clear() is synchronous
    if len(pause_messages) == 1:
        r.ok("only first worker printed BUDGET EXHAUSTED on initial detection")
    else:
        r.fail("only first worker printed BUDGET EXHAUSTED on initial detection",
               f"got {len(pause_messages)} messages")

    return r


async def run_all_tests():
    all_results = []

    print("\n=== test_is_budget_error ===")
    all_results.append(test_is_budget_error())

    print("\n=== test_budget_gate_pauses_workers ===")
    all_results.append(await test_budget_gate_pauses_workers())

    print("\n=== test_resume_file_watcher ===")
    all_results.append(await test_resume_file_watcher())

    print("\n=== test_first_worker_triggers_message ===")
    all_results.append(await test_first_worker_triggers_message())

    print("\n" + "=" * 50)
    total_pass = sum(r.passed for r in all_results)
    total_fail = sum(r.failed for r in all_results)
    print(f"TOTAL: {total_pass}/{total_pass + total_fail} passed")

    if total_fail > 0:
        print("\nFailed tests:")
        for r in all_results:
            for e in r.errors:
                print(f"  - {e}")
        return False
    return True


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
