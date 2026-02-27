import re
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class RunnableBlock:
    code: str
    name: str
    idx: int


class DocIntegrationTest(unittest.TestCase):
    """Base class to execute runnable code blocks embedded in markdown docs."""

    __test__ = False  # Prevent pytest from collecting the base class itself.

    # Path to the markdown file; subclasses should override.
    doc_path: Path | None = None
    # Optional cleanup hook called after each block (e.g., freeing GPU memory).
    cleanup_func = None
    # Flag that marks which fenced code blocks should be executed.
    runnable_flag = "runnable"
    # Internal switch: when True, per-block test methods are generated.
    _dynamic_tests_created = False

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        blocks = cls._collect_runnable_blocks()
        if not blocks:
            # Allow pytest to collect subclasses that provide their own doc_path without blocks.
            cls.__test__ = True
            return

        cls._dynamic_tests_created = True
        cls.__test__ = True  # Enable collection for subclasses with runnable blocks.

        for block in blocks:
            method_name = block.name if block.name.startswith("test") else f"test_{block.name}"
            if hasattr(cls, method_name):
                # Don't override existing explicit tests.
                continue

            def _make_test(code: str, name: str):
                def _test(self):
                    self._execute_block(code, name)

                _test.__name__ = method_name
                return _test

            setattr(cls, method_name, _make_test(block.code, block.name))

        # Avoid double-reporting: neutralize the catch-all method when per-block tests exist.
        if hasattr(cls, "test_markdown_runnable_blocks"):
            setattr(cls, "test_markdown_runnable_blocks", None)

    @classmethod
    def _collect_runnable_blocks(cls) -> list[RunnableBlock]:
        if cls.doc_path is None:
            return []

        resolved_path = Path(cls.doc_path)
        if not resolved_path.exists():
            return []

        doc_text = resolved_path.read_text(encoding="utf-8")
        blocks: list[RunnableBlock] = []
        for idx, (code, name) in enumerate(cls._iter_runnable_code_blocks(doc_text)):
            safe_name = name or f"doc_block_{idx}"
            blocks.append(RunnableBlock(code=code, name=safe_name, idx=idx))
        return blocks

    @classmethod
    def _iter_runnable_code_blocks(cls, text: str) -> Iterable[tuple[str, Optional[str]]]:
        """
        Yield (code, name) pairs for blocks marked with the runnable flag from the given markdown text.
        Supports optional naming via a suffix: ```py runnable:test_name
        """
        pattern = re.compile(r"```(?P<lang>python|py)(?P<flags>[^\n]*)\n(?P<code>.+?)\n```", re.DOTALL)
        for match in pattern.finditer(text):
            flags = [flag.strip() for flag in match.group("flags").split() if flag.strip()]
            runnable_flags = [
                flag for flag in flags if flag == cls.runnable_flag or flag.startswith(f"{cls.runnable_flag}:")
            ]
            if not runnable_flags:
                continue
            name = None
            for flag in runnable_flags:
                if ":" in flag:
                    _, name = flag.split(":", 1)
            yield match.group("code"), name

    def _run_cleanup(self):
        if callable(self.cleanup_func):
            self.cleanup_func()

    def _execute_block(self, code: str, name: str):
        resolved_path = Path(self.doc_path) if self.doc_path is not None else Path("unknown_doc.md")
        namespace = {"__name__": f"{resolved_path.stem}_{name}"}
        try:
            exec(compile(code, str(resolved_path), "exec"), namespace)
        except Exception as err:
            raise AssertionError(f"Doc block '{name}' in {resolved_path} failed to execute:\n{code}") from err
        finally:
            self._run_cleanup()

    def test_markdown_runnable_blocks(self):
        # If dynamic per-block tests are generated, avoid double-running.
        if self._dynamic_tests_created:
            self.skipTest("Per-block doc tests generated dynamically; run those instead.")

        if self.doc_path is None:
            self.skipTest("doc_path not set on DocIntegrationTest subclass")

        resolved_path = Path(self.doc_path)
        doc_text = resolved_path.read_text(encoding="utf-8")
        runnable_blocks = list(self._iter_runnable_code_blocks(doc_text))

        self.assertTrue(runnable_blocks, f"No runnable code blocks were found in {resolved_path}")

        for idx, (code, name) in enumerate(runnable_blocks):
            name = name or f"doc_block_{idx}"
            with self.subTest(block=name):
                self._execute_block(code, name)
