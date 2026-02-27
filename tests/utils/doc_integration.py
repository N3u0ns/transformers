import re
import unittest
from collections.abc import Iterable
from pathlib import Path


class DocIntegrationTest(unittest.TestCase):
    """Base class to execute runnable code blocks embedded in markdown docs."""

    # Path to the markdown file; subclasses should override.
    doc_path: Path | None = None
    # Optional cleanup hook called after each block (e.g., freeing GPU memory).
    cleanup_func = None
    # Flag that marks which fenced code blocks should be executed.
    runnable_flag = "runnable"

    def _iter_runnable_code_blocks(self, text: str) -> Iterable[str]:
        """
        Yield code blocks marked with the runnable flag from the given markdown text.
        Recognizes ```py runnable ...``` and ```python runnable ...``` fences.
        """
        pattern = re.compile(r"```(?P<lang>python|py)(?P<flags>[^\n]*)\n(?P<code>.+?)\n```", re.DOTALL)
        for match in pattern.finditer(text):
            flags = {flag.strip() for flag in match.group("flags").split() if flag.strip()}
            if self.runnable_flag not in flags:
                continue
            yield match.group("code")

    def _run_cleanup(self):
        if callable(self.cleanup_func):
            self.cleanup_func()

    def test_markdown_runnable_blocks(self):
        if self.doc_path is None:
            self.skipTest("doc_path not set on DocIntegrationTest subclass")

        resolved_path = Path(self.doc_path)
        doc_text = resolved_path.read_text(encoding="utf-8")
        runnable_blocks = list(self._iter_runnable_code_blocks(doc_text))

        self.assertTrue(runnable_blocks, f"No runnable code blocks were found in {resolved_path}")

        for idx, code in enumerate(runnable_blocks):
            with self.subTest(block=idx):
                namespace = {"__name__": f"{resolved_path.stem}_doc_example_{idx}"}
                try:
                    exec(compile(code, str(resolved_path), "exec"), namespace)
                except Exception as err:
                    raise AssertionError(
                        f"Doc block {idx} in {resolved_path} failed to execute:\n{code}"
                    ) from err
                finally:
                    self._run_cleanup()
