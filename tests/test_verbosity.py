from __future__ import annotations

import io
import logging
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import pandas as pd
from rich.console import Console
from rich.progress import Progress

from gleamlet.preprocessing.modules import cleaning
from gleamlet.utils.verbosity import (
    control_class_verbosity,
    control_user_output,
    routine_output_enabled,
)


class VerbosityControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.log_output = io.StringIO()
        self.logger = logging.getLogger(f"verbosity-test-{id(self)}")
        self.logger.handlers.clear()
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        self.handler = logging.StreamHandler(self.log_output)
        self.logger.addHandler(self.handler)

    def tearDown(self) -> None:
        self.logger.handlers.clear()

    def test_verbose_false_hides_routine_output_but_keeps_warnings(self) -> None:
        logger = self.logger

        @control_user_output(logger)
        def operation(*, verbose=True):
            print("progress")
            logger.info("routine")
            logger.warning("warning")

        standard_output = io.StringIO()
        with redirect_stdout(standard_output):
            operation(verbose=False)

        self.assertEqual(standard_output.getvalue(), "")
        self.assertEqual(self.log_output.getvalue().strip(), "warning")

    def test_verbose_false_keeps_progress_written_to_stderr(self) -> None:
        @control_user_output()
        def operation(*, verbose=True):
            print("routine")
            with Progress(
                console=Console(stderr=True, force_terminal=True),
            ) as progress:
                progress.add_task("Cleaning files...", total=1, completed=1)

        standard_output = io.StringIO()
        progress_output = io.StringIO()
        with redirect_stdout(standard_output), redirect_stderr(progress_output):
            operation(verbose=False)

        self.assertEqual(standard_output.getvalue(), "")
        self.assertIn("Cleaning files...", progress_output.getvalue())

    def test_verbose_false_is_visible_to_nested_rich_output(self) -> None:
        output_states = []

        @control_user_output()
        def operation(*, verbose=True):
            output_states.append(routine_output_enabled())

        operation(verbose=False)

        self.assertEqual(output_states, [False])

    def test_verbose_false_hides_cleaning_rich_details(self) -> None:
        leaked_output = io.StringIO()

        class NotebookConsole:
            def __init__(self, *, file=None, **kwargs):
                self.file = file or leaked_output

            def print(self, value):
                self.file.write(str(value))

        @control_user_output(cleaning.logger)
        def operation(*, verbose=True):
            cleaning.remove_data(
                pd.DataFrame({"stud_id": [1, 1], "value": [2, 2]}),
                rm_nan_stud_id=False,
                rm_nan_cols_threshold=False,
                rm_dups_threshold="first",
                rm_empty_cols=False,
                rm_constant_cols=False,
            )

        with (
            patch.object(cleaning, "Console", NotebookConsole),
            patch.object(cleaning.cleaning_detail_logger, "info") as detail_log,
        ):
            operation(verbose=False)

        self.assertEqual(leaked_output.getvalue(), "")
        detail_log.assert_any_call("Duplicate rows detected")

    def test_method_override_is_temporary_and_applies_to_nested_calls(self) -> None:
        logger = self.logger

        @control_user_output(logger)
        def nested(*, verbose=None):
            logger.info("nested")

        @control_class_verbosity(logger)
        class Workflow:
            def __init__(self, verbose=True):
                self.verbose = verbose

            def run(self, *, verbose=None):
                logger.info("outer")
                nested()

        workflow = Workflow(verbose=True)
        workflow.run(verbose=False)
        self.assertEqual(self.log_output.getvalue(), "")
        self.assertTrue(workflow.verbose)


if __name__ == "__main__":
    unittest.main()
