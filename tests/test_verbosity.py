from __future__ import annotations

import io
import logging
import unittest
from contextlib import redirect_stdout

from gleamlet.utils.verbosity import control_class_verbosity, control_user_output


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
