"""
Local code execution using nsjail (https://github.com/google/nsjail) as the sandbox.

Requirements:
  - nsjail binary installed and on PATH, or set the NSJAIL_BIN environment variable.
  - For C/C++: g++/gcc compiler available.
  - Linux kernel with user-namespace support (CONFIG_USER_NS=y).

Security notes:
  By default this module uses --disable_clone_newns so that nsjail can access
  the host filesystem without needing chroot/bindmount configuration.  The
  process is still isolated in a new network namespace (no outbound network)
  and resource-limited (time, address-space, file-size).  For stricter
  filesystem isolation, pass extra_nsjail_args such as:
      ["--chroot", "/", "--bindmount", tmpdir]
  and ensure nsjail is run with sufficient capabilities.

Usage:
    from verl.utils.reward_score.cot_monitor.nsjail.utils import check_correctness

    results, metadata = check_correctness(
        in_outs={"inputs": ["3\\n"], "outputs": ["6\\n"]},
        generation="n = int(input()); print(n * 2)",
        timeout=10,
        memory_limit_mb=512,
        language="python",
    )
"""

import concurrent.futures
import functools
import logging
import os
import subprocess
import tempfile
import threading
import traceback
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10

# Path to the nsjail binary; override via environment variable.
NSJAIL_BIN = os.environ.get("NSJAIL_BIN", "nsjail")

# ---------------------------------------------------------------------------
# Language configurations
# ---------------------------------------------------------------------------

# Each entry: extension, optional compile step, run-command factory.
#   compile(src_path, bin_path) -> [cmd, ...]  or  None for interpreted langs
#   run(src_path, bin_path)     -> [cmd, ...]
_LANGUAGE_CONFIGS: dict[str, dict] = {
    "python": {
        "extension": ".py",
        "compile": None,
        "run": lambda src, _bin: ["python3", src],
    },
    "python3": {
        "extension": ".py",
        "compile": None,
        "run": lambda src, _bin: ["python3", src],
    },
    "cpp": {
        "extension": ".cpp",
        "compile": lambda src, out: ["g++", "-O2", "-std=c++17", "-o", out, src],
        "run": lambda _src, binary: [binary],
    },
    "c": {
        "extension": ".c",
        "compile": lambda src, out: ["gcc", "-O2", "-o", out, src],
        "run": lambda _src, binary: [binary],
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_proc(cmd: list[str], stdin_data: Optional[str], timeout: int) -> dict:
    """Run *cmd* as a subprocess and return a result dict.

    Returns:
        {stdout, stderr, return_code, timed_out}
    Raises:
        RuntimeError: if the binary is not found.
    """
    stdin_bytes = stdin_data.encode() if stdin_data else None
    try:
        proc = subprocess.run(
            cmd,
            input=stdin_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout + 5,  # extra buffer beyond nsjail's own time_limit
        )
        return {
            "stdout": proc.stdout.decode(errors="replace"),
            "stderr": proc.stderr.decode(errors="replace"),
            "return_code": proc.returncode,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "Process exceeded outer timeout",
            "return_code": -1,
            "timed_out": True,
        }
    except FileNotFoundError as exc:
        raise RuntimeError(f"Command not found: {cmd[0]!r}. Is it installed? ({exc})")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def call_nsjail_api(
    code: str,
    stdin: Optional[str],
    compile_timeout: int,
    run_timeout: int,
    memory_limit_mb: int,
    language: str = "python",
    nsjail_bin: str = NSJAIL_BIN,
    extra_nsjail_args: Optional[list[str]] = None,
) -> tuple[Optional[dict], Optional[str]]:
    """Execute *code* in an nsjail sandbox and return a response dict that
    mirrors the sandbox_fusion REST API format, so that the existing
    ``_process_single_case`` result-parsing logic works unchanged.

    Args:
        code: Source code string to execute.
        stdin: Standard input for the program (or None).
        compile_timeout: Seconds allowed for the compilation step (compiled langs).
        run_timeout: Seconds allowed for program execution.
        memory_limit_mb: Address-space limit passed to nsjail's ``--rlimit_as``.
        language: One of ``python``, ``python3``, ``cpp``, ``c``.
        nsjail_bin: Path to the nsjail binary.
        extra_nsjail_args: Additional nsjail flags inserted before ``--``.

    Returns:
        ``(response_dict, error_msg)``

        *response_dict* structure (mirrors sandbox_fusion JSON)::

            {
              "status": "Success" | "Failed" | "SandboxError",
              "compile_result": {         # None for interpreted languages
                  "status": "Finished" | "Error" | "TimeLimitExceeded",
                  "stderr": "...",
                  "execution_time": None,
                  "return_code": 0,
              },
              "run_result": {
                  "status": "Finished" | "Error" | "TimeLimitExceeded",
                  "stdout": "...",
                  "stderr": "...",
                  "return_code": 0,
                  "execution_time": None,
              },
            }

        On hard errors (missing binary, unsupported language) ``response_dict``
        is ``None`` and ``error_msg`` contains the description.
    """
    lang_cfg = _LANGUAGE_CONFIGS.get(language)
    if lang_cfg is None:
        return None, (
            f"Unsupported language {language!r} for nsjail backend. "
            f"Supported: {list(_LANGUAGE_CONFIGS)}"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, f"solution{lang_cfg['extension']}")
        with open(src_path, "w") as fh:
            fh.write(code)
        os.chmod(src_path, 0o644)

        response: dict = {"status": None, "compile_result": None, "run_result": None}

        # ----------------------------------------------------------------
        # Compile step (compiled languages only)
        # ----------------------------------------------------------------
        if lang_cfg["compile"] is not None:
            bin_path = os.path.join(tmpdir, "solution")
            compile_cmd = lang_cfg["compile"](src_path, bin_path)
            try:
                compile_res = _run_proc(compile_cmd, None, compile_timeout)
            except RuntimeError as exc:
                return None, str(exc)

            if compile_res["timed_out"]:
                response["compile_result"] = {
                    "status": "TimeLimitExceeded",
                    "stderr": compile_res["stderr"],
                    "execution_time": compile_timeout,
                    "return_code": -1,
                }
                response["status"] = "Failed"
                return response, None

            if compile_res["return_code"] != 0:
                response["compile_result"] = {
                    "status": "Error",
                    "stderr": compile_res["stderr"],
                    "execution_time": None,
                    "return_code": compile_res["return_code"],
                }
                response["status"] = "Failed"
                return response, None

            response["compile_result"] = {
                "status": "Finished",
                "stderr": compile_res["stderr"],
                "execution_time": None,
                "return_code": 0,
            }
            os.chmod(bin_path, 0o755)
            run_target_src = src_path
            run_target_bin = bin_path
        else:
            run_target_src = src_path
            run_target_bin = None

        # ----------------------------------------------------------------
        # Run step (wrapped in nsjail)
        # ----------------------------------------------------------------
        run_cmd = lang_cfg["run"](run_target_src, run_target_bin)

        nsjail_cmd = [
            nsjail_bin,
            "--mode", "o",
            # Use host filesystem so the code file and interpreter are accessible
            # without requiring chroot / bindmount configuration.
            "--disable_clone_newns",
            "--time_limit", str(run_timeout),
            "--rlimit_as", str(memory_limit_mb),   # address-space limit in MB
            "--rlimit_fsize", "16",                # max output file: 16 MB
            "--log", "/dev/null",                  # suppress nsjail internal logs
        ]
        if extra_nsjail_args:
            nsjail_cmd.extend(extra_nsjail_args)
        nsjail_cmd += ["--"] + run_cmd

        try:
            run_res = _run_proc(nsjail_cmd, stdin, run_timeout)
        except RuntimeError as exc:
            return None, str(exc)

        rc = run_res["return_code"]
        # nsjail sends SIGKILL on TLE → child exit code 137 (128+9)
        timed_out = run_res["timed_out"] or rc == 137

        if timed_out:
            run_status = "TimeLimitExceeded"
        elif rc == 0:
            run_status = "Finished"
        else:
            run_status = "Error"

        response["run_result"] = {
            "status": run_status,
            "stdout": run_res["stdout"],
            "stderr": run_res["stderr"],
            "return_code": rc,
            "execution_time": None,
        }
        response["status"] = "Success" if run_status == "Finished" else "Failed"
        return response, None


def check_correctness(
    in_outs: Optional[dict],
    generation: str,
    timeout: int = DEFAULT_TIMEOUT,
    memory_limit_mb: int = 1024,
    language: str = "python",
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    nsjail_bin: str = NSJAIL_BIN,
    extra_nsjail_args: Optional[list[str]] = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Check correctness of *generation* against test cases using nsjail.

    Mirrors the interface of ``sandbox_fusion.utils.check_correctness`` so
    callers can swap backends transparently.

    Args:
        in_outs: ``{"inputs": [...], "outputs": [...]}`` test-case dict.
        generation: Source code to evaluate.
        timeout: Per-test-case time limit in seconds.
        memory_limit_mb: Per-test-case address-space limit in MB.
        language: Programming language of the code.
        concurrent_semaphore: Optional semaphore to cap parallel executions.
        nsjail_bin: Path to the nsjail binary (or ``NSJAIL_BIN`` env var).
        extra_nsjail_args: Extra flags passed verbatim to nsjail before ``--``.

    Returns:
        ``(results, metadata_list)`` — same format as
        ``sandbox_fusion.utils.check_correctness``.
    """
    # Import here to avoid circular imports; _process_single_case is the
    # shared result-parsing layer that works with any call_fn.
    from verl.utils.reward_score.cot_monitor.sandbox_fusion.utils import _process_single_case

    if not in_outs or "inputs" not in in_outs or "outputs" not in in_outs:
        logger.warning("Invalid in_outs format provided.")
        return [-1], [{"error": "Invalid input/output data"}]

    inputs = in_outs["inputs"]
    expected_outputs = in_outs["outputs"]
    fn_name = in_outs.get("fn_name")
    num_cases = len(inputs)
    assert_cases = in_outs.get("assert_case", [""] * num_cases)

    if num_cases == 0:
        return [], []

    if len(inputs) != len(expected_outputs):
        logger.warning("Mismatch between number of inputs and outputs.")
        return [-1] * num_cases, [
            {"error": "Input/output count mismatch", "case_index": i} for i in range(num_cases)
        ]

    if len(assert_cases) != num_cases:
        logger.warning("Mismatch between number of assert cases and inputs/outputs.")
        return [-1] * num_cases, [
            {"error": "Assert case count mismatch", "case_index": i} for i in range(num_cases)
        ]

    call_fn = functools.partial(
        call_nsjail_api,
        nsjail_bin=nsjail_bin,
        extra_nsjail_args=extra_nsjail_args,
    )

    results: list = [None] * num_cases
    metadata_list: list = [None] * num_cases
    first_compile_error_index = -1

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_index = {
            executor.submit(
                _process_single_case,
                i,
                stdin_data,
                expected_outputs[i],
                call_fn,
                generation + "\n\n" + assert_cases[i],
                timeout,
                memory_limit_mb,
                language,
                concurrent_semaphore,
                fn_name,
            ): i
            for i, stdin_data in enumerate(inputs)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result_status, metadata = future.result()
                results[index] = result_status
                metadata_list[index] = metadata
                if result_status == -4 and (
                    first_compile_error_index == -1 or index < first_compile_error_index
                ):
                    first_compile_error_index = index
            except Exception as exc:
                logger.error(f"Test case {index} raised: {exc}")
                traceback.print_exc()
                results[index] = -1
                metadata_list[index] = {
                    "case_index": index,
                    "input": str(inputs[index]),
                    "expected_output": str(expected_outputs[index]) if expected_outputs[index] else None,
                    "status": "internal_error",
                    "api_request_error": str(exc),
                }

    # Propagate compile-error status to subsequent cases (same logic as sandbox_fusion)
    if first_compile_error_index != -1:
        logger.warning(
            f"Compile error in case {first_compile_error_index}; "
            "marking subsequent cases as compile_error_skipped."
        )
        for i in range(first_compile_error_index + 1, num_cases):
            if results[i] != -4:
                results[i] = -4
                if metadata_list[i] is None:
                    metadata_list[i] = {
                        "case_index": i,
                        "input": str(inputs[i]),
                        "expected_output": str(expected_outputs[i]) if expected_outputs[i] else None,
                        "status": "compile_error_skipped",
                    }
                else:
                    metadata_list[i]["status"] = "compile_error_skipped"

    logger.info(f"nsjail correctness check finished. Results: {results}")
    return results, metadata_list