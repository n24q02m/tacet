1. **Fix Missing Exception Logging in Global Exception Handler**
   - In `src/tacet/serve/server.py`, the `@app.exception_handler(Exception)` global error handler explicitly catches exceptions to add security headers but fails to log the exception context (`_exc`). This overrides the default Starlette error handler which would have logged the traceback. As a result, critical unhandled internal server errors are silently swallowed and not logged server-side.
   - I will use `replace_with_git_merge_diff` to add `logging.error("Unhandled exception", exc_info=_exc)` in `unhandled_exception_handler` in `src/tacet/serve/server.py`.

2. **Update Test to Ensure Exception Logging is Maintained**
   - In `tests/test_mvp.py`, I will update `test_unhandled_route_exception_returns_hardened_generic_500` to also assert that the exception was logged with `self.assertLogs(level="ERROR")`.
   - This ensures the contract is maintained and prevents regressions.
   - I will use `replace_with_git_merge_diff` for `tests/test_mvp.py`.

3. **Verify the Fix**
   - Run `uv run --all-extras pytest tests/test_mvp.py` to ensure the updated test passes.
   - Read the modified files to ensure correct application.
   - Run `uv run ruff check .` and `uv run ruff format .` to verify formatting and linting.

4. **Append Learning to Journal**
   - I will append the learning to `.jules/sentinel.md` using `cat << 'EOF' >> .jules/sentinel.md`. The format will be:
     `## YYYY-MM-DD - [Title]`
     `**Vulnerability:** ...`
     `**Learning:** ...`
     `**Prevention:** ...`
   - Use `read_file` to confirm the file was written properly.

5. **Final Testing**
   - Run `uv run --all-extras pytest` to verify the full suite passes.

6. **Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.**

7. **Submit the PR**
   - Title: "🛡️ Sentinel: [MEDIUM] Fix swallowed exceptions in global handler"
   - Description containing:
     * 🚨 Severity: MEDIUM
     * 💡 Vulnerability: Unhandled exceptions caught by the global error handler were not logged server-side, silently swallowing potentially critical backend failures and stack traces, complicating debugging and incident response.
     * 🎯 Impact: Operations teams would have no visibility into root causes of 500 Internal Server Errors, masking backend vulnerabilities or infrastructure failures.
     * 🔧 Fix: Explicitly log the exception `logging.error(..., exc_info=_exc)` in the custom global exception handler. Also added a test assertion to prevent regressions.
     * ✅ Verification: Verified with updated test `test_unhandled_route_exception_returns_hardened_generic_500` asserting `self.assertLogs(level="ERROR")`. Full pytest suite passes.
