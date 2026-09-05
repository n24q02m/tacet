<!-- Sentinel (security) review memory for tacet. The Sentinel bot appends dated "Vulnerability / Learning / Prevention" entries below after each task; this file seeds the convention used across the n24q02m repos. Dates are the date the change landed on main; keep them accurate, because an entry the bot cannot date is an entry it re-proposes. -->

## 2026-06-30 - Prevented Stack Trace Leakage in FastAPI Endpoints
**Vulnerability:** The `/ask` endpoint in `src/tacet/serve/server.py` passed `str(e)` directly into the `detail` parameter of a 500 HTTP Exception response on generic catch-all errors.
**Learning:** Returning `str(e)` directly exposes raw error messages, which could leak internal application state, database schemas, or API errors directly to the client. This violates the principle of "fail securely".
**Prevention:** Changed the endpoint to log the error context server-side securely via `logging.error("Operation failed", exc_info=True)` and return a safe, generic message (`"Internal server error"`) to the client.

## 2026-07-01 - Added Security Headers Middleware to FastAPI App
**Vulnerability:** The FastAPI application was missing standard security headers (X-Content-Type-Options, X-Frame-Options, Strict-Transport-Security), leaving it potentially vulnerable to MIME-type sniffing, clickjacking, and allowing clients to access it over unencrypted HTTP.
**Learning:** Implementing defense-in-depth is essential; even internal or proxy-protected APIs should implement their own fundamental security headers directly via middleware.
**Prevention:** An HTTP middleware layer was added in `src/tacet/serve/server.py` to globally inject standard security headers to all responses.

## 2026-07-10 - Prevent DoS via Unbounded Payloads
**Vulnerability:** FastAPI endpoints using default Pydantic models for nested and flat string fields were not enforcing maximum lengths, creating a Denial of Service (DoS) vector from unbounded inputs.
**Learning:** Pydantic's default `str` fields and `list` collections allow unbounded sizes, meaning maliciously large nested JSON requests could exhaust memory.
**Prevention:** Always enforce max sizes in request models. Use `Field(..., max_length=X)` for strings or collection lengths. For strings nested in lists, use `typing.Annotated`, like `list[Annotated[str, Field(max_length=X)]]`.

## 2026-07-24 - Content-Security-Policy added; the legacy XSS auditor disabled
**Vulnerability:** The header middleware sent no `Content-Security-Policy`, so a response reflected into a browser context had no policy of its own to fall back on.
**Learning:** This service returns JSON only, so the correct policy is `default-src 'none'; frame-ancestors 'none'` -- `default-src 'self'` still permits same-origin script and frame loading and is the wrong shape here. The documentation paths (`/docs`, `/redoc`, `/openapi.json`) must be exempt or Swagger UI stops rendering.
**Prevention:** The middleware in `src/tacet/serve/server.py` now sets that CSP on every non-documentation response, and sets `X-XSS-Protection: 0`.

## 2026-07-27 - The `/ask` error path is a tested contract, not an accident
**Vulnerability:** none -- this entry exists because two bot PRs proposed opposite changes to the same handler from the same wrong premise, and the path had no test to arbitrate.
**Learning:** `/ask` answers a service failure with a JSON 500 whose body is exactly `{"detail": "Internal server error"}`, logged server-side with the traceback. Deleting the `try/except` does NOT preserve that: the exception reaches `ServerErrorMiddleware`, which answers `text/plain` `Internal Server Error`, so a client reading `detail` breaks. Measured on both shapes, not inferred.
**Prevention:** `tests/test_mvp.py::TestServerEndpoints::test_ask_surfaces_a_service_failure_as_a_json_500_and_logs_it` pins status, content-type, body, the absence of the cause from the response, and its presence in the log. Verified to FAIL on the delete-the-wrapper shape. The `# pragma: no cover` that used to sit on the `except` is gone -- it was the marker that this path was unowned.

## Rejected hardening (do not re-propose without the stated evidence)
- **`X-XSS-Protection: 1; mode=block`.** The auditor has been removed from every current browser, and its blocking mode has itself been an XS-leak vector, so the recommended value is `0`. That is what the middleware sets; raising it to `1` is a regression, not a hardening.
- **A generic `try/except Exception` around `/distill`, `/consolidate` or `/ingest` to prevent "stack trace leakage".** Starlette does not put a traceback in the response body for an unhandled exception: `ServerErrorMiddleware` returns a plain 500 and logs the trace server-side. Those endpoints leak nothing today, so the wrapper -- and the `except HTTPException: raise` guard that has to come with it -- is error handling for a path that cannot currently be reached, since the service layer never raises `HTTPException`.
- **`except HTTPException: raise` inside `/ask`** (PR #132, closed). Same evidence, now checked route by route: **401** comes from `_require_api_key`, declared in `dependencies=[Depends(...)]`, so it is raised before the handler body and outside the `try`; **422** is FastAPI request validation, likewise before the body; and `grep -rn "raise HTTPException" src/` returns two hits, both in `server.py` itself. The guard defends a path with no way in.
- **Removing the `try/except` from `/ask`** (PR #135, closed). The opposite change, from the same premise, and it silently rewrites the error contract from a JSON body with `detail` to Starlette's plain-text 500. If this looks attractive again, run the test named in the 2026-07-27 entry first -- it exists precisely to answer this.

## 2025-02-21 - [Missing Referrer-Policy Security Header]
**Vulnerability:** The FastAPI application was missing the `Referrer-Policy` security header, which could leak sensitive path information or query parameters via the `Referer` header to external sites when navigating away from the application.
**Learning:** Even when setting strict `Content-Security-Policy` and `X-Frame-Options`, `Referrer-Policy` is needed to prevent cross-origin information leakage on outbound requests.
**Prevention:** Always include `Referrer-Policy: no-referrer` in the global security headers middleware to strictly drop referrer information on all outbound requests.

## 2025-02-21 - [Prevent swallowed exception tracebacks in Exception Handler]
**Vulnerability:** The `unhandled_exception_handler` middleware caught all internal exceptions returning generic 'Internal Server Error' 500 response while ignoring explicitly logging the exceptions `log.error`, resulting in silently swallowed tracebacks server side.
**Learning:** Returning a safe generic error response prevents internal application stack trace exposure, but it's important to make sure exceptions tracebacks are logged properly on the server side to detect attacks/bugs in the internal app code stack.
**Prevention:** Ensured the custom unhandled exception handler properly logs the exception using `log.error("Unhandled exception", exc_info=_exc)` instead of ignoring it silently.
