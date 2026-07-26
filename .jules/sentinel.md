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

## Rejected hardening (do not re-propose without the stated evidence)
- **`X-XSS-Protection: 1; mode=block`.** The auditor has been removed from every current browser, and its blocking mode has itself been an XS-leak vector, so the recommended value is `0`. That is what the middleware sets; raising it to `1` is a regression, not a hardening.
- **A generic `try/except Exception` around `/distill`, `/consolidate` or `/ingest` to prevent "stack trace leakage".** Starlette does not put a traceback in the response body for an unhandled exception: `ServerErrorMiddleware` returns a plain 500 and logs the trace server-side. Those endpoints leak nothing today, so the wrapper -- and the `except HTTPException: raise` guard that has to come with it -- is error handling for a path that cannot currently be reached, since the service layer never raises `HTTPException`.

## 2026-07-26 - Prevented Framework Exception Obscuration in Endpoints
**Vulnerability:** The `/ask` endpoint caught all exceptions universally (`except Exception:`) without re-raising framework exceptions like `HTTPException`.
**Learning:** Broad exception handling used to prevent data leakage (like hiding stack traces in 500 responses) can accidentally catch intentional framework exceptions (e.g., FastAPI's `HTTPException` for 400 Bad Request or 401 Unauthorized), obscuring them as 500 errors and breaking expected API behavior and security checks.
**Prevention:** When adding generic exception handling to prevent stack trace leaks, always explicitly catch and re-raise framework-specific exceptions (`except HTTPException: raise`) before the generic `except Exception:` block.
