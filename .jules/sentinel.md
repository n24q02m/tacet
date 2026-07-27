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
## 2024-07-27 - Generic Try/Except Masking Framework Exceptions in FastAPI
**Vulnerability:** Adding generic `try/except Exception:` blocks inside FastAPI endpoints to catch unexpected errors and return a generic 500 response risks improperly catching and masking intentional framework-level HTTPExceptions (e.g. 401 Unauthorized, 422 Unprocessable Entity) if they are not explicitly caught and re-raised first.
**Learning:** In Starlette/FastAPI, unhandled exceptions are automatically and securely handled by `ServerErrorMiddleware`, which prevents stack trace leakage by returning a plain 500 response and logging the traceback server-side.
**Prevention:** Avoid adding generic `try/except Exception:` blocks to endpoints solely to prevent stack trace leaks. If generic exception handling is necessary, explicitly catch and re-raise framework exceptions (e.g., `except HTTPException: raise`) before the generic block.
## 2024-07-28 - Understanding API Contracts and Error Handling
**Vulnerability:** A previous finding suggested removing `try/except Exception:` blocks in FastAPI routes, assuming they could mask `HTTPExceptions`. However, `HTTPExceptions` for authentication (401) and validation (422) occur *before* the route handler body due to FastAPI's dependency injection and request validation mechanisms.
**Learning:** Modifying error handling in an API route can inadvertently change the API contract (e.g., from `{"detail":"Internal server error"}` to a plaintext `Internal Server Error`). A `try/except Exception:` block might be intentionally maintaining a specific JSON error format for client compatibility, even if it seems redundant.
**Prevention:** Before altering exception handling, thoroughly trace the execution flow (including dependencies and request validation) to understand if and where specific exceptions can actually be raised. Always verify that changes to error responses do not violate existing API contracts, especially regarding content types and response bodies. Ensure that error paths are covered by automated tests to catch unintended contract changes.
