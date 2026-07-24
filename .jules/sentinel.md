<!-- Sentinel (security) review memory for tacet. The Sentinel bot appends dated "Vulnerability / Learning / Prevention" entries below after each task; this file seeds the convention used across the n24q02m repos. -->

## 2024-05-25 - Prevented Stack Trace Leakage in FastAPI Endpoints
**Vulnerability:** The `/ask` endpoint in `src/tacet/serve/server.py` passed `str(e)` directly into the `detail` parameter of a 500 HTTP Exception response on generic catch-all errors.
**Learning:** Returning `str(e)` directly exposes raw error messages, which could leak internal application state, database schemas, or API errors directly to the client. This violates the principle of "fail securely".
**Prevention:** Changed the endpoint to log the error context server-side securely via `logging.error("Operation failed", exc_info=True)` and return a safe, generic message (`"Internal server error"`) to the client.

## 2026-07-01 - Added Security Headers Middleware to FastAPI App
**Vulnerability:** The FastAPI application was missing standard security headers (, , ), leaving it potentially vulnerable to MIME-type sniffing, clickjacking, and allowing clients to access it over unencrypted HTTP (HSTS).
**Learning:** Implementing defense-in-depth is essential; even internal or proxy-protected APIs should implement their own fundamental security headers directly via middleware in case proxy configuration fails.
**Prevention:** An  layer was added in  to globally inject standard security headers to all responses.


## 2024-05-25 - Added Security Headers Middleware to FastAPI App
**Vulnerability:** The FastAPI application was missing standard security headers (X-Content-Type-Options, X-Frame-Options, Strict-Transport-Security), leaving it potentially vulnerable to MIME-type sniffing, clickjacking, and allowing clients to access it over unencrypted HTTP.
**Learning:** Implementing defense-in-depth is essential; even internal or proxy-protected APIs should implement their own fundamental security headers directly via middleware.
**Prevention:** An HTTP middleware layer was added in src/tacet/serve/server.py to globally inject standard security headers to all responses.
## 2024-05-18 - Prevent DoS via Unbounded Payloads
**Vulnerability:** FastAPI endpoints using default Pydantic models for nested and flat string fields were not enforcing maximum lengths, creating a Denial of Service (DoS) vector from unbounded inputs.
**Learning:** Pydantic's default `str` fields and `list` collections allow unbounded sizes, meaning maliciously large nested JSON requests could exhaust memory.
**Prevention:** Always enforce max sizes in request models. Use `Field(..., max_length=X)` for strings or collection lengths. For strings nested in lists, use `typing.Annotated`, like `list[Annotated[str, Field(max_length=X)]]`.

## 2024-05-25 - Prevent Generic Catch-All from Obscuring HTTPExceptions
**Vulnerability:** A generic `except Exception:` block in the `/ask` endpoint was mistakenly catching intentionally raised `HTTPException`s, masking authentication/authorization errors as 500 errors.
**Learning:** When dealing with FastAPI (or any framework with dedicated exception types like HTTPException), a bare `except Exception:` handler will intercept framework-specific errors too, potentially leaking generic error states or bypassing explicit application logic for 40x codes.
**Prevention:** Always add an explicit `except HTTPException: raise` before the generic `except Exception:` catch-all, or ensure the generic catch block does not interfere with the framework's internal error types.

## 2024-05-25 - Security Headers - Deprecated and Tightened Policies
**Vulnerability:** A previous PR applied `X-XSS-Protection: 1; mode=block` and `Content-Security-Policy: default-src 'self'`.
**Learning:** `X-XSS-Protection` is deprecated in modern browsers and its blocking mode is a known XS-leak vector; the recommended secure value is now `0`. Additionally, for API endpoints returning only JSON, `default-src 'self'` is too permissive as it allows same-origin scripts/frames. The optimal policy for such APIs is `default-src 'none'; frame-ancestors 'none'`.
**Prevention:** When implementing security headers for JSON APIs, set `X-XSS-Protection: 0` and use the most restrictive CSP (`default-src 'none'; frame-ancestors 'none'`), omitting them only for documentation endpoints (like Swagger UI) that genuinely require browser execution capabilities.
