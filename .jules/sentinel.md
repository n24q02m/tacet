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
## 2024-07-20 - Add Missing Security Headers to FastAPI
**Vulnerability:** The FastAPI application was missing `X-XSS-Protection` and `Content-Security-Policy` headers, leaving the application more vulnerable to cross-site scripting (XSS) and other client-side injection attacks.
**Learning:** Security headers in FastAPI can be added centrally via middleware. Care must be taken not to apply strict CSP headers to auto-generated documentation endpoints (`/docs`, `/redoc`, `/openapi.json`), which rely on inline scripts/styles.
**Prevention:** Always verify that security middlewares apply appropriate headers and include exceptions for Swagger UI/Redoc when configuring strict Content-Security-Policies.
