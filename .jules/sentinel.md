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
## 2024-05-24 - [Add missing HTTP security headers]
**Vulnerability:** The FastAPI API lacked the `X-XSS-Protection` header and didn't use `Content-Security-Policy` globally, which violates defense-in-depth best practices against Cross-Site Scripting (XSS).
**Learning:** Automatically generated API documentation (like `/docs` and `/redoc` in FastAPI) requires inline scripts and external assets to render correctly. Applying a strict CSP globally (`default-src 'self'`) breaks the Swagger UI, causing a functional regression.
**Prevention:** When enforcing strict CSP rules globally via middleware, selectively exclude auto-generated documentation endpoints (`/docs`, `/redoc`, `/openapi.json`) to preserve functionality while securing standard API routes.
## 2024-05-24 - [Correction: Security headers X-XSS-Protection and CSP]
**Vulnerability:** Adding security headers with outdated or overly permissive values (e.g., `X-XSS-Protection: 1; mode=block` and `Content-Security-Policy: default-src 'self'`).
**Learning:** `X-XSS-Protection: 1; mode=block` is actually an XS-leak vector and the auditor has been removed from modern browsers; the recommended value is `0` to explicitly disable it. Additionally, for a service that returns *only JSON*, `default-src 'self'` is too permissive (it allows same-origin scripts). The most secure policy for a JSON API is `default-src 'none'; frame-ancestors 'none'`.
**Prevention:** Stay up to date with modern header recommendations (e.g., OWASP). When an API only returns JSON, lock down the CSP completely to `none`.
