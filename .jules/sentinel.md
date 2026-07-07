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
## 2026-07-07 - [Enforce String Length Limits in FastAPI Pydantic Models]
**Vulnerability:** Unbounded string inputs in Pydantic models (e.g. `AskRequest`, `DistillRequest`, `GraphIngestRequest`) can cause excessive memory usage or Denial of Service (DoS).
**Learning:** Pydantic `BaseModel` fields defined simply as `str` or `list[str]` do not enforce length constraints by default. Using `Field` to set `max_length` and `typing.Annotated` to type-hint list elements secures these boundaries. When constraining collections in Pydantic that are required, make sure to use `Field(..., max_length=X)` to avoid unintentionally making them optional (e.g., using `default_factory`).
**Prevention:** Always use `Field(..., max_length=X)` for base fields and `Annotated[str, Field(max_length=X)]` for nested fields inside collections exposed in external endpoints to prevent memory and DoS risks.
