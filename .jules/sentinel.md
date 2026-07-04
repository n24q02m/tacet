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

## 2026-07-03 - Prevented DoS from Unbounded Pydantic Inputs
**Vulnerability:** The FastAPI request schemas in `src/tacet/serve/server.py` (`AskRequest`, `DistillRequest`, `GraphIngestRequest`) lacked length constraints on strings and lists, creating a Denial of Service (DoS) vulnerability via memory exhaustion from maliciously large payloads.
**Learning:** Pydantic models by default do not enforce strict limits on the sizes of strings or lists unless explicitly requested. In public-facing APIs, this allows attackers to send huge payloads that crash the server or dramatically increase latency.
**Prevention:** Used Pydantic's `Field(..., max_length=X)` to enforce strict bounds on all strings (e.g., 256 for names/relations) and lists (e.g., 10,000 for batch ingests) to fail securely at the validation layer before any application logic executes.

## $(date +%Y-%m-%d) - Refined DoS Prevention on Nested Types
**Vulnerability:** The previous DoS mitigation added max_length to list fields (e.g. `answers: list[str] = Field(max_length=100)`) but left the nested elements (the strings themselves) unbounded. An attacker could still bypass the list limit by passing a single massive string inside a 1-element list.
**Learning:** Pydantic Field constraints on a collection like `list` or `tuple` only constrain the outer collection (number of items), not the inner items.
**Prevention:** Used `typing.Annotated` in combination with `Field` to bound the nested string elements (e.g. `list[Annotated[str, Field(max_length=1000)]]`) ensuring full coverage.
