## 2025-02-14 - Enforce strict input length limits on FastAPI endpoints
**Vulnerability:** FastAPI endpoints (`AskRequest`, `DistillRequest`, `GraphIngestRequest`) lacked input length limits on user-provided strings and lists, creating a potential Denial of Service (DoS) vulnerability via excessively large payloads.
**Learning:** Pydantic's `Field` defaults do not implicitly enforce limits; bounded parameters (such as `max_length`) must be explicitly declared on fields like strings and sequences to protect endpoints from resource exhaustion.
**Prevention:** To prevent Denial of Service (DoS) risks from unbounded inputs, always use Pydantic's `Field(..., max_length=X)` to enforce strict input length limits on all incoming Pydantic request models in FastAPI.
