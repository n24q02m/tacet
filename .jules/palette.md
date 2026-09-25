## 2024-05-23 - Interactive Focus and Scrollable Regions
**Learning:** Interactive elements relying solely on subtle brightness changes for focus fail WCAG contrast guidelines. Scrollable overflow containers hide their content from keyboard-only users unless explicitly made focusable with `tabindex="0"`, `role="region"`, and an ARIA label.
**Action:** Always use `outline` with `outline-offset` for `:focus-visible`, and ensure any container with `overflow: auto` or `scroll` is properly labeled and focusable for keyboard users.
