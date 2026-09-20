## 2024-10-24 - Keyboard accessible scroll containers
**Learning:** Scrollable containers (`overflow: auto`) are not accessible by keyboard unless explicitly given a `tabindex="0"`, `role="region"`, and an `aria-label`.
**Action:** Always add these attributes to `.scroller` or similar overflowing elements to ensure users without a mouse can scroll through table content.
