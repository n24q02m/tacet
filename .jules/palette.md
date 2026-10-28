## 2024-05-31 - Keyboard navigation and overflow tables
**Learning:** Data tables that overflow horizontally (like `.scroller`) trap keyboard users if they lack `tabindex="0"`. Additionally, skip links are critical for fast keyboard navigation but were missing.
**Action:** Always ensure horizontal scroll containers have `tabindex="0"`, `role="region"`, and an `aria-label`. Always include a skip-to-content link in the base layout.
