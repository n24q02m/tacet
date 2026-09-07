## 2024-09-06 - Make scrollable tables keyboard accessible
**Learning:** Scrollable data tables that clip overflow content are inaccessible to keyboard users unless they are explicitly placed in the tab order (e.g., via tabindex=0) and given an accessible name (e.g., aria-label).
**Action:** Always add tabindex="0", role="region", an aria-label, and explicit focus-visible styles to containers with overflow-x: auto.
