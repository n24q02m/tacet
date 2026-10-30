## 2026-10-30 - Enhance keyboard navigation focus styles
**Learning:** Some interactive elements like navigation links, theme toggles, and buttons lacked distinct focus indicators, making keyboard navigation difficult. Standard `:focus-visible` styles with explicit outlines improve keyboard accessibility without affecting mouse users.
**Action:** Use `outline` and `outline-offset` in conjunction with `:focus-visible` to ensure interactive elements are clearly highlighted when navigated via keyboard.
