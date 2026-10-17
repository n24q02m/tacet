## 2024-05-30 - Add explicit visible focus indicators
**Learning:** Using only color/brightness changes for focus states makes the UI inaccessible for keyboard users, violating WCAG 1.4.11 Non-text Contrast and 2.4.7 Focus Visible.
**Action:** Always provide a distinct `outline` (e.g., `outline: 2px solid var(--accent); outline-offset: 2px;`) on `:focus-visible` states for interactive elements like buttons and links.
