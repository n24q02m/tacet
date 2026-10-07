## 2025-01-21 - [Keyboard accessibility]
**Learning:** Found that scrollable containers lacked keyboard focus, preventing keyboard-only users from scrolling content. Focus indicators were also too subtle (brightness only), failing WCAG contrast guidelines.
**Action:** Applied `tabindex="0"`, `role="region"`, and context-appropriate `aria-label`s to all scrollable `.scroller` containers. Added explicit `outline` and `outline-offset` properties for `:focus-visible` states across interactive elements (`.btn`, `.nav a`, `.theme-toggle`, `.scroller`) to ensure distinct visual feedback.
