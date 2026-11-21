## 2024-05-24 - Added visible focus styles
**Learning:** Default browser focus rings are sometimes completely absent or very difficult to see against custom themes (especially in custom-styled headers). Explicit `:focus-visible` outlines using design system tokens (`var(--accent)`) are required for basic keyboard accessibility.
**Action:** Always ensure that interactive elements (links, buttons) have an explicitly defined `outline` with `outline-offset` in their `:focus-visible` state.
