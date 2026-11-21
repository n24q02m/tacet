
## 2024-10-06 - Improve keyboard focus visibility
**Learning:** Using explicit `:focus-visible` with `outline-offset` provides a highly visible focus ring for keyboard users without disrupting the visual experience for mouse users, a critical pattern for accessibility in minimal designs.
**Action:** Always include `:focus-visible` styles with an offset instead of relying on default browser outlines or removing outlines entirely on interactive elements.
