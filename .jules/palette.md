## 2024-05-14 - Skip-to-content links require explicit focus states
**Learning:** Adding a basic skip-link isn't enough; it must have a focus-visible state to actually show up on screen when a keyboard user tabs to it, because it is otherwise visually hidden.
**Action:** Always test skip links with Playwright's `.focus()` state or manual keyboard navigation to ensure the `:focus` or `:focus-visible` styles override the absolute off-screen positioning.
