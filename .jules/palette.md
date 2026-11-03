## 2024-11-03 - Skip-to-content link pattern
**Learning:** The base layout lacked a skip-to-content link, which forces keyboard and screen reader users to repeatedly navigate through the header and theme toggle on every page traversal. This is a critical baseline accessibility pattern for layout components.
**Action:** Implement a focus-revealed skip link at the top of the DOM targeting `#main-content` in all top-level layout templates going forward.
