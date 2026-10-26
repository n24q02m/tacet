## 2024-06-25 - Added Skip to Content Link
**Learning:** Found that the Astro site lacked a skip-to-content link, which is a key accessibility feature for keyboard users to bypass repetitive navigation links.
**Action:** Added a hidden "Skip to content" link that becomes visible on focus at the top of `Base.astro`, pointing to `id="main-content"` on the main tag, using existing styling tokens.
