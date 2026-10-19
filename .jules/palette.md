## 2024-05-18 - Add skip-to-content link
**Learning:** The Astro template lacked a skip-to-content link, creating a barrier for keyboard and screen-reader users to bypass the site header and navigate directly to the main content.
**Action:** Always include a visually-hidden skip link that becomes visible on focus at the start of the `<body>` element, and ensure the target main element has `tabindex="-1"` and `outline: none` so it can be focused programmatically without showing a focus ring.
