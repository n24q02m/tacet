import { defineConfig } from "astro/config";

// Published at https://n24q02m.github.io/tacet/
// The paper itself is NOT authored here: `latexml` renders paper/main.tex into
// public/paper/ during the deploy workflow, so the reading copy can never drift
// from the LaTeX source. Only the landing page is hand-written.
export default defineConfig({
  site: "https://n24q02m.github.io",
  base: "/tacet",
  trailingSlash: "always",
  build: {
    format: "directory",
  },
});
