# Security Policy

## Supported Versions

TACET is a research reference implementation. Security fixes are applied to the
latest released version on the `main` branch.

| Version | Supported |
| ------- | --------- |
| latest  | yes       |
| older   | no        |

## Reporting a Vulnerability

Please report security issues privately rather than opening a public issue:

- Use GitHub's **private vulnerability reporting** (the "Report a vulnerability"
  button under the Security tab), or
- Email **quangminh2402.dev@gmail.com**.

Include a description, reproduction steps, and the affected version/commit. You
can expect an acknowledgement within a few days.

## Scope

This repository contains research code and no production secrets. No API keys,
tokens, or credentials are committed; configuration is read from environment
variables at runtime (see `tacet.serve.settings`).
