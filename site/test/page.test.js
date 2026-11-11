import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const read = (name) => readFileSync(new URL(`../${name}`, import.meta.url), "utf8");
const html = read("index.html");
const script = read("main.js");
const css = html.slice(html.indexOf("<style>") + 7, html.indexOf("</style>"));

const tokens = Object.fromEntries([...css.matchAll(/--([\w-]+):\s*(#[0-9a-f]{6})\b/gi)].map((m) => [m[1], m[2]]));

function luminance(hex) {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function contrast(a, b) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

test("contrast maths matches the WCAG reference values", () => {
  assert.equal(Math.round(contrast("#000000", "#ffffff")), 21);
  assert.equal(contrast("#ffffff", "#ffffff"), 1);
  assert.ok(Math.abs(contrast("#777777", "#ffffff") - 4.48) < 0.01);
});

test("every element the script looks up exists in the page", () => {
  const ids = [...script.matchAll(/\$\("([\w-]+)"\)/g)].map((m) => m[1]);
  assert.ok(ids.length > 5);
  for (const id of new Set(ids)) assert.ok(html.includes(`id="${id}"`), `missing #${id}`);
});

test("every class the script writes has a style", () => {
  const classes = [...script.matchAll(/class: [`"]([^`"$]+)/g)].flatMap((m) => m[1].trim().split(/\s+/));
  assert.ok(classes.length > 10);
  for (const c of new Set(classes)) assert.match(css, new RegExp(`\\.${c}\\b`), `no style for .${c}`);
});

test("text colours meet WCAG AA (4.5:1) on the page background", () => {
  for (const name of ["ink", "muted", "faint", "accent", "good", "moderate", "usg", "unhealthy"]) {
    assert.ok(tokens[name], `--${name} is not defined`);
    const ratio = contrast(tokens[name], tokens.bg);
    assert.ok(ratio >= 4.5, `--${name} ${tokens[name]} is ${ratio.toFixed(2)}:1 on ${tokens.bg}`);
  }
  assert.ok(contrast(tokens.warn, tokens["warn-soft"]) >= 4.5, "warning text on its background");
  assert.ok(contrast("#ffffff", tokens.ink) >= 4.5, "pressed horizon button");
});

test("Swiss rules: no rounded boxes, no shadows, no gradients outside the loading shimmer", () => {
  const radii = [...css.matchAll(/border-radius:\s*([^;}]+)/g)].map((m) => m[1].trim());
  for (const r of radii) assert.ok(r === "0" || r === "50%", `rounded box: border-radius ${r}`);
  assert.doesNotMatch(css, /box-shadow|text-shadow/);
  const gradients = css.split("\n").filter((line) => line.includes("gradient("));
  assert.ok(gradients.every((line) => line.includes(".skeleton")), "gradient outside .skeleton");
});

test("no inline styles except the loading placeholders' widths", () => {
  const inline = [...html.matchAll(/style="([^"]*)"/g)].map((m) => m[1]);
  for (const s of inline) assert.match(s, /^display:block;width:\d+%$/, `inline style: ${s}`);
});

test("the page loads nothing from another origin", () => {
  assert.doesNotMatch(html, /<(script|link|img)[^>]+(src|href)="https?:/);
});

test("motion can be switched off and small screens are handled", () => {
  assert.match(css, /prefers-reduced-motion: reduce/);
  assert.match(css, /@media \(max-width: 560px\)/);
  assert.match(html, /<meta name="viewport" content="width=device-width, initial-scale=1">/);
});
