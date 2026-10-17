---
title: Books & Texts
lede: Textbooks, free online books, standards and documentation for the LULC learning path. Each entry has a short note on why it is here.
---

{% for s in site.data.books.sections %}
## {{ s.name }}
{% for b in s.items %}
<div class="paper">
  <div class="paper-title"><a href="{{ b.url }}">{{ b.title }}</a></div>
  <div class="paper-meta">{{ b.authors }}{% if b.year %} · {{ b.year }}{% endif %} · {{ b.publisher }} · <span class="tag">{{ b.access }}</span><span class="tag">{{ b.status }}</span></div>
  <p class="paper-summary">{{ b.why }}</p>
</div>
{% endfor %}
{% endfor %}

## My texts and notes

This section is for my own texts: chapter notes, lecture notes, drafts and slide decks. To add one, put the file in `texts/` (Markdown or PDF) and add a line below.

| Title | Type | Date | Link |
|---|---|---|---|
| *(example)* Notes on Lillesand et al., Ch. 7: Digital image analysis | Chapter notes | — | *coming soon* |
| *(example)* Spectral indices cheat-sheet | Note | — | *coming soon* |

## Adding a book

Add an entry to `_data/books.json` under the right section. Use these fields: `title`, `authors`, `year`, `publisher`, `url`, `access` (Free / Paid / Free online), `why`, and `status` (To read / Reading / Done / Reference). Please link only to the publisher's page or an official open edition.
