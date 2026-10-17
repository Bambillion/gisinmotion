# GIS in Motion

A learning library on **land use / land cover (LULC) classification** with GIS, remote sensing and machine learning, by Temitope Ogungbuji.

- **Reading list:** 24 real arXiv papers, grouped by topic (metadata from the arXiv API)
- **Articles:** a methods-review and tutorial article (with MathJax equations and a Python appendix), plus notes on an external, published Lugbe, Abuja reference study (Ekweozoh et al., 2026, *Global Scientific Journal* 14(8); summary checked against the published PDF)
- **Books & Texts:** verified textbooks, free online books, standards and documentation, plus a slot for your own notes

The site is plain [Jekyll](https://jekyllrb.com/), which GitHub Pages builds automatically. It needs no theme gem and no plugins.

```
gisinmotion/
├── _config.yml              # site title, baseurl (/gisinmotion), markdown + MathJax settings
├── _layouts/                # default, page, article
├── _includes/               # head (CSS + MathJax), nav, footer
├── _data/
│   ├── papers.json          # reading list (generated; see below)
│   └── books.json           # Books & Texts entries
├── assets/css/style.css     # the whole theme
├── assets/data/papers.json  # downloadable copy of the reading list
├── index.md                 # Home
├── reading-list.html        # Reading List (rendered from _data/papers.json)
├── articles/
│   ├── index.md             # Articles index
│   ├── explainable-multisource-lulc-framework.md   # generated from tools/article_framework.src.md
│   └── lugbe-case-study.md
├── books.md  about.md  404.html
├── papers.bib               # BibTeX for the reading list
├── papers.json              # same data as _data/papers.json (not published)
├── code/lulc_pipeline.py    # runnable tutorial pipeline (synthetic demo)
├── texts/                   # put your own notes/texts here
├── tools/                   # helper scripts (not published)
└── screenshots/             # local preview screenshots (not published)
```

## Publish on GitHub Pages

1. Create a **public** repository named `gisinmotion` on GitHub.
2. Push this folder to it:
   ```bash
   cd gisinmotion
   git init -b main
   git add .
   git commit -m "GIS in Motion: first version"
   git remote add origin https://github.com/<your-username>/gisinmotion.git
   git push -u origin main
   ```
3. On GitHub, open **Settings → Pages**. Under *Build and deployment*, choose **Deploy from a branch**, set the branch to `main` and the folder to `/ (root)`, then save.
4. After a minute or two the site is live at `https://bambillion.github.io/gisinmotion/`.
   `url` in `_config.yml` is already set to `https://bambillion.github.io`.

`baseurl` is already `/gisinmotion`. If you rename the repository, change `baseurl` to match. If you publish it as a user site (`<your-username>.github.io`), set `baseurl: ""`.

## Preview locally

```bash
# Option A: exact GitHub Pages versions (Ruby + Bundler; uses the included Gemfile)
bundle install
bundle exec jekyll serve          # open http://localhost:4000/gisinmotion/

# Option B: any Jekyll install, then serve the static output with Python
JEKYLL_NO_BUNDLER_REQUIRE=true jekyll build -d _site/gisinmotion
python3 -m http.server -d _site 8000   # open http://localhost:8000/gisinmotion/
```

Both options were tested: Jekyll 3.10.0 (the github-pages gem) and Jekyll 4.4.1.

## Add a paper to the reading list

Each paper must be a real arXiv record. The scripts pull the metadata from the arXiv API, so titles, authors and IDs are never typed by hand.

```bash
# 1. Optionally search arXiv
python3 tools/arxiv_fetch.py --search 'abs:"land cover" AND abs:Sentinel-2' --max 10
# 2. Fetch metadata for the IDs you want (saved to tools/arxiv_metadata.json)
python3 tools/arxiv_fetch.py 2401.02424
# 3. In tools/curation.py, add an entry to PAPERS:
#      "2401.02424": ("benchmarks", "One or two sentences in your own words, based on the abstract."),
#    (topic keys are listed in TOPICS; add a new topic tuple there if needed)
# 4. Regenerate _data/papers.json, papers.json, assets/data/papers.json and papers.bib
python3 tools/build_data.py
```

You can also edit `_data/papers.json` by hand, but your edits will be overwritten the next time you run `build_data.py`.

## Add an article

1. Create `articles/<short-name>.md` with front matter:
   ```yaml
   ---
   title: "Your title"
   subtitle: "Optional subtitle"
   kind: Methods note          # small label above the title
   authors: Temitope Ogungbuji
   status: Draft               # or Published
   updated: 2026-10-07
   math: true                  # only if the page uses equations
   ---
   ```
2. Write in Markdown. For equations, put display math between `$$` lines (each `$$` on its own line, with blank lines around the block) and inline math as `$$x^2$$` inside a sentence.
3. Add a card for the article to `articles/index.md`.

The framework article is generated from `tools/article_framework.src.md` so that its Python appendix stays in sync with `code/lulc_pipeline.py`. Edit the source file, then run `python3 tools/build_article.py`.

## Add a book or text

Add an entry to `_data/books.json` (fields: `title`, `authors`, `year`, `publisher`, `url`, `access`, `why`, `status`). Put your own notes in `texts/` and list them in the table on `books.md`.

## Run the tutorial code

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install numpy rasterio scikit-learn lightgbm shap matplotlib
python code/lulc_pipeline.py      # end-to-end demo on SYNTHETIC data (numbers are meaningless)
```

## Content policy

- The reading list contains only papers verified through the arXiv API, and every abs link was checked.
- Articles are reviews and tutorials. They do not report results that were not obtained. Numbers from the external Lugbe case study (Ekweozoh et al., 2026; not Temitope's work) are always attributed to it, and anything else is a labelled placeholder.
