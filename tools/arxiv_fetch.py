#!/usr/bin/env python3
"""Fetch real metadata for one or more arXiv IDs and merge it into tools/arxiv_metadata.json.

Usage:
    python3 tools/arxiv_fetch.py 1709.00029 2402.13791
    python3 tools/arxiv_fetch.py --search 'abs:"land cover" AND abs:SHAP' --max 10   # just prints results

After fetching, add each ID with a topic key and a short summary to PAPERS in tools/curation.py,
then run:  python3 tools/build_data.py
Please respect the arXiv API terms (https://info.arxiv.org/help/api/tou.html): at most 1 request every 3 seconds.
"""
import argparse, json, os, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

API = "https://export.arxiv.org/api/query?"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
META = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arxiv_metadata.json")


def query(params):
    url = API + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as r:
        root = ET.fromstring(r.read())
    out = []
    for e in root.findall("a:entry", NS):
        full = e.find("a:id", NS).text.split("/abs/")[-1]
        base = full.rsplit("v", 1)[0]
        doi, jr = e.find("arxiv:doi", NS), e.find("arxiv:journal_ref", NS)
        out.append(dict(
            id=base, version=full,
            title=" ".join(e.find("a:title", NS).text.split()),
            authors=[a.find("a:name", NS).text for a in e.findall("a:author", NS)],
            published=e.find("a:published", NS).text[:10],
            summary=" ".join(e.find("a:summary", NS).text.split()),
            doi=doi.text if doi is not None else None,
            journal_ref=" ".join(jr.text.split()) if jr is not None else None,
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--search")
    ap.add_argument("--max", type=int, default=10)
    a = ap.parse_args()
    if a.search:
        for p in query({"search_query": a.search, "max_results": a.max}):
            print(f"{p['id']}  {p['published'][:4]}  {p['title']}  ({', '.join(p['authors'][:3])})")
        return
    if not a.ids:
        ap.error("give arXiv IDs or --search")
    meta = json.load(open(META)) if os.path.exists(META) else {}
    found = query({"id_list": ",".join(a.ids), "max_results": len(a.ids)})
    for p in found:
        meta[p["id"]] = p
        print(f"OK  {p['id']}  {p['title']}")
    missing = set(a.ids) - {p["id"] for p in found}
    for m in sorted(missing):
        print(f"NOT FOUND  {m}  (check the ID)")
    json.dump(meta, open(META, "w"), indent=1, ensure_ascii=False)
    time.sleep(3)


if __name__ == "__main__":
    main()
