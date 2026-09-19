# 🐾 Petbarn Product Assistant

An agentic Streamlit chatbot that answers questions about [Petbarn](https://www.petbarn.com.au)
products by calling tools on demand — scraping product pages and retrieving real customer reviews
while the conversation is happening, rather than answering from a prompt stuffed with data.

**Live app:** _<!-- DEPLOY_URL -->deployment pending_

```
"What are people saying about the price and quality of the Breeders Choice cat litter?"
"Can you compare the reviews between the Black Hawk lamb and rice and the Prime100 kangaroo roll?"
"List the main pros and cons based on recent customer feedback for the NexGard Spectra."
```

---

## Quick start

```bash
git clone <this-repo> && cd petbarn-assistant
python -m venv .venv && .venv/Scripts/activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add a free Groq key
streamlit run streamlit_app.py
```

A free Groq key comes from [console.groq.com/keys](https://console.groq.com/keys). If no key is
configured the app still runs — it shows the catalog and explains what it needs — and a key can be
pasted into the sidebar at any time.

The repo ships with a working dataset, so nothing needs scraping before the first run.

---

## How it works

```
  Streamlit chat UI  ──►  PetbarnAgent  ──►  4 tools  ──►  cache ─► live fetch ─► snapshot
  (shows tool trace)      (Groq, bounded      (SKU-keyed)        petbarn.com.au / Bazaarvoice
                           tool loop)
```

The model is given four tools and decides which to call. Nothing about a product reaches an answer
unless a tool returned it during that conversation.

### The data sources

Working out where Petbarn's data actually lives was most of the design work, so it is worth
recording what is true of the site:

| What | Where it comes from |
|---|---|
| **Specifications, price, brand** | The **schema.org JSON-LD** embedded server-side in each `/p/<slug>` page. Petbarn's storefront is a Next.js app that assembles most of its markup in the browser, but it publishes complete JSON-LD for search engines. Parsing that is far steadier than scraping rendered HTML, and it includes the **loyalty member price**, availability, barcode, delivery options and return policy. |
| **Customer reviews** | **Bazaarvoice**, the platform Petbarn renders reviews with. Reviews are lazy-loaded client-side, so they are *not* in the page HTML; they come from Bazaarvoice's public display API, keyed by Petbarn's SKU. It returns review text, titles, dates, verified-purchaser badges, helpful votes, the full star distribution, and per-aspect ratings for **Quality**, **Value for money** and **Pet satisfaction**. |
| **Which products to cover** | Bazaarvoice's product listing sorted by review count, which is how `scripts/build_snapshot.py` finds products with enough written feedback to be worth discussing. |

Two traps in the data that the parser handles explicitly:

- **Multi-size products** are published as `@type: "ProductGroup"` with **no price of their own** —
  each entry in `hasVariant` carries its own SKU, size and price. Treating those pages like normal
  products yields a catalog of items with no price.
- **Slugs cannot be guessed.** A plausibly-constructed product URL returns 404. Product URLs come
  from the sitemap or from Bazaarvoice's `ProductPageUrl`, never from string assembly.

There is also a public GraphQL endpoint at `mesh.petbarn.com.au`; it returns **403** to anything
that is not the storefront, and it is not needed.

### The four tools

The brief asks for two. Two more exist because they measurably improve the answers:

| Tool | Purpose |
|---|---|
| `search_catalog` | Turns "the kangaroo roll" into a SKU. Called first, always. Every other tool is keyed by SKU, and a model left to guess a SKU will invent one and then answer confidently about a product that does not exist. Returns *ranked* candidates with confidence scores, so a genuinely ambiguous request becomes a clarifying question instead of a coin flip. |
| `get_product_details` | Specifications, regular and member price, brand, category, size, availability, barcode, description, other sizes, delivery options. |
| `get_product_reviews` | Individual reviews with ratings, dates, verified-purchaser flags and helpful votes, plus the aggregate picture: average, star distribution, recommend rate, and Petbarn's own Quality / Value / Pet-satisfaction averages. Filterable by rating and sortable, so "what do the one-star reviews say" is answerable. |
| `analyze_review_sentiment` | Sentiment broken down by theme, with counts, average stars and verbatim quotes per theme, plus ranked pros and cons. |

### Sentiment is computed, not guessed

The sample questions ask about sentiment *per theme* — "the price **and** quality of X", "pros and
cons". A single polarity score cannot express that shoppers love a food's quality while calling it
expensive, which is the dominant pattern in this data: Value for money consistently averages below
Quality on almost every product in the catalog.

So `petbarn/sentiment.py` does the analysis itself:

- **VADER** scores polarity sentence by sentence. It is rule-based, so the same review always
  produces the same number, and the reviewer's words drive the result rather than the model's
  impression of them.
- Its lexicon is extended with ~37 pet-retail terms VADER lacks (`overpriced`, `clumping`, `dusty`,
  `fussy`, `resealable`, `diarrhoea`…). Terms VADER already scores are left untouched.
- Sentences are bucketed into themes by keyword, so "great food but pricey" counts positively for
  quality and negatively for price.
- **Star ratings are reported alongside polarity, never blended into it.** Where the two disagree —
  a five-star review with a specific complaint — that is counted and surfaced as
  `star_vs_text_disagreements` rather than averaged away.

An aspect needs at least three mentions before it can become a pro or a con, and the thresholds for
the two do not overlap, so an aspect is never presented as both.

The model's job is then to narrate evidence it was handed, with quotes, not to infer sentiment from
a wall of text.

### Reliability

Every tool resolves its data the same way:

1. **A recent local cache** (`.cache/`, 6-hour TTL, keyed by request hash).
2. **A live fetch** through one shared session: 3 retries with exponential backoff, ~1 request per
   second per host, 15-second timeout, realistic user agent.
3. **The snapshot committed to this repo** (`data/snapshot/`) — 10 products and 1,474 real written reviews (drawn from 7,616 underlying ratings).

The consequence is that a blocked request, a rotated Bazaarvoice key, a site outage or a host with
no egress degrades the *freshness* of an answer rather than the ability to answer at all. If the
network fails but an expired cache entry exists, the expired entry is served rather than the error.

Which layer served each call travels back in the payload as `source`, is rendered as a badge in the
tool trace, and the assistant is instructed to say so out loud when an answer rests on the snapshot.
Toggle **Fetch live data** off in the sidebar (or set `PETBARN_LIVE=0`) to force the offline path.

### The agent loop

Groq, defaulting to `llama-3.3-70b-versatile` for its **parallel tool calling** — comparing two
products issues both review lookups in one round, and they execute concurrently in a thread pool
rather than one after the other.

The loop is bounded at 5 rounds; on the last round the model is asked once more with tools disabled,
which turns a runaway into an answer instead of a timeout. Tool failures are passed back to the
model as tool results so it can tell the user what it could not find out. Tool messages are not
carried between turns, keeping context small and preventing stale tool output from leaking into a
later question.

---

## Scraping conduct

- Only `/p/` product pages are requested. `robots.txt` permits them, and none of its `Disallow`
  paths are visited.
- Requests are throttled to roughly one per second per host and cached for six hours, so repeated
  questions about the same product generate no traffic at all.
- Review retrieval uses the same public, read-only Bazaarvoice display endpoint the product page
  itself calls from every visitor's browser. The display passkey that authorises it is served in
  Petbarn's own client-side bundle rather than being a private credential; it is
  **re-discovered at runtime** from that bundle (with a last-known value as a fallback and a
  `BV_PASSKEY` override) so a rotation on Petbarn's side does not break the app.
- Nothing is written, submitted or purchased. No accounts, carts or checkout paths are touched.

This is a technical demonstration, not an approved integration. Anything beyond that would want
Petbarn's permission.

---

## Testing

```bash
python scripts/smoke_test.py       # all 4 tools, live + offline, 94 assertions, no API key needed
python scripts/agent_test.py       # the brief's questions through the real model (needs a key)
python scripts/agent_test.py --offline           # same, forced onto the snapshot
python scripts/build_snapshot.py --verify        # check the committed dataset is intact
```

`smoke_test.py` is the one that matters most: the agent is only as good as the tools beneath it, and
every failure worth guarding against lives there — a scraper that silently returns no price, a
snapshot that cannot satisfy a filter the live API can, a payload missing the provenance the UI
renders. It costs nothing to run.

`agent_test.py` covers what the smoke test cannot: whether the *model* picks the right tools. It
prints the full trace per question, including the cases a demo tends to trip on — an off-catalog
product, a request for only the critical reviews, and a bare "what can you help me with".

### Re-ingesting the data

```bash
python scripts/build_snapshot.py             # rebuild catalog + snapshot from the live site
python scripts/build_snapshot.py --count 12  # a different catalog size
```

Products are selected mechanically rather than by hand: drawn in descending order of review count,
then filtered to those with a resolvable price and 40+ written reviews, capped at one product per
brand and two per category, with multi-buy bundles excluded (they duplicate a single-unit item's
reviews). That is what produces a catalog spanning 10 brands and 6 categories instead of ten
variants of the same flea chew. Raw responses are archived gzipped under `data/raw/` so the parsers
can be re-run and audited without touching the network.

---

## Deployment (Streamlit Community Cloud)

1. Push this repo to a **public** GitHub repository.
2. At [share.streamlit.io](https://share.streamlit.io): **New app** → select the repo → main file
   `streamlit_app.py` → **Advanced settings → Python 3.12**.
3. In **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
4. Deploy, then put the public URL at the top of this file.

`.cache/` is ephemeral on a hosted container, which is expected — the snapshot is the durable
fallback, not the cache.

---

## Project layout

```
streamlit_app.py            Chat UI, sidebar, tool-trace rendering
petbarn/
  config.py                 Endpoints, tuning, env overrides — no magic strings elsewhere
  models.py                 Dataclasses shared by every layer; JSON round-trips both ways
  textutils.py              Text repair (Petbarn's copy contains stray U+0092 control chars)
  http.py                   One session: retries, throttling, disk cache, stale-while-broken
  scraper.py                Product page → JSON-LD → ProductDetails (Product + ProductGroup)
  reviews.py                Bazaarvoice client + runtime passkey discovery
  sentiment.py              VADER + retail lexicon + aspect mining → pros and cons
  catalog.py                Curated catalog and fuzzy "what did they mean" → SKU
  tools.py                  Tool schemas, payload shaping, the 3-layer fallback chain
  agent.py                  Groq tool loop: parallel calls, bounded rounds, trace capture
data/
  catalog.json              The 10 curated products
  snapshot/                 Offline fallback: product + review JSON per SKU
  raw/                      Gzipped raw HTML and API responses, for auditing
scripts/
  build_snapshot.py         Ingestion and dataset verification
  smoke_test.py             Tool-level tests, live and offline
  agent_test.py             End-to-end tests through the model
```

---

## Limitations

- **Ten products.** `search_catalog` says so plainly and lists what it does cover rather than
  improvising. Widening it is a matter of re-running the ingestion with a larger `--count`.
- **No stock, order or delivery data**, and no veterinary advice — the assistant is instructed to
  refer those to a vet.
- **Review coverage is capped** at 150 written reviews per product. On a product with 1,200+
  reviews, sentiment describes a recent sample rather than the entire history; the tool reports how
  many it analysed so the answer can be honest about it.
- **Aspect matching is keyword-based**, so a sentence can land in the wrong theme — "waste of money"
  in a review praising a *different* brand reads as a price complaint. Quotes are returned with every
  finding precisely so a reader can judge for themselves.
- **Answers are not streamed.** The tool loop needs each round's result before the next, and Groq is
  fast enough that a live tool-progress indicator is the better trade.
- **No automated tests for the UI layer** beyond a render check via Streamlit's `AppTest`.
