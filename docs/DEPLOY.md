# Deploying the project page

The `docs/` folder is a self-contained static site (HTML + CSS, no build
step). Two deployment paths, depending on the URL you want:

## Option A &mdash; `pigformer.github.io` (the URL you asked for)

This URL requires a GitHub user **or** organization named `pigformer`,
which you have to create yourself (GitHub doesn't let me create accounts).

```bash
# 1. On github.com: create an organization (or user) named `pigformer`.
#    https://github.com/organizations/new

# 2. Under that org, create a NEW public repo named exactly:
#    pigformer.github.io
#    (special repo name → served at https://pigformer.github.io)

# 3. Push the contents of this docs/ folder to that repo:
cd docs/
git init -b main
git remote add origin git@github.com:pigformer/pigformer.github.io.git
git add -A
git commit -m "Initial site"
git push -u origin main

# 4. Site is live at https://pigformer.github.io (allow ~1 min after push).
```

## Option B &mdash; `iambashar.github.io/Pigformer` (no new account)

Serve the site from `/docs/` on the existing `iambashar/Pigformer` repo.
No new GitHub account needed.

1. Push this `docs/` folder as part of the main repo (it's already here).
2. On github.com, open the repo &rarr; **Settings &rarr; Pages**.
3. Under **Build and deployment**:
   - Source: **Deploy from a branch**
   - Branch: **main** &middot; folder: **/docs**
4. Save. Site is live at `https://iambashar.github.io/Pigformer/`
   (allow ~1&ndash;2 min for the first build).

## Local preview

```bash
cd docs/
python -m http.server 8000
# Open http://localhost:8000
```

## File map

| File | Purpose |
|---|---|
| `index.html` | The site. Header, abstract, pipeline figure, results table, BibTeX. |
| `styles.css` | All styling, no JS framework. |
| `pipeline.png` | Figure 1 from the paper. |
| `.nojekyll` | Tells GitHub Pages to serve files as-is, no Jekyll processing. |

To update content, edit `index.html` directly and re-push. No build needed.
