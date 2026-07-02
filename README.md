# loganrudd.github.io

Personal site for Logan Rudd — ML infrastructure engineer for physical systems.

Live at [loganrudd.github.io](https://loganrudd.github.io).

## Stack

Jekyll, served by GitHub Pages. Theme is a customized fork of [Galada](https://github.com/artemsheludko/galada) by Artem Sheludko.

## Local development

```
bundle install
bundle exec jekyll serve
```

Site is served at `http://localhost:4000`.

## Structure

- `index.html` — home page: positioning statement + project list
- `_posts/` — the three project write-ups (problem / architecture / what was hard)
- `_pages/about.md` — background
- `_sass/`, `_includes/` — theme partials and styles
- `img/*.svg` — hand-authored architecture headers for each post
