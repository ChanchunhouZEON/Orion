# Orion Astraeus Website

[Open the homepage](https://chanchunhouzeon.github.io/Orion/) · [Orion library](https://github.com/ChanchunhouZEON/Orion/tree/main)

This independent `pages` branch owns the project website and its curated
measurement archive. It has no Rust source checkout. Website updates stay on
this branch; do not merge it into `main`.

```sh
git clone --single-branch --branch pages https://github.com/ChanchunhouZEON/Orion.git orion-pages
cd orion-pages/orion-site
npm ci
npm test
npm run build -- --base /Orion/
npm run dev
```

Pushes to `pages` run the tests and production build before GitHub Pages deploys
the artifact. Pull requests targeting `pages` validate without deploying.
For a manual deployment, select `pages` in the Orion Website Pages workflow.
The deployment environment only permits the `pages` branch.

See [orion-site/README.md](orion-site/README.md) for the UI, data protocol and
source layout. Keep the Pages workflow definition synchronized with the copy
on `main`, which makes manual runs discoverable from the default branch.
