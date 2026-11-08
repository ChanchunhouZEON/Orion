// Copyright (c) Chanchunhou. All rights reserved.
// Licensed under the MIT License.
import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import tailwindcss from '@tailwindcss/vite';
export default defineConfig({
  plugins: [svelte(), tailwindcss()], base: './',
  server: { port: 8079, strictPort: true },
});
