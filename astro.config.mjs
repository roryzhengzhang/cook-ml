// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeMathNowrap from './scripts/rehype-math-nowrap.mjs';
import { seriesSidebar } from './scripts/series-sidebar.mjs';

export default defineConfig({
  site: 'https://cook-ml.com',
  trailingSlash: 'always',
  // No floating Astro toolbar in the dev server.
  devToolbar: { enabled: false },
  markdown: {
    // Tutorials use $...$ and $$...$$ math, rendered with KaTeX at build time.
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [[rehypeKatex, { strict: false }], rehypeMathNowrap],
    }),
  },
  integrations: [
    starlight({
      title: 'Cook ML',
      description: 'Clear, worked-through tutorials on machine learning, from first principles to practical systems.',
      logo: { src: './src/assets/logo.svg' },
      favicon: '/favicon.svg',
      social: [{ icon: 'github', label: 'Cook ML on GitHub', href: 'https://github.com/roryzhengzhang/cook-ml' }],
      lastUpdated: true,
      pagination: true,
      routeMiddleware: './src/routeData.ts',
      // Chapter sections live in the left sidebar, so the right-hand table of contents is off.
      tableOfContents: false,
      customCss: [
        '@fontsource-variable/inter',
        '@fontsource-variable/jetbrains-mono',
        'katex/dist/katex.min.css',
        './src/styles/custom.css',
      ],
      components: {
        Header: './src/components/Header.astro',
        Footer: './src/components/Footer.astro',
        Hero: './src/components/Hero.astro',
      },
      head: [
        // Restore a collapsed contents sidebar before the page paints (see Header.astro).
        {
          tag: 'script',
          content:
            "try{if(localStorage.getItem('cm-sidebar')==='collapsed')document.documentElement.dataset.cmSidebar='collapsed'}catch(e){}",
        },
        { tag: 'meta', attrs: { property: 'og:image', content: 'https://cook-ml.com/og.png' } },
      ],
      sidebar: [
        { label: 'Tutorials', link: '/tutorials/' },
        {
          label: 'LLM Fundamentals',
          items: seriesSidebar('llm-fundamentals'),
        },
      ],
    }),
  ],
});
