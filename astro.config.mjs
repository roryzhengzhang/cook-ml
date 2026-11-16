// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightBlog from 'starlight-blog';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { seriesSidebar } from './scripts/series-sidebar.mjs';

export default defineConfig({
  site: 'https://cook-ml.com',
  trailingSlash: 'always',
  markdown: {
    // Tutorials use $...$ and $$...$$ math, rendered with KaTeX at build time.
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [[rehypeKatex, { strict: false }]],
    }),
  },
  integrations: [
    starlight({
      title: 'Cook ML',
      description: 'Clear, worked-through tutorials and notes on machine learning and large language models.',
      logo: { src: './src/assets/logo.svg' },
      favicon: '/favicon.svg',
      social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/roryzhengzhang' }],
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
      },
      head: [
        { tag: 'meta', attrs: { property: 'og:image', content: 'https://cook-ml.com/og.png' } },
      ],
      sidebar: [
        { label: 'Tutorials', link: '/tutorials/' },
        {
          label: 'LLM Fundamentals',
          items: seriesSidebar('llm-fundamentals'),
        },
      ],
      plugins: [
        starlightBlog({
          title: 'Blog',
          navigation: 'none',
          postCount: 10,
          recentPostCount: 8,
          metrics: { readingTime: true },
          authors: {
            zheng: {
              name: 'Zheng Zhang',
              title: 'ML Engineer',
              url: 'https://github.com/roryzhengzhang',
            },
          },
        }),
      ],
    }),
  ],
});
