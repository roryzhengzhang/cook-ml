// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import remarkStoryNotebook, { notebookProjects, rlNotebook } from './scripts/remark-story-notebook.mjs';
import remarkAttentionPlayground, { attentionFiles } from './scripts/remark-attention-playground.mjs';
import rehypeKatex from 'rehype-katex';
import rehypeMathNowrap from './scripts/rehype-math-nowrap.mjs';
import { notebookSidebar, seriesSidebar } from './scripts/series-sidebar.mjs';

export default defineConfig({
  site: 'https://cook-ml.com',
  trailingSlash: 'always',
  // No floating Astro toolbar in the dev server.
  devToolbar: { enabled: false },
  markdown: {
    // Tutorials use $...$ and $$...$$ math, rendered with KaTeX at build time.
    processor: unified({
      remarkPlugins: [remarkMath, remarkStoryNotebook, remarkAttentionPlayground],
      rehypePlugins: [[rehypeKatex, { strict: false }], rehypeMathNowrap],
    }),
  },
  integrations: [
    {
      name: 'cook-ml-content',
      hooks: {
        'astro:config:setup': ({ addWatchFile, config }) => {
          for (const project of notebookProjects) {
            addWatchFile(project.file);
            if (project.readme) addWatchFile(new URL(project.readme, config.publicDir));
            addWatchFile(new URL(`./src/content/docs/${project.pageFile}`, import.meta.url));
          }
          addWatchFile(new URL('./scripts/remark-story-notebook.mjs', import.meta.url));
          for (const file of attentionFiles) addWatchFile(file);
        },
      },
    },
    starlight({
      title: 'Cook ML',
      description: 'Clear, worked-through tutorials on machine learning, from first principles to practical systems.',
      logo: { src: './src/assets/logo.svg' },
      favicon: '/favicon.svg',
      social: [{ icon: 'github', label: 'Cook ML on GitHub', href: 'https://github.com/roryzhengzhang/cook-ml' }],
      lastUpdated: true,
      pagination: true,
      routeMiddleware: './src/routeData.ts',
      expressiveCode: {
        themes: ['starlight-dark'],
        useStarlightDarkModeSwitch: false,
        styleOverrides: {
          codeBackground: '#211c2e',
          borderColor: '#463950',
          borderRadius: '0.5rem',
        },
      },
      // Chapter sections live in the left sidebar, so the right-hand table of contents is off.
      tableOfContents: false,
      customCss: [
        '@fontsource-variable/inter',
        '@fontsource-variable/jetbrains-mono',
        'katex/dist/katex.min.css',
        './src/styles/custom.css',
        './src/styles/notebook-panel.css',
        './src/styles/notebook-reader.css',
        './src/styles/link-preview.css',
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
        // Hover previews for links in tutorial text (see public/link-preview.js).
        { tag: 'script', attrs: { src: '/link-preview.js', defer: true } },
        { tag: 'script', attrs: { src: '/story-codepad.js', defer: true } },
      ],
      sidebar: [
        { label: 'Tutorials', link: '/tutorials/' },
        {
          label: 'LLM Fundamentals',
          items: seriesSidebar('llm-fundamentals'),
        },
        {
          label: 'LLM Deep Dive',
          items: seriesSidebar('llm-deep-dive'),
        },
        {
          label: 'Training Tool-Using Agent With RL',
          items: [
            ...seriesSidebar('rl-for-llm-agents'),
            notebookSidebar(rlNotebook),
          ],
        },
      ],
    }),
  ],
});
