import { defineCollection } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';
import { blogSchema } from 'starlight-blog/schema';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { notebookProjects } from '../scripts/remark-story-notebook.mjs';
import { attentionFiles, attentionMarker } from '../scripts/remark-attention-playground.mjs';

const starlightDocs = docsLoader();

export const collections = {
  docs: defineCollection({
    loader: {
      ...starlightDocs,
      async load(context) {
        // Astro caches rendered Markdown by its digest. The codepads also
        // depend on the notebook and on the plugin that renders it, so edits to either must
        // invalidate that HTML.
        const hash = createHash('sha256');
        for (const { file } of notebookProjects) hash.update(readFileSync(file));
        const notebook = hash.update(readFileSync(new URL('../scripts/remark-story-notebook.mjs', import.meta.url))).digest('hex');
        const playgroundHash = createHash('sha256');
        for (const file of attentionFiles) playgroundHash.update(readFileSync(file));
        const attention = playgroundHash.digest('hex');
        await starlightDocs.load({
          ...context,
          generateDigest: (contents) => {
            const hasPlayground = typeof contents === 'string' && contents.includes(attentionMarker);
            return context.generateDigest(hasPlayground ? { contents, notebook, attention } : { contents, notebook });
          },
        });
      },
    },
    schema: docsSchema({ extend: (context) => blogSchema(context) }),
  }),
};
