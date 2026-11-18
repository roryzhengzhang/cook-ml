import { defineCollection } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';
import { blogSchema } from 'starlight-blog/schema';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { notebookFile } from '../scripts/remark-story-notebook.mjs';

const starlightDocs = docsLoader();

export const collections = {
  docs: defineCollection({
    loader: {
      ...starlightDocs,
      async load(context) {
        // Astro caches rendered Markdown by its digest. The codepads also
        // depend on the notebook and on the plugin that renders it, so edits to either must
        // invalidate that HTML.
        const notebook = createHash('sha256').update(readFileSync(notebookFile))
          .update(readFileSync(new URL('../scripts/remark-story-notebook.mjs', import.meta.url))).digest('hex');
        await starlightDocs.load({
          ...context,
          generateDigest: (contents) => context.generateDigest({ contents, notebook }),
        });
      },
    },
    schema: docsSchema({ extend: (context) => blogSchema(context) }),
  }),
};
