import { defineRouteMiddleware } from '@astrojs/starlight/route-data';
import type { SidebarEntry, SidebarLink } from '@astrojs/starlight/route-data';

// Tutorial sidebars list each chapter's sections as anchor links, and each chapter
// starts with an "Introduction" link. Without this, Previous/Next would step through
// those anchors. Here pagination moves between pages, labelled with chapter names.

const withoutHash = (href: string) => href.split('#')[0];

function pageLinks(entries: SidebarEntry[], groupLabel?: string): SidebarLink[] {
  const links: SidebarLink[] = [];
  for (const entry of entries) {
    if (entry.type === 'group') {
      links.push(...pageLinks(entry.entries, entry.label));
    } else if (!entry.href.includes('#')) {
      const label = entry.label === 'Introduction' && groupLabel ? groupLabel : entry.label;
      links.push({ ...entry, label });
    }
  }
  return links;
}

export const onRequest = defineRouteMiddleware((context) => {
  const route = context.locals.starlightRoute;
  if (!context.url.pathname.startsWith('/tutorials/')) return;

  const links = pageLinks(route.sidebar);
  const here = links.findIndex((l) => withoutHash(l.href) === context.url.pathname);
  if (here === -1) return;

  route.pagination = {
    prev: links[here - 1],
    next: links[here + 1],
  };
});
