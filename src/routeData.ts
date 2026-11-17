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
      // Chapter name for a chapter's Introduction link, without the sidebar's "2. " numbering.
      const label = (entry.label === 'Introduction' && groupLabel ? groupLabel : entry.label).replace(/^\d+\.\s+/, '');
      links.push({ ...entry, label });
    }
  }
  return links;
}

export const onRequest = defineRouteMiddleware((context) => {
  const route = context.locals.starlightRoute;
  if (!context.url.pathname.startsWith('/tutorials/')) return;
  // The tutorials landing page lists the series; it has no sidebar and no previous or next page.
  if (context.url.pathname === '/tutorials/') {
    route.hasSidebar = false;
    route.pagination = { prev: undefined, next: undefined };
    return;
  }

  // Each tutorial page shows only its own series in the sidebar, next to the "Tutorials" link.
  // Pagination is then computed from that series alone, so Previous/Next never cross series.
  const series = context.url.pathname.split('/')[2];
  const prefix = `/tutorials/${series}/`;
  const inSeries = (entry: SidebarEntry): boolean =>
    entry.type === 'group' ? entry.entries.some(inSeries) : entry.href.startsWith(prefix);
  route.sidebar = route.sidebar.filter((entry) => entry.type === 'link' || inSeries(entry));

  const links = pageLinks(route.sidebar);
  const here = links.findIndex((l) => withoutHash(l.href) === context.url.pathname);
  if (here === -1) return;

  route.pagination = {
    prev: links[here - 1],
    next: links[here + 1],
  };
});
