export const routes = [
  { path: '/', name: 'overview', meta: { pageTitle: 'Overview' }, component: () => import('../views/OverviewView.vue') },
  { path: '/atlas', name: 'atlas', meta: { pageTitle: 'Study Atlas' }, component: () => import('../views/AtlasView.vue') },
  { path: '/findings', name: 'findings', meta: { pageTitle: 'Findings' }, component: () => import('../views/FindingsView.vue') },
  { path: '/methods', name: 'methods', meta: { pageTitle: 'Methods' }, component: () => import('../views/MethodsView.vue') },
  { path: '/credits', name: 'credits', meta: { pageTitle: 'Credits / Policy' }, component: () => import('../views/CreditsView.vue') },
]
