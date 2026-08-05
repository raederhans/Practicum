export const routes = [
  { path: '/', name: 'overview', component: () => import('../views/OverviewView.vue') },
  { path: '/atlas', name: 'atlas', component: () => import('../views/AtlasView.vue') },
  { path: '/findings', name: 'findings', component: () => import('../views/FindingsView.vue') },
  { path: '/methods', name: 'methods', component: () => import('../views/MethodsView.vue') },
  { path: '/credits', name: 'credits', component: () => import('../views/CreditsView.vue') },
]
