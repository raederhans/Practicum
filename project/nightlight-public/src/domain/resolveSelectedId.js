export function resolveSelectedId(events, selectedId) {
  if (events.some((event) => event.id === selectedId)) return selectedId
  return events[0]?.id ?? null
}
