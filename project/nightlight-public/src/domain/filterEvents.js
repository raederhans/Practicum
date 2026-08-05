export function filterEvents(events, { type = 'All', query = '' } = {}) {
  const normalizedQuery = query.trim().toLocaleLowerCase()
  if (type === 'All' && normalizedQuery === '') return events

  return events.filter((event) => {
    const matchesType = type === 'All' || event.type === type
    const haystack = `${event.name} ${event.location} ${event.year}`.toLocaleLowerCase()
    return matchesType && haystack.includes(normalizedQuery)
  })
}
