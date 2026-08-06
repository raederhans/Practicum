export function filterEvents(events, { hazardFamily = 'All', query = '' } = {}) {
  const normalizedQuery = query.trim().toLocaleLowerCase()
  if (hazardFamily === 'All' && normalizedQuery === '') return events

  return events.filter((event) => {
    const matchesFamily = hazardFamily === 'All' || event.hazardFamily === hazardFamily
    const haystack = `${event.name} ${event.location} ${event.year}`.toLocaleLowerCase()
    return matchesFamily && haystack.includes(normalizedQuery)
  })
}
