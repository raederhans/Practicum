const bounds = Object.freeze({ west: -170, east: 45, north: 72, south: 16 })
const frame = Object.freeze({ left: 48, right: 912, top: 48, bottom: 492 })

export function projectPoint([longitude, latitude]) {
  const x = frame.left + ((longitude - bounds.west) / (bounds.east - bounds.west)) * (frame.right - frame.left)
  const y = frame.top + ((bounds.north - latitude) / (bounds.north - bounds.south)) * (frame.bottom - frame.top)
  return [Math.round(x * 10) / 10, Math.round(y * 10) / 10]
}
