/* The reader's moves between Scope Tabs.
 *
 * A move touches both stores -- the list's tab and selection, and the pane's
 * first block on an external agent's tab -- and the pane's store already
 * follows the list's, so the moves live here, beside both, rather than in
 * either. The surfaces call these; the stores keep the rules.
 */

import * as details from './detailStore'
import * as list from './store'

import type { TrajectoryEntry } from './types'

/** The blocks an external agent's entry opens on, best first: what the CLI agent said, else the ACP frames. */
export const EXTERNAL_BLOCKS = ['transcript', 'frames']

/* The tab a row leads to: the scope an entry opens, or the scope whose
   report started a turn -- on the turn's first row only, which is the row
   the list marks; the turn's end carries the same scope and leads nowhere. */
export function openTarget(entry: TrajectoryEntry): string | null {
  const opens = list.opensOf(entry)
  if (opens !== null) return opens
  return entry.turn_start ? list.reportedFromOf(entry) : null
}

/* Goes to a sub-agent's tab. A tab opened just now on an external agent's
   run lands on that agent's entry with the pane open on what the agent
   recorded, since that entry is all such a run has to show. */
export function enter(scopeId: string): void {
  if (!list.openScope(scopeId)) return
  const external = list.get().visible.find((e) => e.span_name === 'subagent.external')
  if (!external) return
  details.prefer(external.entry_id, EXTERNAL_BLOCKS)
  list.select(external.entry_id, { source: 'link' })
}

/** Opens the tab a row leads to; false for a row that leads nowhere. */
export function openFromEntry(entry: TrajectoryEntry): boolean {
  const target = openTarget(entry)
  if (target === null) return false
  enter(target)
  return true
}

export function close(scopeId: string): void {
  list.closeScope(scopeId)
}

/** Back to the main session, on the entry that opens the scope; false when the index no longer has it. */
export function locate(scopeId: string): boolean {
  const door = list.get().doors[scopeId]
  if (door === undefined) return false
  list.select(door, { source: 'link' })
  return true
}
