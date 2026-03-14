import { useState } from 'react'
import { ChevronIcon } from './Icons'

/**
 * Collapsible sidebar section with a small header.
 *
 * Keeps sidebar panels tidy and lets the user focus on what matters.
 */
export default function SidebarSection({ title, icon: Icon, defaultOpen = true, children }) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <div className="sidebar-section">
      <button
        className="sidebar-header"
        onClick={() => setOpen(prev => !prev)}
        type="button"
        aria-expanded={open}
      >
        <span className="sidebar-title">
          {Icon && <Icon size={13} />}
          {title}
        </span>
        <span className={`sidebar-toggle ${open ? '' : 'collapsed'}`}>
          <ChevronIcon size={14} />
        </span>
      </button>
      {open && <div className="sidebar-content">{children}</div>}
    </div>
  )
}
