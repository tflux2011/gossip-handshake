/**
 * Model controls header bar.
 *
 * Provides model scale toggle, router type selector,
 * and merge comparison toggle.
 */
import { useState } from 'react'
import { switchModel } from '../api/client'
import { KeyIcon, TargetIcon, LoaderIcon } from './Icons'

export default function ControlBar({
  modelScale,
  setModelScale,
  routerType,
  setRouterType,
  compareMerged,
  setCompareMerged,
  modelStatus,
}) {
  const [switching, setSwitching] = useState(false)

  const handleScaleToggle = async (scale) => {
    if (scale === modelScale || switching) return
    setSwitching(true)
    try {
      await switchModel(scale)
      setModelScale(scale)
    } catch (err) {
      console.error('Failed to switch model:', err)
    } finally {
      setSwitching(false)
    }
  }

  return (
    <div className="control-bar">
      <div className="control-group">
        <label className="control-label">Model</label>
        <div className="toggle-group">
          {['0.5B', '1.5B'].map((scale) => (
            <button
              key={scale}
              className={`toggle-btn ${modelScale === scale ? 'toggle-active' : ''}`}
              onClick={() => handleScaleToggle(scale)}
              disabled={switching}
              type="button"
            >
              {switching && modelScale !== scale ? <LoaderIcon size={13} /> : null} {scale}
            </button>
          ))}
        </div>
      </div>

      <div className="control-group">
        <label className="control-label">Router</label>
        <div className="toggle-group">
          {[
            { value: 'keyword', label: 'Keyword', icon: KeyIcon },
            { value: 'cosine', label: 'Cosine', icon: TargetIcon },
          ].map((opt) => (
            <button
              key={opt.value}
              className={`toggle-btn ${routerType === opt.value ? 'toggle-active' : ''}`}
              onClick={() => setRouterType(opt.value)}
              type="button"
            >
              <opt.icon size={13} /> {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="control-group">
        <label className="control-label">
          <input
            type="checkbox"
            checked={compareMerged}
            onChange={(e) => setCompareMerged(e.target.checked)}
            className="control-checkbox"
          />
          TIES Merge
        </label>
      </div>

      {modelStatus && (
        <div className="status-indicator">
          <span className={`status-dot ${modelStatus.loaded ? 'status-online' : 'status-offline'}`} />
          <span className="status-text">
            {modelStatus.loaded
              ? `${modelStatus.domains_loaded?.length || 0} adapters loaded`
              : 'Loading...'}
          </span>
        </div>
      )}
    </div>
  )
}
