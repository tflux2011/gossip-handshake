/**
 * Animated protocol diagram showing the Gossip Handshake Protocol.
 *
 * Visualises the 5 community nodes exchanging adapters and
 * the routing step at inference time.
 */
import { useState, useEffect } from 'react'
import { HandshakeIcon, ShuffleIcon, DomainIcon } from './Icons'

const NODES = [
  { id: 'agronomy',     label: 'Agronomy',     color: '#22C55E', x: 50,  y: 15 },
  { id: 'veterinary',   label: 'Veterinary',   color: '#F59E0B', x: 85,  y: 40 },
  { id: 'irrigation',   label: 'Irrigation',   color: '#0EA5E9', x: 72,  y: 78 },
  { id: 'soil_science', label: 'Soil Sci',     color: '#92400E', x: 28,  y: 78 },
  { id: 'aquaculture',  label: 'Aquaculture',  color: '#06B6D4', x: 15,  y: 40 },
]

export default function ProtocolDiagram({ activeDomain, phase }) {
  const [animPhase, setAnimPhase] = useState(0)
  const [pulseNode, setPulseNode] = useState(null)

  // Auto-cycle animation phases when idle
  useEffect(() => {
    if (activeDomain) {
      setPulseNode(activeDomain)
      return
    }

    const interval = setInterval(() => {
      setAnimPhase(prev => (prev + 1) % NODES.length)
    }, 2000)

    return () => clearInterval(interval)
  }, [activeDomain])

  // When auto-cycling, pulse the current node
  useEffect(() => {
    if (!activeDomain) {
      setPulseNode(NODES[animPhase].id)
    }
  }, [animPhase, activeDomain])

  return (
    <div className="protocol-diagram" aria-label="Gossip Handshake Protocol diagram">
      <h3><HandshakeIcon size={16} /> Gossip Network</h3>
      <div className="diagram-container">
        <svg viewBox="0 0 100 100" className="diagram-svg">
          {/* Connection lines between all nodes */}
          {NODES.map((nodeA, i) =>
            NODES.slice(i + 1).map((nodeB) => (
              <line
                key={`${nodeA.id}-${nodeB.id}`}
                x1={nodeA.x}
                y1={nodeA.y}
                x2={nodeB.x}
                y2={nodeB.y}
                className={`diagram-line ${
                  pulseNode === nodeA.id || pulseNode === nodeB.id
                    ? 'diagram-line-active'
                    : ''
                }`}
              />
            ))
          )}

          {/* Adapter transfer animation dots */}
          {pulseNode && NODES.filter(n => n.id !== pulseNode).map((node) => {
            const active = NODES.find(n => n.id === pulseNode)
            if (!active) return null
            return (
              <circle
                key={`dot-${node.id}`}
                className="transfer-dot"
                r="1.2"
                fill={active.color}
              >
                <animateMotion
                  dur="1.5s"
                  repeatCount="indefinite"
                  path={`M${active.x},${active.y} L${node.x},${node.y}`}
                />
              </circle>
            )
          })}
        </svg>

        {/* Node badges */}
        {NODES.map((node) => (
          <div
            key={node.id}
            className={`diagram-node ${
              pulseNode === node.id ? 'diagram-node-active' : ''
            } ${activeDomain === node.id ? 'diagram-node-selected' : ''}`}
            style={{
              left: `${node.x}%`,
              top: `${node.y}%`,
              '--node-color': node.color,
            }}
          >
            <span className="node-icon" style={{ color: node.color }}>
              <DomainIcon domain={node.id} size={18} />
            </span>
            <span className="node-label">{node.label}</span>
          </div>
        ))}

        {/* Central router indicator */}
        <div className="diagram-router">
          <span className="router-icon"><ShuffleIcon size={14} /></span>
          <span className="router-label">Router</span>
        </div>
      </div>

      <div className="protocol-steps">
        <div className={`step ${phase === 'exchange' || !phase ? 'step-active' : ''}`}>
          <span className="step-num">1</span>
          <span className="step-text">Adapter Exchange</span>
        </div>
        <div className="step-arrow">→</div>
        <div className={`step ${phase === 'routing' ? 'step-active' : ''}`}>
          <span className="step-num">2</span>
          <span className="step-text">Route Query</span>
        </div>
        <div className="step-arrow">→</div>
        <div className={`step ${phase === 'generate' ? 'step-active' : ''}`}>
          <span className="step-num">3</span>
          <span className="step-text">Generate</span>
        </div>
      </div>
    </div>
  )
}
