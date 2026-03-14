/**
 * Routing visualization panel.
 *
 * Shows the keyword match scores for each domain and highlights
 * the winning route with animation.
 */
import {
  ShuffleIcon,
  CheckIcon,
  DomainIcon,
  WheatIcon,
  StethoscopeIcon,
  DropletIcon,
  LayersIcon,
  FishIcon,
} from './Icons'

export default function RoutingPanel({ routingResult }) {
  if (!routingResult) {
    return (
      <aside className="routing-panel routing-panel-empty" aria-label="Routing">
        <h3><ShuffleIcon size={16} /> Router</h3>
        <p className="routing-hint">
          Send a query to see how the router classifies it across 5 domains.
        </p>
        <div className="routing-domains-preview">
          {[
            { key: 'agronomy', label: 'Agronomy', color: '#22C55E' },
            { key: 'veterinary', label: 'Veterinary', color: '#F59E0B' },
            { key: 'irrigation', label: 'Irrigation', color: '#0EA5E9' },
            { key: 'soil_science', label: 'Soil Science', color: '#92400E' },
            { key: 'aquaculture', label: 'Aquaculture', color: '#06B6D4' },
          ].map((d) => (
            <div key={d.label} className="domain-preview-chip" style={{ color: d.color }}>
              <DomainIcon domain={d.key} size={16} />
              <span>{d.label}</span>
            </div>
          ))}
        </div>
      </aside>
    )
  }

  const { domain, scores, keyword_matches, confidence, router_type, domain_label, domain_color } = routingResult
  const maxScore = Math.max(...Object.values(scores), 1)

  const domainMeta = {
    agronomy:     { label: 'Agronomy',     color: '#22C55E' },
    veterinary:   { label: 'Veterinary',   color: '#F59E0B' },
    irrigation:   { label: 'Irrigation',   color: '#0EA5E9' },
    soil_science: { label: 'Soil Science', color: '#92400E' },
    aquaculture:  { label: 'Aquaculture',  color: '#06B6D4' },
  }

  return (
    <aside className="routing-panel" aria-label="Routing result">
      <h3><ShuffleIcon size={16} /> Router Result</h3>

      <div className="routing-winner" style={{ '--winner-color': domain_color }}>
        <div className="winner-icon-wrap">
          <DomainIcon domain={domain} size={28} />
        </div>
        <div className="winner-info">
          <span className="winner-label">{domain_label}</span>
          <span className="winner-confidence">
            {Math.round(confidence * 100)}% confidence
          </span>
        </div>
        <span className="winner-badge"><CheckIcon size={12} /> Selected</span>
      </div>

      <div className="routing-scores">
        <h4>Domain Scores ({router_type})</h4>
        {Object.entries(scores)
          .sort(([, a], [, b]) => b - a)
          .map(([key, score]) => {
            const meta = domainMeta[key] || { label: key, color: '#888' }
            const isWinner = key === domain
            const barWidth = maxScore > 0 ? (score / maxScore) * 100 : 0

            return (
              <div
                key={key}
                className={`score-row ${isWinner ? 'score-row-winner' : ''}`}
              >
                <div className="score-label" style={{ color: meta.color }}>
                  <DomainIcon domain={key} size={15} />
                  <span className="score-name">{meta.label}</span>
                </div>
                <div className="score-bar-track">
                  <div
                    className="score-bar-fill"
                    style={{
                      width: `${barWidth}%`,
                      backgroundColor: meta.color,
                    }}
                  />
                </div>
                <span className="score-value">{score}</span>
              </div>
            )
          })}
      </div>

      {keyword_matches && keyword_matches[domain] && keyword_matches[domain].length > 0 && (
        <div className="keyword-matches">
          <h4>Matched Keywords</h4>
          <div className="keyword-chips">
            {keyword_matches[domain].map((kw, i) => (
              <span key={i} className="keyword-chip" style={{ '--kw-color': domain_color }}>
                {kw}
              </span>
            ))}
          </div>
        </div>
      )}
    </aside>
  )
}
