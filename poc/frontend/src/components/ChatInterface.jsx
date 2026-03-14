import { useState, useRef, useEffect } from 'react'
import { sendQuery } from '../api/client'
import {
  HandshakeIcon,
  SendIcon,
  LoaderIcon,
  ZapIcon,
  RadioIcon,
  AlertIcon,
  DomainIcon,
} from './Icons'

/**
 * Chat interface for interacting with the Gossip Handshake Protocol.
 *
 * Displays message history and handles query submission.
 */
export default function ChatInterface({
  modelScale,
  compareMerged,
  routerType,
  onResult,
  sampleQuestions,
}) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = async (e, overrideQuery) => {
    if (e) e.preventDefault()
    const query = overrideQuery || input.trim()
    if (!query || loading) return

    setInput('')
    setError(null)

    const userMessage = { role: 'user', content: query, timestamp: Date.now() }
    setMessages(prev => [...prev, userMessage])
    setLoading(true)

    try {
      const result = await sendQuery({
        query,
        modelScale,
        compareMerged,
        routerType,
      })

      const assistantMessage = {
        role: 'assistant',
        content: result.specialist.response,
        routing: result.routing,
        specialist: result.specialist,
        merged: result.merged,
        totalTime: result.total_time_ms,
        timestamp: Date.now(),
      }

      setMessages(prev => [...prev, assistantMessage])
      onResult(result)
    } catch (err) {
      setError(err.message)
      setMessages(prev => [
        ...prev,
        { role: 'error', content: err.message, timestamp: Date.now() },
      ])
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  const handleSampleClick = (question) => {
    handleSubmit(null, question)
  }

  return (
    <section className="chat-interface" aria-label="Chat">
      <div className="chat-messages" role="log" aria-live="polite">
        {messages.length === 0 && (
          <div className="chat-welcome">
            <div className="welcome-icon"><HandshakeIcon size={36} /></div>
            <h2>Gossip Handshake</h2>
            <p>
              Ask a question about African agriculture. The router selects the
              best specialist adapter automatically.
            </p>
            {sampleQuestions && sampleQuestions.length > 0 && (
              <div className="sample-questions">
                <p className="sample-label">Try a question</p>
                <div className="sample-grid">
                  {sampleQuestions.slice(0, 6).map((sq, i) => (
                    <button
                      key={i}
                      className="sample-btn"
                      onClick={() => handleSampleClick(sq.question)}
                      disabled={loading}
                      type="button"
                    >
                      <span className="sample-icon"><DomainIcon domain={sq.domain} size={15} /></span>
                      <span className="sample-text">{sq.question}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`chat-message chat-message-${msg.role}`}>
            {msg.role === 'user' && (
              <div className="message-bubble user-bubble">
                <p>{msg.content}</p>
              </div>
            )}

            {msg.role === 'assistant' && (
              <div className="message-bubble assistant-bubble">
                {msg.routing && (
                  <div
                    className="routing-badge"
                    style={{ '--domain-color': msg.routing.domain_color }}
                  >
                    <span className="routing-icon"><DomainIcon domain={msg.routing.domain} size={16} /></span>
                    <span className="routing-label">
                      Routed to <strong>{msg.routing.domain_label}</strong>
                    </span>
                    <span className="routing-confidence">
                      {Math.round(msg.routing.confidence * 100)}% confidence
                    </span>
                  </div>
                )}
                <p className="response-text">{msg.content}</p>
                {msg.specialist && (
                  <div className="message-meta">
                    <span className="meta-item"><ZapIcon size={13} /> {msg.specialist.generation_time_ms.toFixed(0)}ms</span>
                    <span className="meta-item"><RadioIcon size={13} /> {msg.routing?.router_type}</span>
                  </div>
                )}

                {msg.merged && (
                  <div className="merged-comparison">
                    <div className="merged-header">
                      <span className="merged-icon"><AlertIcon size={15} /></span>
                      <strong>TIES Merge (for comparison)</strong>
                      <span className="merged-time">
                        {msg.merged.generation_time_ms.toFixed(0)}ms
                      </span>
                    </div>
                    <p className="merged-text">{msg.merged.response}</p>
                  </div>
                )}
              </div>
            )}

            {msg.role === 'error' && (
              <div className="message-bubble error-bubble">
                <p><AlertIcon size={15} style={{verticalAlign: 'middle', marginRight: 4}} /> {msg.content}</p>
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="chat-message chat-message-loading">
            <div className="message-bubble assistant-bubble loading-bubble">
              <div className="loading-dots">
                <span className="dot" />
                <span className="dot" />
                <span className="dot" />
              </div>
              <span className="loading-text">Routing & generating...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <form className="chat-input-form" onSubmit={handleSubmit}>
        <input
          ref={inputRef}
          type="text"
          className="chat-input"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask about crops, livestock, irrigation, soil, or fish farming..."
          disabled={loading}
          maxLength={1000}
          aria-label="Your question"
          autoComplete="off"
        />
        <button
          type="submit"
          className="chat-send-btn"
          disabled={loading || !input.trim()}
          aria-label="Send"
        >
          {loading ? <LoaderIcon size={18} /> : <SendIcon size={18} />}
        </button>
      </form>

      {error && (
        <div className="chat-error" role="alert">
          {error}
        </div>
      )}
    </section>
  )
}
