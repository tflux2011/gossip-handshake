import { useState, useEffect } from 'react'
import { getModelStatus, getSampleQuestions } from './api/client'
import ChatInterface from './components/ChatInterface'
import RoutingPanel from './components/RoutingPanel'
import ProtocolDiagram from './components/ProtocolDiagram'
import ControlBar from './components/ControlBar'
import SidebarSection from './components/SidebarSection'
import { HandshakeIcon, ShuffleIcon } from './components/Icons'

export default function App() {
  const [modelScale, setModelScale] = useState('0.5B')
  const [routerType, setRouterType] = useState('keyword')
  const [compareMerged, setCompareMerged] = useState(false)
  const [modelStatus, setModelStatus] = useState(null)
  const [sampleQuestions, setSampleQuestions] = useState([])
  const [lastRouting, setLastRouting] = useState(null)
  const [activeDomain, setActiveDomain] = useState(null)
  const [phase, setPhase] = useState(null)
  const [backendReady, setBackendReady] = useState(false)

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const status = await getModelStatus()
        setModelStatus(status)
        setModelScale(status.model_scale)
        setBackendReady(status.loaded)
      } catch {
        setBackendReady(false)
        setTimeout(checkBackend, 3000)
      }
    }

    const loadSamples = async () => {
      try {
        const questions = await getSampleQuestions()
        setSampleQuestions(questions)
      } catch {
        /* Samples are optional */
      }
    }

    checkBackend()
    loadSamples()
  }, [])

  const handleResult = (result) => {
    setLastRouting(result.routing)
    setActiveDomain(result.routing.domain)
    setPhase('generate')

    setTimeout(() => setPhase(null), 3000)
  }

  if (!backendReady) {
    return (
      <div className="app-loading">
        <div className="loading-spinner" />
        <h2>Loading Gossip Handshake Protocol</h2>
        <p>Starting model and loading adapters...</p>
        <p className="loading-hint">
          This may take 30–60 seconds on first launch.
        </p>
      </div>
    )
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-brand">
          <span className="header-icon"><HandshakeIcon size={28} /></span>
          <div>
            <h1>Gossip Handshake</h1>
            <p className="header-subtitle">
              Routing-Based LoRA Adapter Selection for Decentralised Knowledge Sharing
            </p>
          </div>
        </div>
        <ControlBar
          modelScale={modelScale}
          setModelScale={setModelScale}
          routerType={routerType}
          setRouterType={setRouterType}
          compareMerged={compareMerged}
          setCompareMerged={setCompareMerged}
          modelStatus={modelStatus}
        />
      </header>

      <main className="app-main">
        <div className="main-left">
          <SidebarSection title="Network" icon={HandshakeIcon}>
            <ProtocolDiagram activeDomain={activeDomain} phase={phase} />
          </SidebarSection>
          <SidebarSection title="Router" icon={ShuffleIcon}>
            <RoutingPanel routingResult={lastRouting} />
          </SidebarSection>
        </div>
        <div className="main-right">
          <ChatInterface
            modelScale={modelScale}
            compareMerged={compareMerged}
            routerType={routerType}
            onResult={handleResult}
            sampleQuestions={sampleQuestions}
          />
        </div>
      </main>
    </div>
  )
}
