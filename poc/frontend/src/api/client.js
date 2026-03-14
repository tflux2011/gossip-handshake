/**
 * API client for the Gossip Handshake backend.
 *
 * All requests go through this module for centralised error handling.
 */

const BASE_URL = '/api';

async function request(path, options = {}) {
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }

  return res.json();
}

export async function getHealth() {
  return request('/health');
}

export async function getDomains() {
  return request('/domains');
}

export async function getModelStatus() {
  return request('/model-status');
}

export async function getSampleQuestions() {
  return request('/sample-questions');
}

export async function sendQuery({
  query,
  modelScale = '0.5B',
  compareMerged = false,
  routerType = 'keyword',
  temperature = 0.3,
}) {
  return request('/query', {
    method: 'POST',
    body: JSON.stringify({
      query,
      model_scale: modelScale,
      compare_merged: compareMerged,
      router_type: routerType,
      temperature,
    }),
  });
}

export async function switchModel(modelScale) {
  return request('/switch-model', {
    method: 'POST',
    body: JSON.stringify({ model_scale: modelScale }),
  });
}
