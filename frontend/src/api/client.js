/**
 * API Client for interacting with the FastAPI backend.
 */

const BASE_URL = '/api/v1';

export async function checkHealth() {
  const res = await fetch('/health');
  return res.json();
}

export async function uploadPaper(file, subject, year = '2023', semester = 'Fall') {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('subject', subject);
  if (year) formData.append('year', year);
  if (semester) formData.append('semester', semester);

  const res = await fetch(`${BASE_URL}/papers/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(err.detail || 'Upload failed');
  }
  return res.json();
}

export async function listPapers(subject) {
  const url = subject ? `${BASE_URL}/papers?subject=${encodeURIComponent(subject)}` : `${BASE_URL}/papers`;
  const res = await fetch(url);
  return res.json();
}

export async function getPaperQuestions(paperId) {
  const res = await fetch(`${BASE_URL}/papers/${paperId}/questions`);
  return res.json();
}

export async function getHistoricalAnalysis(subject) {
  const res = await fetch(`${BASE_URL}/analysis/historical?subject=${encodeURIComponent(subject)}`);
  return res.json();
}

export async function getWeightageAnalysis(subject) {
  const res = await fetch(`${BASE_URL}/analysis/weightage?subject=${encodeURIComponent(subject)}`);
  return res.json();
}

export async function classifyQuestion(questionText, subject = null) {
  const payload = { question_text: questionText };
  if (subject) payload.subject = subject;
  const res = await fetch(`${BASE_URL}/analysis/classify-question`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return res.json();
}

export async function generateQuestion(payload) {
  const res = await fetch(`${BASE_URL}/generation/generate-question`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return res.json();
}

export async function generatePaper(blueprint, syllabus = null) {
  const res = await fetch(`${BASE_URL}/generation/generate-paper`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ blueprint, syllabus }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Paper generation failed' }));
    throw new Error(err.detail || 'Paper generation failed');
  }
  return res.json();
}

export async function validatePaper(paper, blueprint, syllabus = null) {
  const res = await fetch(`${BASE_URL}/generation/validate-paper`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paper, blueprint, syllabus }),
  });
  return res.json();
}

export async function downloadPaperPDF(paper) {
  const res = await fetch(`${BASE_URL}/generation/render-pdf`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(paper),
  });
  if (!res.ok) throw new Error('PDF export failed');
  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${(paper.subject || 'exam').replace(/\s+/g, '_').toLowerCase()}_paper.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
}

export async function getDefaultSyllabus() {
  const res = await fetch(`${BASE_URL}/syllabus/default`);
  return res.json();
}
