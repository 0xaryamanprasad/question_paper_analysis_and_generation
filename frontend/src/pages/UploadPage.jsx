import React, { useState, useEffect } from 'react';
import { uploadPaper, listPapers, getPaperQuestions } from '../api/client';

export default function UploadPage() {
  const [file, setFile] = useState(null);
  const [subject, setSubject] = useState('Data Structures and Algorithms');
  const [year, setYear] = useState('2023');
  const [semester, setSemester] = useState('Fall');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [questions, setQuestions] = useState([]);
  const [error, setError] = useState(null);
  const [papersList, setPapersList] = useState([]);

  useEffect(() => {
    loadPapers();
  }, []);

  const loadPapers = async () => {
    try {
      const data = await listPapers();
      setPapersList(data.papers || []);
    } catch (e) {
      console.error(e);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a PDF file');
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    setQuestions([]);

    try {
      const res = await uploadPaper(file, subject, year, semester);
      setResult(res);
      // Fetch extracted questions
      const qData = await getPaperQuestions(res.id);
      setQuestions(qData || []);
      loadPapers();
    } catch (err) {
      setError(err.message || 'Error uploading paper');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Upload & OCR Extraction</h1>
        <p className="text-slate-600 mt-1">
          Upload prior-year examination papers in PDF format. Supports both digitally generated text and scanned images via PaddleOCR PP-OCRv5.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Upload Form */}
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
          <h2 className="text-lg font-semibold text-slate-800">New Paper Upload</h2>
          <form onSubmit={handleUpload} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">PDF File</label>
              <input
                type="file"
                accept=".pdf"
                onChange={(e) => setFile(e.target.files[0])}
                className="block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Subject Title</label>
              <input
                type="text"
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                required
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Exam Year</label>
                <input
                  type="text"
                  value={year}
                  onChange={(e) => setYear(e.target.value)}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Semester</label>
                <input
                  type="text"
                  value={semester}
                  onChange={(e) => setSemester(e.target.value)}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500"
                />
              </div>
            </div>

            {error && (
              <div className="p-3 bg-red-50 text-red-700 text-xs rounded-lg border border-red-200">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 px-4 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-lg text-sm transition shadow-sm disabled:opacity-50"
            >
              {loading ? 'Processing OCR & Extracting...' : 'Upload & Process Paper'}
            </button>
          </form>
        </div>

        {/* Extraction Results */}
        <div className="lg:col-span-2 space-y-6">
          {result ? (
            <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-6">
              <div className="flex items-center justify-between border-b pb-4">
                <div>
                  <h3 className="font-semibold text-lg text-slate-900">{result.subject} ({result.year})</h3>
                  <p className="text-xs text-slate-500 mt-0.5">{result.original_filename}</p>
                </div>
                <div className="flex gap-2">
                  <span className={`px-2.5 py-1 text-xs font-semibold rounded-full ${
                    result.is_scanned ? 'bg-amber-100 text-amber-800' : 'bg-emerald-100 text-emerald-800'
                  }`}>
                    {result.is_scanned ? 'Scanned (PP-OCRv5)' : 'Digital Text'}
                  </span>
                  <span className="px-2.5 py-1 bg-blue-100 text-blue-800 text-xs font-semibold rounded-full">
                    {result.total_pages} Pages
                  </span>
                </div>
              </div>

              <div>
                <h4 className="text-sm font-semibold text-slate-800 mb-3">
                  Extracted Questions ({questions.length})
                </h4>
                {questions.length === 0 ? (
                  <p className="text-xs text-slate-500 italic">No questions segmented yet.</p>
                ) : (
                  <div className="space-y-3 max-h-96 overflow-y-auto pr-2">
                    {questions.map((q, idx) => (
                      <div key={q.id || idx} className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-xs">
                        <div className="flex justify-between items-start mb-1 font-semibold text-slate-800">
                          <span>{q.question_number ? `Q${q.question_number}.` : `Question ${idx + 1}`}</span>
                          <span className="text-indigo-600">[{q.marks || 0} Marks]</span>
                        </div>
                        <p className="text-slate-700">{q.cleaned_text || q.raw_text}</p>
                        {q.sub_parts && q.sub_parts.length > 0 && (
                          <div className="mt-2 pl-3 border-l-2 border-indigo-200 space-y-1">
                            {q.sub_parts.map((sp, sIdx) => (
                              <div key={sIdx} className="text-slate-600">
                                <span className="font-medium">({sp.label})</span> {sp.text} <span className="text-slate-400">[{sp.marks}m]</span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="bg-white p-12 rounded-xl border border-dashed border-slate-300 text-center">
              <svg className="w-12 h-12 text-slate-400 mx-auto mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <h3 className="text-sm font-semibold text-slate-700">No Paper Uploaded Yet</h3>
              <p className="text-xs text-slate-500 mt-1">Upload a past exam paper PDF on the left to view real-time OCR extraction results.</p>
            </div>
          )}

          {/* Past Uploads Library */}
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
            <h3 className="font-semibold text-base text-slate-800 mb-3">Paper Repository ({papersList.length})</h3>
            {papersList.length === 0 ? (
              <p className="text-xs text-slate-500">Repository is currently empty.</p>
            ) : (
              <div className="divide-y divide-slate-100">
                {papersList.map((p) => (
                  <div key={p.id} className="py-2.5 flex items-center justify-between text-xs">
                    <div>
                      <span className="font-medium text-slate-800">{p.subject}</span>
                      <span className="text-slate-500 ml-2">({p.year} {p.semester || ''})</span>
                    </div>
                    <span className="text-slate-400">{new Date(p.created_at).toLocaleDateString()}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
