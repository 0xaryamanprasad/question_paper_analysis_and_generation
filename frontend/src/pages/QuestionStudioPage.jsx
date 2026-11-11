import React, { useState } from 'react';
import { generateQuestion, classifyQuestion } from '../api/client';

export default function QuestionStudioPage() {
  const [topic, setTopic] = useState('Shortest path algorithms (Dijkstra, Bellman-Ford)');
  const [bloomsLevel, setBloomsLevel] = useState('Apply');
  const [marks, setMarks] = useState(10);
  const [difficulty, setDifficulty] = useState('medium');
  const [loading, setLoading] = useState(false);
  const [generatedQuestion, setGeneratedQuestion] = useState(null);

  // Classification Playground State
  const [playSubject, setPlaySubject] = useState('auto');
  const [testQuestionText, setTestQuestionText] = useState(
    "Explain deadlock prevention and avoidance using Banker's algorithm and resource allocation graphs."
  );
  const [classifying, setClassifying] = useState(false);
  const [classificationResult, setClassificationResult] = useState(null);

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const q = await generateQuestion({
        subject: 'Data Structures and Algorithms',
        unit_number: 3,
        unit_name: 'Graphs',
        topic,
        keywords: ['Dijkstra', 'shortest path', 'graph'],
        blooms_level: bloomsLevel,
        marks: parseFloat(marks),
        difficulty,
        question_type: marks >= 8 ? 'long' : (marks <= 2 ? 'short' : 'medium'),
      });
      setGeneratedQuestion(q);
    } catch (e) {
      alert('Generation error: ' + e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleClassify = async () => {
    setClassifying(true);
    try {
      const res = await classifyQuestion(testQuestionText, playSubject);
      setClassificationResult(res);
    } catch (e) {
      alert('Classification error: ' + e.message);
    } finally {
      setClassifying(false);
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">GenAI Question Studio & Classifier</h1>
        <p className="text-slate-600 text-sm mt-1">
          Interactive workbench to author individual Bloom's taxonomy-targeted exam questions and test the NLP multi-tier classification engine.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* GenAI Authoring Card */}
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
          <h2 className="text-base font-bold text-slate-800 flex items-center gap-2">
            <span>✨</span> GenAI Question Generator
          </h2>

          <div>
            <label className="block text-xs font-semibold text-slate-600 mb-1">Target Topic</label>
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg text-xs"
            />
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="block text-xs font-semibold text-slate-600 mb-1">Bloom's Level</label>
              <select
                value={bloomsLevel}
                onChange={(e) => setBloomsLevel(e.target.value)}
                className="w-full px-2 py-1.5 border border-slate-300 rounded-lg text-xs"
              >
                {['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create'].map((l) => (
                  <option key={l} value={l}>{l}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-600 mb-1">Marks</label>
              <input
                type="number"
                value={marks}
                onChange={(e) => setMarks(e.target.value)}
                className="w-full px-2 py-1.5 border border-slate-300 rounded-lg text-xs"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-600 mb-1">Difficulty</label>
              <select
                value={difficulty}
                onChange={(e) => setDifficulty(e.target.value)}
                className="w-full px-2 py-1.5 border border-slate-300 rounded-lg text-xs"
              >
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
              </select>
            </div>
          </div>

          <button
            onClick={handleGenerate}
            disabled={loading}
            className="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-lg text-xs transition"
          >
            {loading ? 'Synthesizing Question...' : 'Generate Targeted Question'}
          </button>

          {generatedQuestion && (
            <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg text-xs space-y-3 mt-4">
              <div className="flex justify-between items-center">
                <span className="font-bold text-slate-800">Generated Candidate:</span>
                <span className="px-2 py-0.5 bg-indigo-100 text-indigo-800 font-semibold rounded text-[11px]">
                  {generatedQuestion.blooms_level} ({generatedQuestion.marks}m)
                </span>
              </div>
              <p className="text-slate-800 leading-relaxed">{generatedQuestion.question_text}</p>
              {generatedQuestion.sub_parts?.length > 0 && (
                <div className="pl-3 border-l-2 border-indigo-300 space-y-1">
                  {generatedQuestion.sub_parts.map((sp, idx) => (
                    <div key={idx}>
                      <span className="font-semibold">({sp.label})</span> {sp.text} <span className="text-slate-500">[{sp.marks}m]</span>
                    </div>
                  ))}
                </div>
              )}
              {generatedQuestion.marking_scheme?.length > 0 && (
                <div className="mt-2 pt-2 border-t border-slate-200">
                  <span className="font-semibold text-slate-700 block mb-1">Marking Scheme:</span>
                  <ul className="list-disc pl-4 space-y-0.5 text-slate-600">
                    {generatedQuestion.marking_scheme.map((r, rIdx) => (
                      <li key={rIdx}>{r}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>

        {/* NLP Multi-tier Classifier Playground */}
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
          <div className="flex justify-between items-center">
            <h2 className="text-base font-bold text-slate-800 flex items-center gap-2">
              <span>🔬</span> NLP Classification Playground
            </h2>
            <span className="text-[11px] font-mono text-slate-500">Auto-detects OS, Math, DSA, DBMS</span>
          </div>

          {/* Subject / Domain Switcher */}
          <div>
            <label className="block text-xs font-semibold text-slate-600 mb-1">Target Subject / Domain</label>
            <div className="flex flex-wrap gap-1.5">
              {[
                { id: 'auto', label: '⚡ Auto-Detect' },
                { id: 'Computer Organization and Architecture', label: '⚙️ Architecture (COA)' },
                { id: 'Operating Systems', label: '💻 Operating Systems' },
                { id: 'Mathematics - II', label: '📐 Mathematics-II' },
                { id: 'Data Structures and Algorithms', label: '🌳 DSA' },
                { id: 'Database Management Systems', label: '🗄️ DBMS' },
              ].map((sub) => (
                <button
                  key={sub.id}
                  type="button"
                  onClick={() => setPlaySubject(sub.id)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-medium transition ${
                    playSubject === sub.id
                      ? 'bg-indigo-600 text-white shadow-sm'
                      : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                  }`}
                >
                  {sub.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <div className="flex justify-between items-center mb-1">
              <label className="block text-xs font-semibold text-slate-600">Exam Question Prompt</label>
              <div className="flex flex-wrap gap-1">
                <span className="text-[10px] text-slate-400 font-medium">Try:</span>
                <button
                  type="button"
                  onClick={() => {
                    setPlaySubject('Computer Organization and Architecture');
                    setTestQuestionText("What is pipelining in computer architecture? Explain pipeline hazards.");
                  }}
                  className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 hover:bg-slate-200 text-slate-700 font-mono"
                >
                  COA (Pipelining)
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setPlaySubject('Computer Organization and Architecture');
                    setTestQuestionText("What is the meaning of a gigabyte in computer storage hierarchy?");
                  }}
                  className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 hover:bg-slate-200 text-slate-700 font-mono"
                >
                  COA (Gigabyte)
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setPlaySubject('Operating Systems');
                    setTestQuestionText("Explain deadlock prevention and avoidance using Banker's algorithm and resource allocation graphs.");
                  }}
                  className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 hover:bg-slate-200 text-slate-700 font-mono"
                >
                  OS (Deadlock)
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setPlaySubject('Mathematics - II');
                    setTestQuestionText("Find the particular integral of the differential equation y'' + y = cos x.");
                  }}
                  className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 hover:bg-slate-200 text-slate-700 font-mono"
                >
                  Math (ODE)
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setPlaySubject('Data Structures and Algorithms');
                    setTestQuestionText("Explain the working of AVL Tree rotations with an example of Left-Right (LR) imbalance.");
                  }}
                  className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 hover:bg-slate-200 text-slate-700 font-mono"
                >
                  DSA (Trees)
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setPlaySubject('Database Management Systems');
                    setTestQuestionText("Define BCNF (Boyce-Codd Normal Form) and explain lossless-join decomposition with functional dependencies.");
                  }}
                  className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 hover:bg-slate-200 text-slate-700 font-mono"
                >
                  DBMS (BCNF)
                </button>
              </div>
            </div>
            <textarea
              rows={3}
              value={testQuestionText}
              onChange={(e) => setTestQuestionText(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg text-xs focus:ring-2 focus:ring-indigo-500 font-mono"
              placeholder="Paste exam question prompt..."
            />
          </div>

          <button
            onClick={handleClassify}
            disabled={classifying}
            className="w-full py-2 bg-slate-800 hover:bg-slate-900 text-white font-medium rounded-lg text-xs transition"
          >
            {classifying ? 'Analyzing Linguistic Patterns...' : 'Run Topic & Bloom’s Classification'}
          </button>

          {classificationResult && (
            <div className="space-y-3 mt-4 text-xs">
              <div className="p-3 bg-indigo-50 border border-indigo-200 rounded-lg space-y-1">
                <div className="flex justify-between items-center">
                  <span className="font-bold text-indigo-900">Topic Classification:</span>
                  <span className="px-2 py-0.5 bg-indigo-200 text-indigo-900 rounded font-bold text-[10px]">
                    {classificationResult.topic_classification?.course_code || 'CS202'}
                  </span>
                </div>
                <div className="text-slate-800">
                  <span className="text-slate-500">Domain:</span> <b>{classificationResult.topic_classification?.subject || 'Operating Systems'}</b>
                </div>
                <div className="text-slate-800">
                  <span className="text-slate-500">Unit:</span> <b>{classificationResult.topic_classification?.unit_name}</b> (Unit {classificationResult.topic_classification?.unit_number})
                </div>
                <div className="text-slate-800">
                  <span className="text-slate-500">Topic:</span> <b>{classificationResult.topic_classification?.topic}</b>
                </div>
                <div className="text-indigo-700 font-semibold text-[11px] pt-1">
                  Confidence: {Math.round((classificationResult.topic_classification?.confidence || 0) * 100)}% | Method: {classificationResult.topic_classification?.classification_method}
                </div>
                <div className="text-slate-500 text-[11px]">
                  Matched Keywords: {classificationResult.topic_classification?.matched_keywords?.join(', ') || 'None'}
                </div>
              </div>

              <div className="p-3 bg-purple-50 border border-purple-200 rounded-lg space-y-1">
                <div className="font-bold text-purple-900">Bloom’s Revised Taxonomy:</div>
                <div className="text-slate-800 font-medium">
                  {classificationResult.blooms_classification?.blooms_level} ({classificationResult.blooms_classification?.level_code})
                </div>
                <div className="text-purple-700 font-semibold text-[11px]">
                  Confidence: {Math.round((classificationResult.blooms_classification?.confidence || 0) * 100)}%
                </div>
                <div className="text-slate-600 text-[11px]">
                  Rationale: {classificationResult.blooms_classification?.rationale}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
