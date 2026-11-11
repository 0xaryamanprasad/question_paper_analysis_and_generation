import React, { useState, useEffect } from 'react';
import { getHistoricalAnalysis, getWeightageAnalysis } from '../api/client';

export default function DashboardPage() {
  const [subject, setSubject] = useState('Data Structures and Algorithms');
  const [histData, setHistData] = useState(null);
  const [weightData, setWeightData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchAnalytics();
  }, [subject]);

  const fetchAnalytics = async () => {
    setLoading(true);
    try {
      const [hist, weight] = await Promise.all([
        getHistoricalAnalysis(subject),
        getWeightageAnalysis(subject),
      ]);
      setHistData(hist);
      setWeightData(weight);
    } catch (e) {
      console.error('Error fetching analytics:', e);
    } finally {
      setLoading(false);
    }
  };

  const getTrendBadge = (trend) => {
    switch (trend) {
      case 'increasing':
        return <span className="px-2 py-0.5 text-xs font-semibold rounded bg-emerald-100 text-emerald-800">↗ Increasing</span>;
      case 'decreasing':
        return <span className="px-2 py-0.5 text-xs font-semibold rounded bg-rose-100 text-rose-800">↘ Decreasing</span>;
      case 'stable':
        return <span className="px-2 py-0.5 text-xs font-semibold rounded bg-blue-100 text-blue-800">→ Stable</span>;
      default:
        return <span className="px-2 py-0.5 text-xs font-semibold rounded bg-slate-100 text-slate-700">~ Intermittent</span>;
    }
  };

  return (
    <div className="space-y-8">
      {/* Header & Subject Filter */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Historical Analytics Dashboard</h1>
          <p className="text-slate-600 text-sm mt-1">
            Exam pattern mining, topic recurrence rates, unit weightages, and Bloom's taxonomy distribution.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
            className="px-3 py-1.5 border border-slate-300 rounded-lg text-sm bg-white shadow-sm focus:ring-2 focus:ring-indigo-500"
            placeholder="Search Subject"
          />
          <button
            onClick={fetchAnalytics}
            className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-sm font-medium transition"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Papers Analyzed</span>
          <div className="text-2xl font-bold text-slate-900 mt-2">{histData ? histData.total_papers_analyzed : '-'}</div>
          <p className="text-xs text-slate-400 mt-1">Years: {histData?.years_covered?.join(', ') || 'N/A'}</p>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Total Questions</span>
          <div className="text-2xl font-bold text-slate-900 mt-2">{histData ? histData.total_questions : '-'}</div>
          <p className="text-xs text-slate-400 mt-1">Extracted & classified</p>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Repetition Rate</span>
          <div className="text-2xl font-bold text-indigo-600 mt-2">{histData ? `${histData.overall_repetition_rate}%` : '-'}</div>
          <p className="text-xs text-slate-400 mt-1">Semantic duplicates across years</p>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Avg Exam Marks</span>
          <div className="text-2xl font-bold text-slate-900 mt-2">{weightData ? `${weightData.average_paper_total_marks}m` : '-'}</div>
          <p className="text-xs text-slate-400 mt-1">Total marks: {weightData ? `${weightData.total_marks_all_papers}m` : '-'}</p>
        </div>
      </div>

      {/* Charts & Weightages Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Unit-wise Weightage Distribution */}
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
          <h2 className="text-base font-bold text-slate-800">Unit-wise Marks Weightage Distribution</h2>
          <div className="space-y-3">
            {weightData?.unit_weightages?.map((u) => (
              <div key={u.unit_number} className="space-y-1">
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-slate-800">{u.unit_name}</span>
                  <span className="text-indigo-600 font-bold">{u.weightage_percentage}% ({u.total_marks}m)</span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden flex">
                  <div
                    className="bg-indigo-600 h-2 rounded-full transition-all"
                    style={{ width: `${Math.min(u.weightage_percentage, 100)}%` }}
                  />
                </div>
                {u.deviation_percentage !== null && (
                  <div className="flex justify-between text-[11px] text-slate-400">
                    <span>Expected: {u.expected_percentage}%</span>
                    <span className={u.deviation_percentage > 0 ? 'text-amber-600 font-medium' : 'text-slate-500'}>
                      Dev: {u.deviation_percentage > 0 ? `+${u.deviation_percentage}%` : `${u.deviation_percentage}%`}
                    </span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Bloom's Cognitive Distribution */}
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
          <h2 className="text-base font-bold text-slate-800">Bloom's Taxonomy Cognitive Distribution</h2>
          <div className="space-y-2.5">
            {['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create'].map((level) => {
              const pct = histData?.blooms_percentage?.[level] || 0.0;
              const count = histData?.blooms_distribution?.[level] || 0;
              const levelColors = {
                Remember: 'bg-sky-500',
                Understand: 'bg-emerald-500',
                Apply: 'bg-indigo-500',
                Analyze: 'bg-amber-500',
                Evaluate: 'bg-orange-500',
                Create: 'bg-purple-500',
              };
              return (
                <div key={level} className="space-y-1">
                  <div className="flex justify-between text-xs font-medium">
                    <span className="text-slate-700">{level}</span>
                    <span className="text-slate-900 font-semibold">{pct}% ({count} Qs)</span>
                  </div>
                  <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                    <div
                      className={`${levelColors[level] || 'bg-indigo-500'} h-2 rounded-full transition-all`}
                      style={{ width: `${Math.min(pct, 100)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Topic Trends Table */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="p-6 border-b border-slate-100 flex justify-between items-center">
          <div>
            <h2 className="text-base font-bold text-slate-800">Longitudinal Topic Recurrence & Trends</h2>
            <p className="text-xs text-slate-500 mt-0.5">Year-over-year appearance frequency and trend trajectories</p>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-700">
            <thead className="bg-slate-50 text-slate-500 uppercase font-semibold border-b border-slate-200">
              <tr>
                <th className="py-3 px-4">Topic</th>
                <th className="py-3 px-4">Unit</th>
                <th className="py-3 px-4 text-center">Occurrences</th>
                <th className="py-3 px-4 text-center">Trend</th>
                <th className="py-3 px-4 text-right">Avg Marks</th>
                <th className="py-3 px-4 text-right">Importance</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {histData?.topic_trends?.map((t, idx) => (
                <tr key={idx} className="hover:bg-slate-50">
                  <td className="py-3 px-4 font-medium text-slate-900">{t.topic}</td>
                  <td className="py-3 px-4 text-slate-500">Unit {t.unit_number}</td>
                  <td className="py-3 px-4 text-center font-bold text-indigo-600">{t.total_occurrences}</td>
                  <td className="py-3 px-4 text-center">{getTrendBadge(t.trend_direction)}</td>
                  <td className="py-3 px-4 text-right">{t.average_marks}m</td>
                  <td className="py-3 px-4 text-right font-semibold text-slate-800">{t.importance_score}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Recurrence Clusters */}
      {histData?.recurring_questions?.length > 0 && (
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
          <h2 className="text-base font-bold text-slate-800">
            Multi-Year Recurrence Clusters ({histData.recurring_questions.length})
          </h2>
          <div className="space-y-3">
            {histData.recurring_questions.map((g, idx) => (
              <div key={idx} className="p-4 bg-amber-50/50 border border-amber-200 rounded-lg text-xs space-y-1.5">
                <div className="flex justify-between items-center text-amber-900 font-semibold">
                  <span>Cluster #{g.cluster_id}: {g.topic || 'General'}</span>
                  <span className="bg-amber-100 px-2 py-0.5 rounded text-[11px]">
                    Years: {g.years_appeared.join(', ')} ({g.occurrence_count}x Repeated)
                  </span>
                </div>
                <p className="text-slate-700 italic">"{g.canonical_question}"</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
