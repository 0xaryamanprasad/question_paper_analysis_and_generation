import React, { useState, useEffect } from 'react';
import UploadPage from './pages/UploadPage';
import DashboardPage from './pages/DashboardPage';
import GeneratePage from './pages/GeneratePage';
import QuestionStudioPage from './pages/QuestionStudioPage';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [isDark, setIsDark] = useState(() => {
    const saved = localStorage.getItem('theme');
    if (saved) return saved === 'dark';
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [isDark]);

  const tabs = [
    {
      id: 'dashboard',
      label: 'Intelligence & Analytics',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      ),
      component: <DashboardPage />,
    },
    {
      id: 'upload',
      label: 'Ingestion & OCR Extraction',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
        </svg>
      ),
      component: <UploadPage />,
    },
    {
      id: 'generate',
      label: 'Paper Studio & Generator',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
        </svg>
      ),
      component: <GeneratePage />,
    },
    {
      id: 'studio',
      label: 'Linguistic NLP Lab',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
      ),
      component: <QuestionStudioPage />,
    },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-white dark:bg-[#020617] text-slate-900 dark:text-slate-100 transition-colors duration-200 relative overflow-hidden">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-30 backdrop-blur-xl bg-[#0b0f19]/80 border-b border-slate-800/80 transition-colors">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            {/* PaperMind Brand Emblem */}
            <img 
              src="/icon.png" 
              alt="PaperMind" 
              className="w-9 h-9 object-contain rounded-lg p-0.5"
              onError={(e) => { e.target.src = '/assets/papermind_icon.png'; }}
            />
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-white text-lg tracking-tight font-sans">
                PaperMind
              </span>
              <span className="px-2 py-0.5 rounded-full text-[11px] font-mono text-slate-300 bg-slate-900 border border-slate-800 inline-flex items-center gap-1">
                v2.4
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3 sm:gap-6 text-xs font-medium">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`transition cursor-pointer ${activeTab === 'dashboard' ? 'text-white font-semibold' : 'text-slate-400 hover:text-slate-200'}`}
            >
              Docs
            </button>
            <button
              onClick={() => setActiveTab('studio')}
              className={`transition cursor-pointer ${activeTab === 'studio' ? 'text-white font-semibold' : 'text-slate-400 hover:text-slate-200'}`}
            >
              Play
            </button>
            <button
              onClick={() => setActiveTab('generate')}
              className={`transition cursor-pointer ${activeTab === 'generate' ? 'text-white font-semibold' : 'text-slate-400 hover:text-slate-200'}`}
            >
              Studio
            </button>
            <button
              onClick={() => setActiveTab('upload')}
              className={`transition cursor-pointer hidden sm:inline ${activeTab === 'upload' ? 'text-white font-semibold' : 'text-slate-400 hover:text-slate-200'}`}
            >
              Partners
            </button>

            {/* Plus Pill */}
            <span className="px-2 py-0.5 rounded text-xs font-semibold text-sky-400 bg-sky-500/10 border border-sky-500/30">
              Plus
            </span>

            {/* Theme Toggle Button */}
            <button
              type="button"
              onClick={() => setIsDark(!isDark)}
              className="p-1 text-slate-400 hover:text-white transition cursor-pointer"
              title={isDark ? "Switch to Light Mode" : "Switch to Dark Mode"}
              aria-label="Toggle Theme"
            >
              {isDark ? (
                <svg className="w-4 h-4 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 9H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
              ) : (
                <svg className="w-4 h-4 text-sky-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content View */}
      <main className="grow max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 relative z-10">
        {tabs.find((t) => t.id === activeTab)?.component}
      </main>

      {/* Tailwind Modern Footer */}
      <footer className="mt-auto border-t border-slate-200 dark:border-slate-800/80 bg-white/80 dark:bg-slate-950/80 backdrop-blur-md py-6 text-center text-xs text-slate-500 dark:text-slate-400 transition-colors relative z-10">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5 text-sky-400" viewBox="0 0 54 33" fill="none">
              <path fillRule="evenodd" clipRule="evenodd" d="M27 0c-7.2 0-11.7 3.6-13.5 10.8 2.7-3.6 5.85-4.95 9.45-4.05 2.054.513 3.522 2.004 5.147 3.653C30.744 13.09 33.808 16.2 40.5 16.2c7.2 0 11.7-3.6 13.5-10.8-2.7 3.6-5.85 4.95-9.45 4.05-2.054-.514-3.522-2.004-5.147-3.653C36.756 3.11 33.692 0 27 0zM13.5 16.2C6.3 16.2 1.8 19.8 0 27c2.7-3.6 5.85-4.95 9.45-4.05 2.054.514 3.522 2.004 5.147 3.653C17.244 29.29 20.308 32.4 27 32.4c7.2 0 11.7-3.6 13.5-10.8-2.7 3.6-5.85 4.95-9.45 4.05-2.054-.513-3.522-2.004-5.147-3.653C23.256 19.31 20.192 16.2 13.5 16.2z" fill="currentColor"/>
            </svg>
            <span className="font-semibold text-slate-700 dark:text-slate-300">PaperMind Studio</span>
            <span className="text-slate-400 dark:text-slate-600">·</span>
            <span>Next-Gen Academic Exam Engineering</span>
          </div>
          <div className="flex items-center gap-2 font-mono text-[11px] text-slate-500">
            <span className="px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">PaddleOCR PP-OCRv5</span>
            <span className="px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">spaCy NLP</span>
            <span className="px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">ReportLab Typesetter</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
      </footer>
    </div>
  );
}
