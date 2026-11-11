import React, { useState } from 'react';
import { generatePaper, validatePaper, downloadPaperPDF } from '../api/client';

export default function GeneratePage() {
  const [subject, setSubject] = useState('Data Structures and Algorithms');
  const [courseCode, setCourseCode] = useState('CS201');
  const [examName, setExamName] = useState('End Semester Degree Examinations');
  const [institution, setInstitution] = useState('National Institute of Technology');
  const [durationHours, setDurationHours] = useState(2.0);
  const [incorporateTrends, setIncorporateTrends] = useState(true);
  const [avoidDuplicates, setAvoidDuplicates] = useState(true);
  const [selectedPreset, setSelectedPreset] = useState('midterm');
  const [showMetadata, setShowMetadata] = useState(false);

  // Syllabus Customization State
  const [useCustomSyllabus, setUseCustomSyllabus] = useState(false);
  const [syllabusMode, setSyllabusMode] = useState('structured'); // 'structured' | 'text'
  const [rawSyllabusText, setRawSyllabusText] = useState('');
  const [customUnits, setCustomUnits] = useState([
    {
      id: 1,
      unit_name: 'Supervised Learning & Linear Models',
      topics: 'Linear Regression, Cost Functions, Gradient Descent, Normal Equation, Regularization (L1/L2 Ridge & Lasso)',
      keywords: 'linear regression, cost function, gradient descent, regularization, Ridge, Lasso, MSE',
    },
    {
      id: 2,
      unit_name: 'Classification & Neural Networks',
      topics: 'Logistic Regression, Multi-class Softmax, Support Vector Machines, Perceptron, Backpropagation, Activation Functions',
      keywords: 'logistic regression, sigmoid, SVM, margin, perceptron, backpropagation, activation, ReLU',
    },
    {
      id: 3,
      unit_name: 'Tree Ensembles & Model Evaluation',
      topics: 'Decision Trees (ID3, CART), Information Gain, Random Forests, Boosting (AdaBoost, XGBoost), Cross-Validation, ROC-AUC',
      keywords: 'decision tree, entropy, Gini, random forest, bagging, boosting, AdaBoost, cross-validation, ROC-AUC',
    },
  ]);

  const SYLLABUS_PRESETS = {
    ml: {
      name: 'Machine Learning (CS304)',
      subject: 'Machine Learning',
      course_code: 'CS304',
      units: [
        {
          id: 1,
          unit_name: 'Supervised Learning & Linear Models',
          topics: 'Linear Regression, Cost Functions, Gradient Descent, Normal Equation, Polynomial Regression, Regularization (L1/L2 Ridge & Lasso)',
          keywords: 'linear regression, cost function, gradient descent, regularization, Ridge, Lasso, MSE, convex optimization',
        },
        {
          id: 2,
          unit_name: 'Classification & Support Vector Machines',
          topics: 'Logistic Regression, Cross-Entropy Loss, Decision Boundaries, Multi-class Softmax, Support Vector Machines, Kernel Trick',
          keywords: 'logistic regression, cross-entropy, sigmoid, SVM, margin, hyperplane, kernel, soft margin, slack variable',
        },
        {
          id: 3,
          unit_name: 'Decision Trees & Ensemble Methods',
          topics: 'Decision Trees (ID3, C4.5, CART), Information Gain, Gini Impurity, Bagging & Random Forests, Boosting (AdaBoost, Gradient Boosting)',
          keywords: 'decision tree, information gain, entropy, Gini impurity, random forest, bagging, boosting, AdaBoost, ensemble',
        },
        {
          id: 4,
          unit_name: 'Deep Neural Networks & Representation',
          topics: 'Perceptron, Multi-Layer Perceptron, Activation Functions (ReLU, Sigmoid), Backpropagation, Gradient vanishing/exploding, CNNs',
          keywords: 'perceptron, MLP, backpropagation, activation, ReLU, gradient descent, loss, convolutional, pooling, neural network',
        },
      ],
    },
    dbms: {
      name: 'Database Management Systems (CS204)',
      subject: 'Database Management Systems',
      course_code: 'CS204',
      units: [
        {
          id: 1,
          unit_name: 'Data Models & Relational Architecture',
          topics: 'Entity-Relationship (ER) Modeling, Relational Data Model, Integrity Constraints, Relational Algebra, Relational Calculus',
          keywords: 'ER diagram, entity, relationship, primary key, foreign key, relational algebra, selection, projection, join',
        },
        {
          id: 2,
          unit_name: 'SQL & Relational Normalization',
          topics: 'SQL DDL/DML, Nested Queries, Aggregations, Functional Dependencies, Normal Forms (1NF, 2NF, 3NF, BCNF), Decomposition',
          keywords: 'SQL, SELECT, JOIN, GROUP BY, functional dependency, 1NF, 2NF, 3NF, BCNF, normalization, lossless decomposition',
        },
        {
          id: 3,
          unit_name: 'Transaction & Concurrency Control',
          topics: 'ACID Properties, Transaction States, Schedules, Conflict Serializability, Two-Phase Locking (2PL), Deadlocks',
          keywords: 'ACID, atomicity, consistency, isolation, durability, serializability, 2PL, lock, deadlock, timestamp ordering',
        },
        {
          id: 4,
          unit_name: 'Indexing, Storage & Query Optimization',
          topics: 'RAID, B-Trees and B+ Trees, Static & Dynamic Hashing, Query Optimization Cost Estimation',
          keywords: 'indexing, B-tree, B+ tree, hash index, query optimization, cost estimation, file organization',
        },
      ],
    },
    os: {
      name: 'Operating Systems (CS203)',
      subject: 'Operating Systems',
      course_code: 'CS203',
      units: [
        {
          id: 1,
          unit_name: 'Process & Thread Management',
          topics: 'Process Concept, Process Control Block (PCB), Context Switching, Thread Models, Inter-Process Communication (IPC)',
          keywords: 'process, PCB, thread, context switch, IPC, shared memory, message passing, fork, exec',
        },
        {
          id: 2,
          unit_name: 'CPU Scheduling & Synchronization',
          topics: 'Scheduling Criteria, FCFS, SJF, Round Robin, Priority Scheduling, Critical Section, Semaphores, Mutexes',
          keywords: 'CPU scheduling, FCFS, SJF, Round Robin, semaphore, mutex, critical section, race condition, monitor',
        },
        {
          id: 3,
          unit_name: 'Deadlocks & Virtual Memory',
          topics: 'Deadlock Conditions, Banker Algorithm, Paging, Segmentation, Virtual Memory, Page Replacement (LRU, FIFO)',
          keywords: 'deadlock, Banker algorithm, paging, page table, TLB, segmentation, virtual memory, page fault, LRU, FIFO',
        },
        {
          id: 4,
          unit_name: 'File Systems & Disk I/O',
          topics: 'File Concepts, Directory Structures, Allocation Methods (Contiguous, Linked, Indexed), Disk Scheduling (SSTF, SCAN, C-SCAN)',
          keywords: 'file system, inode, contiguous allocation, indexed allocation, disk scheduling, SSTF, SCAN, C-SCAN',
        },
      ],
    },
    dsa: {
      name: 'Data Structures & Algorithms (CS201)',
      subject: 'Data Structures and Algorithms',
      course_code: 'CS201',
      units: [
        {
          id: 1,
          unit_name: 'Linear Data Structures',
          topics: 'Abstract Data Types, Arrays, Linked Lists (Singly, Doubly, Circular), Stacks & applications, Queues & variants',
          keywords: 'ADT, array, linked list, singly linked, doubly linked, circular linked, stack, LIFO, queue, FIFO, priority queue',
        },
        {
          id: 2,
          unit_name: 'Trees & Hierarchical Structures',
          topics: 'Binary Trees & Traversals, Binary Search Trees, AVL Trees & Rotations, B-Trees and B+ Trees, Heaps & Heap Sort',
          keywords: 'binary tree, BST, traversal, inorder, preorder, postorder, AVL, rotation, balance factor, B-tree, heap, heapify',
        },
        {
          id: 3,
          unit_name: 'Graphs & Network Algorithms',
          topics: 'Graph representations (Adjacency Matrix/List), BFS, DFS, Dijkstra, Bellman-Ford, Prim, Kruskal, Topological Sort',
          keywords: 'graph, vertex, edge, adjacency matrix, BFS, DFS, Dijkstra, Bellman-Ford, MST, Prim, Kruskal, topological sort',
        },
        {
          id: 4,
          unit_name: 'Sorting, Searching & Hashing',
          topics: 'Bubble, Selection, Insertion Sort, Merge Sort, Quick Sort, Binary Search, Hash Tables & Collision Resolution',
          keywords: 'sorting, bubble sort, merge sort, quick sort, partition, binary search, hashing, hash table, collision, chaining',
        },
        {
          id: 5,
          unit_name: 'Algorithm Design & Complexity',
          topics: 'Asymptotic Notations (Big-O, Omega, Theta), Divide & Conquer, Greedy Algorithms, Dynamic Programming, Backtracking',
          keywords: 'time complexity, Big-O, asymptotic, divide and conquer, greedy, dynamic programming, memoization, backtracking',
        },
      ],
    },
  };

  const loading_unused = false;

  const [loading, setLoading] = useState(false);
  const [generatedPaper, setGeneratedPaper] = useState(null);
  const [validationReport, setValidationReport] = useState(null);
  const [showSolutions, setShowSolutions] = useState(false);
  const [error, setError] = useState(null);

  const [sections, setSections] = useState([
    {
      id: 1,
      section_name: 'Part A',
      description: '',
      question_count: 5,
      questions_to_attempt: 5,
      marks_per_question: 2.0,
      allowed_question_types: ['short'],
      has_internal_choice: false,
      unit_coverage: [],
    },
    {
      id: 2,
      section_name: 'Part B',
      description: '',
      question_count: 4,
      questions_to_attempt: 4,
      marks_per_question: 10.0,
      allowed_question_types: ['long'],
      has_internal_choice: true,
      unit_coverage: [],
    },
  ]);

  const PRESETS = {
    endsem: {
      name: 'End-Semester (100M)',
      duration: 3.0,
      sections: [
        {
          id: 1,
          section_name: 'Part A',
          description: '',
          question_count: 10,
          questions_to_attempt: 10,
          marks_per_question: 2.0,
          allowed_question_types: ['short'],
          has_internal_choice: false,
          unit_coverage: [],
        },
        {
          id: 2,
          section_name: 'Part B',
          description: '',
          question_count: 5,
          questions_to_attempt: 5,
          marks_per_question: 16.0,
          allowed_question_types: ['long'],
          has_internal_choice: true,
          unit_coverage: [],
        },
      ],
    },
    midterm: {
      name: 'Mid-Term Exam (50M)',
      duration: 2.0,
      sections: [
        {
          id: 1,
          section_name: 'Part A',
          description: '',
          question_count: 5,
          questions_to_attempt: 5,
          marks_per_question: 2.0,
          allowed_question_types: ['short'],
          has_internal_choice: false,
          unit_coverage: [],
        },
        {
          id: 2,
          section_name: 'Part B',
          description: '',
          question_count: 4,
          questions_to_attempt: 4,
          marks_per_question: 10.0,
          allowed_question_types: ['long'],
          has_internal_choice: true,
          unit_coverage: [],
        },
      ],
    },
    threetier: {
      name: '3-Tier Exam (75M)',
      duration: 3.0,
      sections: [
        {
          id: 1,
          section_name: 'Section A (Objective & Short)',
          description: '',
          question_count: 10,
          questions_to_attempt: 10,
          marks_per_question: 1.0,
          allowed_question_types: ['short', 'mcq'],
          has_internal_choice: false,
          unit_coverage: [],
        },
        {
          id: 2,
          section_name: 'Section B (Conceptual)',
          description: '',
          question_count: 5,
          questions_to_attempt: 5,
          marks_per_question: 5.0,
          allowed_question_types: ['medium'],
          has_internal_choice: false,
          unit_coverage: [],
        },
        {
          id: 3,
          section_name: 'Section C (Analytical & Problems)',
          description: '',
          question_count: 4,
          questions_to_attempt: 4,
          marks_per_question: 10.0,
          allowed_question_types: ['long', 'numerical'],
          has_internal_choice: true,
          unit_coverage: [],
        },
      ],
    },
    choice: {
      name: 'Choice Pattern (50M)',
      duration: 2.0,
      sections: [
        {
          id: 1,
          section_name: 'Part A',
          description: '',
          question_count: 5,
          questions_to_attempt: 5,
          marks_per_question: 2.0,
          allowed_question_types: ['short'],
          has_internal_choice: false,
          unit_coverage: [],
        },
        {
          id: 2,
          section_name: 'Part B',
          description: '',
          question_count: 8,
          questions_to_attempt: 5,
          marks_per_question: 8.0,
          allowed_question_types: ['long'],
          has_internal_choice: false,
          unit_coverage: [],
        },
      ],
    },
    quiz: {
      name: 'Unit Quiz (25M)',
      duration: 1.0,
      sections: [
        {
          id: 1,
          section_name: 'Part A (Quick Recall)',
          description: '',
          question_count: 5,
          questions_to_attempt: 5,
          marks_per_question: 1.0,
          allowed_question_types: ['mcq', 'short'],
          has_internal_choice: false,
          unit_coverage: [],
        },
        {
          id: 2,
          section_name: 'Part B (Descriptive)',
          description: '',
          question_count: 4,
          questions_to_attempt: 4,
          marks_per_question: 5.0,
          allowed_question_types: ['medium'],
          has_internal_choice: false,
          unit_coverage: [],
        },
      ],
    },
  };

  const applyPreset = (key) => {
    setSelectedPreset(key);
    if (PRESETS[key]) {
      setDurationHours(PRESETS[key].duration);
      setSections(PRESETS[key].sections.map((s) => ({ ...s, id: Math.random() })));
    }
  };

  const addSection = () => {
    setSelectedPreset('custom');
    const nextChar = String.fromCharCode(65 + sections.length);
    const newSec = {
      id: Date.now(),
      section_name: `Part ${nextChar}`,
      description: '',
      question_count: 4,
      questions_to_attempt: 4,
      marks_per_question: 5.0,
      allowed_question_types: ['medium'],
      has_internal_choice: false,
      unit_coverage: [],
    };
    setSections([...sections, newSec]);
  };

  const removeSection = (id) => {
    if (sections.length <= 1) {
      alert('A question paper must have at least one section.');
      return;
    }
    setSelectedPreset('custom');
    setSections(sections.filter((s) => s.id !== id));
  };

  const updateSection = (id, field, value) => {
    setSelectedPreset('custom');
    setSections(sections.map((s) => (s.id === id ? { ...s, [field]: value } : s)));
  };

  const moveSection = (index, direction) => {
    setSelectedPreset('custom');
    const targetIndex = index + direction;
    if (targetIndex < 0 || targetIndex >= sections.length) return;
    const newSecs = [...sections];
    const temp = newSecs[index];
    newSecs[index] = newSecs[targetIndex];
    newSecs[targetIndex] = temp;
    setSections(newSecs);
  };

  const toggleQuestionType = (secId, type) => {
    setSelectedPreset('custom');
    setSections(
      sections.map((s) => {
        if (s.id !== secId) return s;
        const types = s.allowed_question_types || [];
        const exists = types.includes(type);
        let newTypes = exists ? types.filter((t) => t !== type) : [...types, type];
        if (newTypes.length === 0) newTypes = ['medium'];
        return { ...s, allowed_question_types: newTypes };
      })
    );
  };

  const toggleUnitCoverage = (secId, unitNo) => {
    setSelectedPreset('custom');
    setSections(
      sections.map((s) => {
        if (s.id !== secId) return s;
        const units = s.unit_coverage || [];
        const exists = units.includes(unitNo);
        const newUnits = exists ? units.filter((u) => u !== unitNo) : [...units, unitNo].sort((a, b) => a - b);
        return { ...s, unit_coverage: newUnits };
      })
    );
  };

  // Custom Syllabus Handlers
  const addCustomUnit = () => {
    const nextNo = customUnits.length + 1;
    setCustomUnits([
      ...customUnits,
      {
        id: Date.now(),
        unit_name: `Unit ${nextNo}: Domain & Concepts`,
        topics: '',
        keywords: '',
      },
    ]);
  };

  const removeCustomUnit = (id) => {
    if (customUnits.length <= 1) {
      alert('A custom syllabus must have at least one unit.');
      return;
    }
    setCustomUnits(customUnits.filter((u) => u.id !== id));
  };

  const updateCustomUnit = (id, field, value) => {
    setCustomUnits(customUnits.map((u) => (u.id === id ? { ...u, [field]: value } : u)));
  };

  const applySyllabusPreset = (presetKey) => {
    const p = SYLLABUS_PRESETS[presetKey];
    if (!p) return;
    setSubject(p.subject);
    setCourseCode(p.course_code);
    setCustomUnits(p.units.map((u) => ({ ...u, id: Math.random() })));
    setUseCustomSyllabus(true);
  };

  const parseSyllabusFromText = () => {
    if (!rawSyllabusText.trim()) {
      alert('Please enter or paste your syllabus text first.');
      return;
    }

    const lines = rawSyllabusText.split('\n');
    const parsedUnits = [];
    let currentUnit = null;

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      const unitMatch =
        trimmed.match(/^(?:###\s*)?(?:unit|module|chapter|part)\s*(\d+)[\s*:\-–—]+(.*)$/i) ||
        trimmed.match(/^(\d+)[\.\)]\s+(.*)$/);

      if (unitMatch) {
        if (currentUnit) parsedUnits.push(currentUnit);
        currentUnit = {
          id: Date.now() + Math.random(),
          unit_name: unitMatch[2] ? unitMatch[2].trim() : `Unit ${unitMatch[1]}`,
          topics: '',
          keywords: '',
        };
      } else if (currentUnit) {
        if (/^topics?[\s*:\-–—]+/i.test(trimmed)) {
          const topText = trimmed.replace(/^topics?[\s*:\-–—]+/i, '').trim();
          currentUnit.topics = currentUnit.topics ? `${currentUnit.topics}, ${topText}` : topText;
        } else if (/^keywords?[\s*:\-–—]+/i.test(trimmed)) {
          const kwText = trimmed.replace(/^keywords?[\s*:\-–—]+/i, '').trim();
          currentUnit.keywords = currentUnit.keywords ? `${currentUnit.keywords}, ${kwText}` : kwText;
        } else {
          currentUnit.topics = currentUnit.topics ? `${currentUnit.topics}, ${trimmed}` : trimmed;
        }
      }
    }
    if (currentUnit) parsedUnits.push(currentUnit);

    if (parsedUnits.length > 0) {
      setCustomUnits(parsedUnits);
      setSyllabusMode('structured');
      alert(`Successfully parsed ${parsedUnits.length} syllabus unit(s)!`);
    } else {
      setCustomUnits([
        {
          id: Date.now(),
          unit_name: 'Custom Unit 1',
          topics: lines.filter((l) => l.trim()).join(', '),
          keywords: '',
        },
      ]);
      setSyllabusMode('structured');
      alert('Created 1 unit from provided text.');
    }
  };

  const calculatedTotalMarks = sections.reduce((sum, s) => {
    const attempt = s.questions_to_attempt ? parseInt(s.questions_to_attempt, 10) : parseInt(s.question_count, 10) || 1;
    return sum + attempt * (parseFloat(s.marks_per_question) || 0);
  }, 0);

  const totalPrimaryQuestions = sections.reduce((sum, s) => sum + (parseInt(s.question_count, 10) || 1), 0);
  const totalGeneratedQuestions = sections.reduce((sum, s) => {
    const count = parseInt(s.question_count, 10) || 1;
    return sum + (s.has_internal_choice ? count * 2 : count);
  }, 0);

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    setGeneratedPaper(null);
    setValidationReport(null);

    const bp = {
      subject,
      course_code: courseCode,
      total_marks: calculatedTotalMarks,
      duration_hours: parseFloat(durationHours),
      institution_name: institution,
      examination_name: examName,
      sections: sections.map((s) => {
        const qCount = parseInt(s.question_count, 10) || 1;
        const qAttempt = s.questions_to_attempt ? parseInt(s.questions_to_attempt, 10) : qCount;
        const marksPerQ = parseFloat(s.marks_per_question) || 1.0;
        let desc = s.description;
        if (!desc || !desc.trim()) {
          if (qAttempt < qCount) {
            desc = `Answer any ${qAttempt} questions out of ${qCount}. (${qAttempt} x ${marksPerQ} = ${qAttempt * marksPerQ} Marks)`;
          } else if (s.has_internal_choice) {
            desc = `Answer ALL questions. Each question has an internal choice. (${qCount} x ${marksPerQ} = ${qCount * marksPerQ} Marks)`;
          } else {
            desc = `Answer ALL questions. (${qCount} x ${marksPerQ} = ${qCount * marksPerQ} Marks)`;
          }
        }
        return {
          section_name: s.section_name || 'Section',
          description: desc,
          question_count: qCount,
          questions_to_attempt: qAttempt,
          marks_per_question: marksPerQ,
          allowed_question_types: s.allowed_question_types && s.allowed_question_types.length > 0 ? s.allowed_question_types : ['medium'],
          has_internal_choice: !!s.has_internal_choice,
          unit_coverage: s.unit_coverage && s.unit_coverage.length > 0 ? s.unit_coverage : null,
        };
      }),
      incorporate_historical_trends: incorporateTrends,
      avoid_recent_duplicates: avoidDuplicates,
    };

    let syllabusPayload = null;
    if (useCustomSyllabus && customUnits.length > 0) {
      syllabusPayload = {
        subject: subject || 'Custom Course',
        course_code: courseCode || null,
        units: customUnits.map((u, idx) => ({
          unit_number: idx + 1,
          unit_name: u.unit_name || `Unit ${idx + 1}`,
          topics: (Array.isArray(u.topics) ? u.topics : (u.topics || '').split(','))
            .map((t) => t.trim())
            .filter(Boolean),
          keywords: (Array.isArray(u.keywords) ? u.keywords : (u.keywords || '').split(','))
            .map((k) => k.trim())
            .filter(Boolean),
        })),
      };
    }

    try {
      const paper = await generatePaper(bp, syllabusPayload);
      setGeneratedPaper(paper);

      // Validate paper automatically
      const report = await validatePaper(paper, bp, syllabusPayload);
      setValidationReport(report);
    } catch (e) {
      setError(e.message || 'Generation failed');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPDF = async () => {
    if (!generatedPaper) return;
    try {
      await downloadPaperPDF(generatedPaper);
    } catch (e) {
      alert('Failed to download PDF: ' + e.message);
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Question Paper Generator & Validator</h1>
        <p className="text-slate-600 text-sm mt-1">
          Automated examination paper assembly using Gemini Generative AI, multi-tiered constraint optimization, independent validation, and camera-ready PDF export.
        </p>
      </div>

      {/* Top Banner & Presets */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-100 pb-4">
          <div>
            <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2">
              <span>Examination Paper Blueprint & Structure Customizer</span>
              <span className="text-[11px] font-semibold px-2 py-0.5 bg-indigo-50 text-indigo-700 rounded-full border border-indigo-200">
                Interactive Section Builder
              </span>
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Customize sections, question counts, marks, internal choices, question types, and unit weightage.
            </p>
          </div>

          {/* Preset Buttons */}
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-xs font-semibold text-slate-500 mr-1">Presets:</span>
            {Object.keys(PRESETS).map((key) => (
              <button
                key={key}
                type="button"
                onClick={() => applyPreset(key)}
                className={`px-2.5 py-1 text-xs rounded-lg font-medium transition cursor-pointer ${
                  selectedPreset === key
                    ? 'bg-indigo-600 text-white shadow-xs'
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                }`}
              >
                {PRESETS[key].name}
              </button>
            ))}
            <button
              type="button"
              onClick={() => setSelectedPreset('custom')}
              className={`px-2.5 py-1 text-xs rounded-lg font-medium transition cursor-pointer ${
                selectedPreset === 'custom'
                  ? 'bg-indigo-600 text-white shadow-xs'
                  : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
              }`}
            >
              Custom
            </button>
          </div>
        </div>

        {/* Live Calculation Banner */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-3 bg-gradient-to-r from-indigo-50/60 to-purple-50/60 rounded-lg border border-indigo-100 text-xs">
          <div>
            <span className="text-slate-500 block text-[11px]">Total Configured Marks</span>
            <span className="text-base font-bold text-indigo-700">{calculatedTotalMarks} Marks</span>
          </div>
          <div>
            <span className="text-slate-500 block text-[11px]">Duration</span>
            <span className="text-base font-bold text-slate-800">{durationHours} Hours</span>
          </div>
          <div>
            <span className="text-slate-500 block text-[11px]">Sections</span>
            <span className="text-base font-bold text-slate-800">{sections.length} Section{sections.length > 1 ? 's' : ''}</span>
          </div>
          <div>
            <span className="text-slate-500 block text-[11px]">Questions</span>
            <span className="text-base font-bold text-slate-800">{totalPrimaryQuestions} Primary ({totalGeneratedQuestions} with OR)</span>
          </div>
        </div>

        {/* Collapsible Metadata Settings */}
        <div className="border border-slate-200 rounded-lg overflow-hidden">
          <button
            type="button"
            onClick={() => setShowMetadata(!showMetadata)}
            className="w-full px-4 py-2 bg-slate-50 hover:bg-slate-100 flex items-center justify-between text-xs font-semibold text-slate-700 transition cursor-pointer"
          >
            <span>Paper Header & Examination Metadata ({subject} - {courseCode})</span>
            <span>{showMetadata ? '▲ Hide Details' : '▼ Edit Subject, Code, Institution & Duration'}</span>
          </button>
          {showMetadata && (
            <div className="p-4 bg-white grid grid-cols-1 md:grid-cols-3 gap-3 text-xs border-t border-slate-200">
              <div>
                <label className="font-semibold text-slate-600 block mb-1">Subject / Course Title</label>
                <input
                  type="text"
                  value={subject}
                  onChange={(e) => setSubject(e.target.value)}
                  className="w-full px-2.5 py-1.5 border border-slate-300 rounded text-xs focus:ring-1 focus:ring-indigo-500"
                />
              </div>
              <div>
                <label className="font-semibold text-slate-600 block mb-1">Course Code</label>
                <input
                  type="text"
                  value={courseCode}
                  onChange={(e) => setCourseCode(e.target.value)}
                  className="w-full px-2.5 py-1.5 border border-slate-300 rounded text-xs"
                />
              </div>
              <div>
                <label className="font-semibold text-slate-600 block mb-1">Duration (Hours)</label>
                <input
                  type="number"
                  step="0.5"
                  min="0.5"
                  max="6.0"
                  value={durationHours}
                  onChange={(e) => setDurationHours(parseFloat(e.target.value) || 1.0)}
                  className="w-full px-2.5 py-1.5 border border-slate-300 rounded text-xs"
                />
              </div>
              <div>
                <label className="font-semibold text-slate-600 block mb-1">Institution Name</label>
                <input
                  type="text"
                  value={institution}
                  onChange={(e) => setInstitution(e.target.value)}
                  className="w-full px-2.5 py-1.5 border border-slate-300 rounded text-xs"
                />
              </div>
              <div>
                <label className="font-semibold text-slate-600 block mb-1">Examination Title</label>
                <input
                  type="text"
                  value={examName}
                  onChange={(e) => setExamName(e.target.value)}
                  className="w-full px-2.5 py-1.5 border border-slate-300 rounded text-xs"
                />
              </div>
              <div className="flex flex-col justify-center space-y-1.5 pt-1">
                <label className="flex items-center gap-2 cursor-pointer text-slate-700">
                  <input
                    type="checkbox"
                    checked={incorporateTrends}
                    onChange={(e) => setIncorporateTrends(e.target.checked)}
                    className="rounded text-indigo-600"
                  />
                  <span>Prioritize Historical High-Frequency Topics</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer text-slate-700">
                  <input
                    type="checkbox"
                    checked={avoidDuplicates}
                    onChange={(e) => setAvoidDuplicates(e.target.checked)}
                    className="rounded text-indigo-600"
                  />
                  <span>De-duplicate against Past Papers</span>
                </label>
              </div>
            </div>
          )}
        </div>

        {/* Course Syllabus Configuration Card */}
        <div className="bg-slate-50/70 p-4.5 rounded-xl border border-slate-200 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-200 pb-3.5">
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                  <span>📖 Course Syllabus & Curriculum Context</span>
                </h3>
                <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded-full border ${
                  useCustomSyllabus
                    ? 'bg-emerald-500/10 text-emerald-700 border-emerald-500/20'
                    : 'bg-slate-500/10 text-slate-600 border-slate-500/20'
                }`}>
                  {useCustomSyllabus ? `Custom Curriculum (${customUnits.length} Units)` : 'Benchmark Syllabus (Default)'}
                </span>
              </div>
              <p className="text-xs text-slate-500 mt-0.5">
                {useCustomSyllabus
                  ? 'Questions will be strictly bound to your custom units, topics, and domain keywords.'
                  : 'Standard benchmark syllabus active (DSA). Enable custom syllabus to input your own course units.'}
              </p>
            </div>

            {/* Toggle Switch */}
            <label className="flex items-center gap-2.5 cursor-pointer self-start sm:self-center bg-white px-3 py-1.5 rounded-lg border border-slate-200 shadow-xs">
              <span className="text-xs font-semibold text-slate-700">
                {useCustomSyllabus ? 'Custom Syllabus ON' : 'Use Custom Syllabus'}
              </span>
              <input
                type="checkbox"
                checked={useCustomSyllabus}
                onChange={(e) => setUseCustomSyllabus(e.target.checked)}
                className="sr-only"
              />
              <div className={`w-9 h-5 flex items-center rounded-full p-0.5 transition duration-300 ${
                useCustomSyllabus ? 'bg-indigo-600' : 'bg-slate-300'
              }`}>
                <div className={`bg-white w-4 h-4 rounded-full shadow-md transform transition duration-300 ${
                  useCustomSyllabus ? 'translate-x-4' : 'translate-x-0'
                }`} />
              </div>
            </label>
          </div>

          {!useCustomSyllabus ? (
            <div className="bg-white/80 p-3.5 rounded-lg border border-slate-200/80 text-xs flex flex-col md:flex-row md:items-center justify-between gap-3">
              <div className="text-slate-600">
                <span className="font-semibold text-slate-800">Standard Syllabus Active:</span> Questions are automatically synthesized from the standard 5-unit curriculum (CS201 Data Structures & Algorithms). Turn on the toggle or select a preset to generate for any other subject.
              </div>
              <div className="flex flex-wrap items-center gap-1.5 shrink-0">
                <span className="text-[11px] font-medium text-slate-500">Quick-Load:</span>
                <button
                  type="button"
                  onClick={() => applySyllabusPreset('ml')}
                  className="px-2 py-1 bg-indigo-50 text-indigo-700 border border-indigo-200 rounded-md text-[11px] font-semibold hover:bg-indigo-100 transition cursor-pointer"
                >
                  ML (CS304)
                </button>
                <button
                  type="button"
                  onClick={() => applySyllabusPreset('dbms')}
                  className="px-2 py-1 bg-indigo-50 text-indigo-700 border border-indigo-200 rounded-md text-[11px] font-semibold hover:bg-indigo-100 transition cursor-pointer"
                >
                  DBMS (CS204)
                </button>
                <button
                  type="button"
                  onClick={() => applySyllabusPreset('os')}
                  className="px-2 py-1 bg-indigo-50 text-indigo-700 border border-indigo-200 rounded-md text-[11px] font-semibold hover:bg-indigo-100 transition cursor-pointer"
                >
                  OS (CS203)
                </button>
              </div>
            </div>
          ) : (
            <div className="space-y-3.5">
              {/* Syllabus Mode Tabs & Preset Loader */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5">
                <div className="flex items-center gap-1 bg-slate-200/70 p-0.5 rounded-lg text-xs font-medium">
                  <button
                    type="button"
                    onClick={() => setSyllabusMode('structured')}
                    className={`px-3 py-1 rounded-md transition cursor-pointer ${
                      syllabusMode === 'structured'
                        ? 'bg-white text-slate-900 shadow-xs font-semibold'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Structured Unit Cards ({customUnits.length})
                  </button>
                  <button
                    type="button"
                    onClick={() => setSyllabusMode('text')}
                    className={`px-3 py-1 rounded-md transition cursor-pointer ${
                      syllabusMode === 'text'
                        ? 'bg-white text-slate-900 shadow-xs font-semibold'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Raw Text / Paste Parser
                  </button>
                </div>

                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[11px] font-medium text-slate-500">Presets:</span>
                  {Object.keys(SYLLABUS_PRESETS).map((key) => (
                    <button
                      key={key}
                      type="button"
                      onClick={() => applySyllabusPreset(key)}
                      className="px-2 py-0.5 bg-white border border-slate-200 rounded text-[11px] font-semibold text-slate-700 hover:border-indigo-400 transition cursor-pointer"
                    >
                      {SYLLABUS_PRESETS[key].name.split(' ')[0]}
                    </button>
                  ))}
                  <button
                    type="button"
                    onClick={addCustomUnit}
                    className="px-2.5 py-0.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-[11px] font-semibold transition cursor-pointer"
                  >
                    + Add Unit
                  </button>
                </div>
              </div>

              {/* MODE 1: STRUCTURED UNIT CARDS */}
              {syllabusMode === 'structured' && (
                <div className="space-y-2.5 max-h-[380px] overflow-y-auto pr-1">
                  {customUnits.map((u, idx) => (
                    <div
                      key={u.id}
                      className="bg-white p-3.5 rounded-lg border border-slate-200 space-y-2.5 text-xs shadow-xs"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2 grow">
                          <span className="font-mono font-bold text-indigo-600 text-xs px-2 py-0.5 bg-indigo-50 rounded border border-indigo-200 shrink-0">
                            Unit {idx + 1}
                          </span>
                          <input
                            type="text"
                            value={u.unit_name}
                            onChange={(e) => updateCustomUnit(u.id, 'unit_name', e.target.value)}
                            placeholder="Unit Title (e.g. Supervised Learning & Regression)"
                            className="font-semibold text-xs text-slate-900 bg-slate-50 px-2.5 py-1 border border-slate-200 rounded-md w-full focus:ring-1 focus:ring-indigo-500"
                          />
                        </div>
                        <button
                          type="button"
                          onClick={() => removeCustomUnit(u.id)}
                          disabled={customUnits.length <= 1}
                          className="px-2 py-1 bg-rose-50 text-rose-600 hover:bg-rose-100 rounded border border-rose-200 text-[11px] font-bold disabled:opacity-30 cursor-pointer"
                          title="Remove Unit"
                        >
                          ✕
                        </button>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                        <div>
                          <label className="block text-[11px] font-medium text-slate-500 mb-0.5">
                            Topics (Comma-separated syllabus topics)
                          </label>
                          <input
                            type="text"
                            value={u.topics}
                            onChange={(e) => updateCustomUnit(u.id, 'topics', e.target.value)}
                            placeholder="e.g. Linear Regression, Cost Functions, Gradient Descent"
                            className="w-full px-2.5 py-1 bg-slate-50 border border-slate-200 rounded-md text-slate-900 text-xs"
                          />
                        </div>
                        <div>
                          <label className="block text-[11px] font-medium text-slate-500 mb-0.5">
                            Domain Keywords (Technical terms for prompt & matching)
                          </label>
                          <input
                            type="text"
                            value={u.keywords}
                            onChange={(e) => updateCustomUnit(u.id, 'keywords', e.target.value)}
                            placeholder="e.g. regression, MSE, convex, normal equation"
                            className="w-full px-2.5 py-1 bg-slate-50 border border-slate-200 rounded-md text-slate-900 font-mono text-[11px]"
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* MODE 2: RAW TEXT / PASTE PARSER */}
              {syllabusMode === 'text' && (
                <div className="bg-white p-4 rounded-lg border border-slate-200 space-y-3 text-xs">
                  <div className="flex items-center justify-between">
                    <label className="font-semibold text-slate-800">
                      Paste Course Syllabus Text / Curriculum Outline:
                    </label>
                    <span className="text-[11px] text-slate-400">
                      Format: "Unit 1: Name", "Topics: A, B", "Keywords: C, D"
                    </span>
                  </div>
                  <textarea
                    rows={6}
                    value={rawSyllabusText}
                    onChange={(e) => setRawSyllabusText(e.target.value)}
                    placeholder={`Unit 1: Supervised Learning & Regression\nTopics: Linear Regression, Cost Functions, Gradient Descent, Normal Equation\nKeywords: regression, MSE, convex, normal equation\n\nUnit 2: Neural Networks & Deep Learning\nTopics: Perceptron, Multi-Layer Perceptron, Backpropagation, Activation Functions\nKeywords: weights, sigmoid, ReLU, gradient`}
                    className="w-full p-2.5 bg-slate-50 border border-slate-200 rounded-lg text-slate-900 font-mono text-xs focus:ring-1 focus:ring-indigo-500"
                  />
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] text-slate-500">
                      Parses headers like "Unit 1: Title", "Module 2 - Title", or numbered lists.
                    </span>
                    <button
                      type="button"
                      onClick={parseSyllabusFromText}
                      className="px-4 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-bold transition flex items-center gap-1.5 cursor-pointer shadow-xs"
                    >
                      <span>⚡</span>
                      <span>Parse to Structured Units</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Section List / Customizer */}
        <div className="space-y-4 pt-2">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-800">
              Configured Sections ({sections.length})
            </h3>
            <button
              type="button"
              onClick={addSection}
              className="px-3 py-1.5 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 text-xs font-semibold rounded-lg border border-indigo-200 flex items-center gap-1 transition cursor-pointer"
            >
              <span>+ Add New Section</span>
            </button>
          </div>

          <div className="space-y-3">
            {sections.map((s, idx) => {
              const qCount = parseInt(s.question_count, 10) || 1;
              const qAttempt = s.questions_to_attempt ? parseInt(s.questions_to_attempt, 10) : qCount;
              const marksEach = parseFloat(s.marks_per_question) || 1.0;
              const secTotal = qAttempt * marksEach;
              const availableUnits = useCustomSyllabus && customUnits.length > 0
                ? customUnits.map((_, i) => i + 1)
                : [1, 2, 3, 4, 5];

              return (
                <div key={s.id} className="bg-slate-50/80 p-4 rounded-xl border border-slate-200 space-y-3">
                  {/* Section Header Row */}
                  <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 pb-2">
                    <div className="flex items-center gap-2 grow">
                      <span className="font-bold text-xs text-slate-500 w-6">#{idx + 1}</span>
                      <input
                        type="text"
                        value={s.section_name}
                        onChange={(e) => updateSection(s.id, 'section_name', e.target.value)}
                        placeholder="e.g. Part A, Section 1"
                        className="font-bold text-xs text-slate-800 bg-white px-2 py-1 border border-slate-300 rounded max-w-xs focus:ring-1 focus:ring-indigo-500"
                      />
                      <span className="px-2 py-0.5 bg-indigo-100 text-indigo-800 font-bold text-[11px] rounded-full">
                        {qAttempt} x {marksEach}m = {secTotal} Marks
                      </span>
                      {s.has_internal_choice && (
                        <span className="px-2 py-0.5 bg-amber-100 text-amber-800 font-semibold text-[11px] rounded-full">
                          Internal OR Choice Enabled ({qCount * 2} Questions)
                        </span>
                      )}
                    </div>

                    <div className="flex items-center gap-1 text-xs">
                      <button
                        type="button"
                        onClick={() => moveSection(idx, -1)}
                        disabled={idx === 0}
                        title="Move Up"
                        className="px-2 py-1 bg-white hover:bg-slate-100 border border-slate-200 rounded text-slate-600 disabled:opacity-30 cursor-pointer"
                      >
                        ▲
                      </button>
                      <button
                        type="button"
                        onClick={() => moveSection(idx, 1)}
                        disabled={idx === sections.length - 1}
                        title="Move Down"
                        className="px-2 py-1 bg-white hover:bg-slate-100 border border-slate-200 rounded text-slate-600 disabled:opacity-30 cursor-pointer"
                      >
                        ▼
                      </button>
                      <button
                        type="button"
                        onClick={() => removeSection(s.id)}
                        disabled={sections.length <= 1}
                        title="Remove Section"
                        className="px-2 py-1 bg-rose-50 hover:bg-rose-100 border border-rose-200 text-rose-700 rounded font-semibold disabled:opacity-30 cursor-pointer"
                      >
                        ✕ Remove
                      </button>
                    </div>
                  </div>

                  {/* Parameters Grid */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                    <div>
                      <label className="block text-slate-600 font-medium mb-1">Question Count</label>
                      <input
                        type="number"
                        min="1"
                        max="20"
                        value={s.question_count}
                        onChange={(e) => updateSection(s.id, 'question_count', parseInt(e.target.value, 10) || 1)}
                        className="w-full px-2 py-1 bg-white border border-slate-300 rounded text-xs"
                      />
                    </div>

                    <div>
                      <label className="block text-slate-600 font-medium mb-1">
                        Questions to Attempt
                      </label>
                      <input
                        type="number"
                        min="1"
                        max={s.question_count}
                        value={s.questions_to_attempt || s.question_count}
                        onChange={(e) => updateSection(s.id, 'questions_to_attempt', parseInt(e.target.value, 10) || 1)}
                        className="w-full px-2 py-1 bg-white border border-slate-300 rounded text-xs"
                      />
                    </div>

                    <div>
                      <label className="block text-slate-600 font-medium mb-1">Marks per Question</label>
                      <input
                        type="number"
                        min="0.5"
                        step="0.5"
                        value={s.marks_per_question}
                        onChange={(e) => updateSection(s.id, 'marks_per_question', parseFloat(e.target.value) || 1.0)}
                        className="w-full px-2 py-1 bg-white border border-slate-300 rounded text-xs"
                      />
                    </div>

                    <div>
                      <label className="block text-slate-600 font-medium mb-1">Section Attempt Marks</label>
                      <div className="px-2 py-1 bg-slate-100 border border-slate-200 rounded font-bold text-slate-800 text-xs">
                        {secTotal} Marks
                      </div>
                    </div>
                  </div>

                  {/* Question Types & Internal Choice */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs pt-1">
                    <div>
                      <label className="block text-slate-600 font-medium mb-1.5">Allowed Question Types:</label>
                      <div className="flex flex-wrap gap-1.5">
                        {[
                          { key: 'short', label: 'Short' },
                          { key: 'medium', label: 'Medium' },
                          { key: 'long', label: 'Long / Essay' },
                          { key: 'numerical', label: 'Numerical' },
                          { key: 'mcq', label: 'MCQ' },
                        ].map((qt) => {
                          const active = (s.allowed_question_types || []).includes(qt.key);
                          return (
                            <button
                              key={qt.key}
                              type="button"
                              onClick={() => toggleQuestionType(s.id, qt.key)}
                              className={`px-2 py-0.5 rounded text-[11px] font-semibold transition border cursor-pointer ${
                                active
                                  ? 'bg-indigo-600 text-white border-indigo-600'
                                  : 'bg-white text-slate-600 border-slate-300 hover:bg-slate-100'
                              }`}
                            >
                              {active ? '✓ ' : ''}{qt.label}
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    <div>
                      <label className="block text-slate-600 font-medium mb-1.5">Choice Structure:</label>
                      <label className="flex items-center gap-2 cursor-pointer bg-white p-1.5 border border-slate-300 rounded">
                        <input
                          type="checkbox"
                          checked={s.has_internal_choice}
                          onChange={(e) => updateSection(s.id, 'has_internal_choice', e.target.checked)}
                          className="rounded text-indigo-600"
                        />
                        <span className="text-[11px] text-slate-700">
                          <b>Internal OR Choice:</b> Generates alternative Q(OR) for every question in same unit
                        </span>
                      </label>
                    </div>
                  </div>

                  {/* Unit Coverage Selector */}
                  <div className="text-xs pt-1">
                    <label className="block text-slate-600 font-medium mb-1.5">
                      Unit Coverage:
                    </label>
                    <div className="flex flex-wrap items-center gap-1.5">
                      <button
                        type="button"
                        onClick={() => updateSection(s.id, 'unit_coverage', [])}
                        className={`px-2 py-0.5 rounded text-[11px] font-semibold transition border cursor-pointer ${
                          !s.unit_coverage || s.unit_coverage.length === 0
                            ? 'bg-emerald-600 text-white border-emerald-600'
                            : 'bg-white text-slate-600 border-slate-300 hover:bg-slate-100'
                        }`}
                      >
                        {!s.unit_coverage || s.unit_coverage.length === 0 ? '✓ ' : ''}All Units ({availableUnits.length > 0 ? `${availableUnits[0]}-${availableUnits[availableUnits.length - 1]}` : '1-5'})
                      </button>
                      {availableUnits.map((u) => {
                        const active = (s.unit_coverage || []).includes(u);
                        return (
                          <button
                            key={u}
                            type="button"
                            onClick={() => toggleUnitCoverage(s.id, u)}
                            className={`px-2 py-0.5 rounded text-[11px] font-semibold transition border cursor-pointer ${
                              active
                                ? 'bg-indigo-600 text-white border-indigo-600'
                                : 'bg-white text-slate-600 border-slate-300 hover:bg-slate-100'
                            }`}
                          >
                            {active ? '✓ ' : ''}Unit {u}
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Custom Description */}
                  <div className="text-xs pt-1">
                    <div className="flex items-center justify-between mb-1">
                      <label className="text-slate-600 font-medium">Custom Instructions / Heading:</label>
                      <button
                        type="button"
                        onClick={() => {
                          const desc =
                            s.questions_to_attempt && s.questions_to_attempt < s.question_count
                              ? `Answer any ${s.questions_to_attempt} questions out of ${s.question_count}. (${s.questions_to_attempt} x ${s.marks_per_question} = ${secTotal} Marks)`
                              : s.has_internal_choice
                              ? `Answer ALL questions. Each question has an internal choice. (${s.question_count} x ${s.marks_per_question} = ${secTotal} Marks)`
                              : `Answer ALL questions. (${s.question_count} x ${s.marks_per_question} = ${secTotal} Marks)`;
                          updateSection(s.id, 'description', desc);
                        }}
                        className="text-[11px] text-indigo-600 hover:text-indigo-800 font-medium cursor-pointer"
                      >
                        Auto-Fill Instruction ↺
                      </button>
                    </div>
                    <input
                      type="text"
                      value={s.description}
                      onChange={(e) => updateSection(s.id, 'description', e.target.value)}
                      placeholder="e.g. Answer ALL questions. (5 x 2 = 10 Marks)"
                      className="w-full px-2.5 py-1.5 bg-white border border-slate-300 rounded text-xs"
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {error && (
          <div className="p-3 bg-red-50 text-red-700 text-xs rounded-lg border border-red-200">
            {error}
          </div>
        )}

        {/* Generate Button Row */}
        <div className="pt-2 flex flex-col md:flex-row items-center justify-between gap-4 border-t border-slate-100">
          <div className="text-xs text-slate-600">
            <span>Ready to synthesize </span>
            <b className="text-indigo-700">{calculatedTotalMarks} Marks</b>
            <span> examination paper with {sections.length} custom section(s).</span>
          </div>

          <button
            onClick={handleGenerate}
            disabled={loading}
            className="w-full md:w-auto px-8 py-3 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl text-xs font-bold transition shadow-md disabled:opacity-50 flex items-center justify-center gap-2 cursor-pointer"
          >
            {loading && <span className="animate-spin text-sm">↻</span>}
            <span>{loading ? 'Synthesizing & Validating Paper...' : 'Generate Complete Question Paper'}</span>
          </button>
        </div>
      </div>

      {/* Paper Preview & Validation */}
      <div>
        {generatedPaper ? (
            <div className="space-y-6">
              {/* Validation Summary Card */}
              {validationReport && (
                <div className={`p-5 rounded-xl border shadow-sm ${
                  validationReport.is_valid ? 'bg-emerald-50/70 border-emerald-200' : 'bg-rose-50/70 border-rose-200'
                }`}>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <span className={`w-3 h-3 rounded-full ${validationReport.is_valid ? 'bg-emerald-500' : 'bg-rose-500'}`} />
                      <h3 className="font-bold text-sm text-slate-900">
                        Independent Validation Engine Report
                      </h3>
                    </div>
                    <span className="px-3 py-1 bg-white font-bold text-xs rounded-full border border-slate-200 text-indigo-700 shadow-xs">
                      Quality Score: {validationReport.overall_quality_score}/100
                    </span>
                  </div>

                  <div className="space-y-1 text-xs text-slate-700">
                    {validationReport.passed_checks.map((c, idx) => (
                      <div key={idx} className="flex items-center gap-2 text-emerald-800">
                        <span>✓</span>
                        <span>{c}</span>
                      </div>
                    ))}
                    {validationReport.failed_checks.map((c, idx) => (
                      <div key={idx} className="flex items-center gap-2 text-rose-800 font-semibold">
                        <span>✗</span>
                        <span>{c}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Action Toolbar */}
              <div className="flex justify-between items-center bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                <button
                  onClick={() => setShowSolutions(!showSolutions)}
                  className="px-3 py-1.5 text-xs font-semibold rounded-lg border border-slate-300 hover:bg-slate-50 transition"
                >
                  {showSolutions ? 'Hide Rubrics & Solutions' : 'Show Marking Schemes'}
                </button>

                <button
                  onClick={handleDownloadPDF}
                  className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold rounded-lg shadow-sm transition"
                >
                  <span>Download Camera-Ready PDF</span>
                </button>
              </div>

              {/* Paper Preview */}
              <div className="bg-white p-8 rounded-xl border border-slate-300 shadow-sm font-serif text-slate-900 space-y-6">
                {/* Formal Exam Header */}
                <div className="text-center space-y-1 border-b-2 border-slate-900 pb-4">
                  <h2 className="text-lg font-bold tracking-wide uppercase">{generatedPaper.institution_name?.toLowerCase().includes('antigravity') ? 'NATIONAL INSTITUTE OF TECHNOLOGY' : generatedPaper.institution_name}</h2>
                  <h3 className="text-sm font-semibold uppercase">{generatedPaper.examination_name}</h3>
                  <div className="flex justify-between text-xs font-sans font-medium text-slate-700 pt-2 border-t border-slate-200 mt-2">
                    <span><b>Course:</b> {generatedPaper.course_code} - {generatedPaper.subject}</span>
                    <span><b>Max Marks:</b> {generatedPaper.total_marks}</span>
                    <span><b>Time:</b> {generatedPaper.duration_hours} Hours</span>
                  </div>
                </div>

                {/* Sections */}
                {generatedPaper.sections.map((sec, sIdx) => (
                  <div key={sIdx} className="space-y-4">
                    <div className="text-center font-sans">
                      <h4 className="font-bold text-sm tracking-wide uppercase">{sec.section_name}</h4>
                      <p className="text-xs italic text-slate-600">{sec.description}</p>
                    </div>

                    <div className="space-y-4 text-xs">
                      {sec.questions.map((q, qIdx) => {
                        const isOrChoice = q.question_number?.includes('(OR)');
                        return (
                          <div key={q.id || qIdx} className="space-y-1">
                            {isOrChoice && (
                              <div className="text-center font-bold text-[11px] text-slate-500 my-2">
                                [ OR ]
                              </div>
                            )}
                            <div className="flex justify-between items-start gap-4">
                              <span className="font-bold font-sans w-12 shrink-0">
                                {isOrChoice ? '' : `${q.question_number}.`}
                              </span>
                              <div className="grow space-y-1">
                                <p className="leading-relaxed">{q.question_text}</p>
                                {q.sub_parts && q.sub_parts.length > 0 && (
                                  <div className="pl-4 space-y-1 border-l border-slate-300 mt-1">
                                    {q.sub_parts.map((sp, idx) => (
                                      <div key={idx} className="flex justify-between">
                                        <span>({sp.label}) {sp.text}</span>
                                        <span className="font-sans font-semibold">[{sp.marks}]</span>
                                      </div>
                                    ))}
                                  </div>
                                )}

                                {/* Marking scheme display */}
                                {showSolutions && q.marking_scheme?.length > 0 && (
                                  <div className="mt-2 p-2 bg-indigo-50/60 rounded border border-indigo-100 font-sans text-[11px]">
                                    <span className="font-bold text-indigo-900 block mb-1">Marking Scheme:</span>
                                    <ul className="list-disc pl-4 space-y-0.5 text-indigo-800">
                                      {q.marking_scheme.map((rubric, rIdx) => (
                                        <li key={rIdx}>{rubric}</li>
                                      ))}
                                    </ul>
                                  </div>
                                )}
                              </div>
                              <span className="font-bold font-sans shrink-0">[{q.marks}]</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="bg-white p-16 rounded-xl border border-dashed border-slate-300 text-center">
              <svg className="w-12 h-12 text-slate-400 mx-auto mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
              <h3 className="text-sm font-semibold text-slate-700">No Paper Generated Yet</h3>
              <p className="text-xs text-slate-500 mt-1">Configure your blueprint on the left and click "Generate Complete Paper".</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
