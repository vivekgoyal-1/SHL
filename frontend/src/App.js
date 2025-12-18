import React, { useState } from 'react';
import { Search, AlertCircle, ExternalLink, Loader2, CheckCircle } from 'lucide-react';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState('');
  const [apiStatus, setApiStatus] = useState(null);

  // IMPORTANT: Replace this with your actual deployed API URL
  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  // Check API health on mount
  React.useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (response.ok) {
        setApiStatus('healthy');
      } else {
        setApiStatus('unhealthy');
      }
    } catch (err) {
      setApiStatus('unhealthy');
    }
  };

  const handleSearch = async () => {
    if (!query.trim()) {
      setError('Please enter a job description or query');
      return;
    }

    setLoading(true);
    setError('');
    setResults([]);

    try {
      const response = await fetch(`${API_BASE_URL}/recommend`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query.trim() }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      setResults(data.recommendations || []);
      
      if (!data.recommendations || data.recommendations.length === 0) {
        setError('No recommendations found. Try a different query.');
      }
    } catch (err) {
      setError('Failed to get recommendations. Please check if the API is running.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSearch();
    }
  };

  const sampleQueries = [
    "I am hiring for Java developers who can also collaborate effectively with my business teams.",
    "Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript.",
    "Need cognitive and personality tests for analyst position screening."
  ];

  const loadSampleQuery = (sampleQuery) => {
    setQuery(sampleQuery);
  };

  return (
    <div className="app">
      <div className="container">
        {/* Header */}
        <header className="header">
          <div className="header-content">
            <h1 className="title">SHL Assessment Finder</h1>
            <p className="subtitle">Find the right assessments for your hiring needs</p>
          </div>
          {apiStatus && (
            <div className={`api-status ${apiStatus}`}>
              {apiStatus === 'healthy' ? (
                <>
                  <CheckCircle size={16} />
                  <span>API Connected</span>
                </>
              ) : (
                <>
                  <AlertCircle size={16} />
                  <span>API Offline</span>
                </>
              )}
            </div>
          )}
        </header>

        {/* Search Section */}
        <div className="search-card">
          <label className="label">Job Description or Requirements</label>
          <div className="search-container">
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="E.g., I am hiring for Java developers who can also collaborate effectively with my business teams"
              className="textarea"
              rows="4"
            />
            <button
              onClick={handleSearch}
              disabled={loading || !query.trim()}
              className="search-button"
            >
              {loading ? (
                <>
                  <Loader2 className="icon spin" size={20} />
                  Searching...
                </>
              ) : (
                <>
                  <Search className="icon" size={20} />
                  Search Assessments
                </>
              )}
            </button>
          </div>
          
          {error && (
            <div className="error-message">
              <AlertCircle size={16} />
              {error}
            </div>
          )}

          {/* Sample Queries */}
          <div className="sample-queries">
            <span className="sample-label">Try these examples:</span>
            <div className="sample-buttons">
              {sampleQueries.map((sample, idx) => (
                <button
                  key={idx}
                  onClick={() => loadSampleQuery(sample)}
                  className="sample-button"
                >
                  {sample.substring(0, 50)}...
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Results Section */}
        {results.length > 0 && (
          <div className="results-card">
            <div className="results-header">
              <h2 className="results-title">
                Recommended Assessments
              </h2>
              <span className="results-count">{results.length} results</span>
            </div>
            <div className="results-table">
              <table>
                <thead>
                  <tr>
                    <th className="col-rank">#</th>
                    <th className="col-name">Assessment Name</th>
                    <th className="col-score">Relevance</th>
                    <th className="col-action">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((result, index) => (
                    <tr key={index}>
                      <td className="col-rank">{index + 1}</td>
                      <td className="col-name">
                        <div className="assessment-name">
                          {result.assessment_name}
                        </div>
                      </td>
                      <td className="col-score">
                        {result.score ? (
                          <div className="score-badge">
                            {(result.score * 100).toFixed(0)}%
                          </div>
                        ) : (
                          <span className="score-na">N/A</span>
                        )}
                      </td>
                      <td className="col-action">
                        <a
                          href={result.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="view-link"
                        >
                          View Details
                          <ExternalLink size={14} />
                        </a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Empty State */}
        {!loading && results.length === 0 && !error && (
          <div className="empty-state">
            <Search size={48} className="empty-icon" />
            <h3 className="empty-title">No results yet</h3>
            <p className="empty-text">
              Enter a job description to find relevant assessments
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;